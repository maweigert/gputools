"""
an adaptation of pyopencl's reduction kernel for weighted avarages

like sum(a*b)

mweigert@mpi-cbg.de

"""

from __future__ import division
from __future__ import absolute_import
from six.moves import zip

__copyright__ = "Copyright (C) 2010 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
Based on code/ideas by Mark Harris <mharris@nvidia.com>.
None of the original source code remains.
"""

import pyopencl as cl
from pyopencl.tools import (
    context_dependent_memoize,
    dtype_to_ctype, KernelTemplateBase,
    _process_code_for_macro)
import numpy as np
from gputools import get_device


import sys


# {{{ kernel source

KERNEL = r"""//CL//

    <%
    inds = range(len(map_exprs))
    %>
    #define GROUP_SIZE ${group_size}
    % for i,m in enumerate(map_exprs):
    #define READ_AND_MAP_${i}(i) (${m})
    % endfor

    #define REDUCE(a, b) (${reduce_expr})
    % if double_support:
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
    % endif
    #include <pyopencl-complex.h>
    ${preamble}
    typedef ${out_type} out_type;
    __kernel void ${name}(
    % for i in inds:
      __global out_type *out__base_${i},
    % endfor
      long out__offset, ${arguments},
      unsigned int seq_count, unsigned int n)
    {

        % for i in inds:
            __global out_type *out_${i} = (__global out_type *) (
            (__global char *) out__base_${i} + out__offset);
        % endfor

        ${arg_prep}

        % for i in inds:
        __local out_type ldata_${i}[GROUP_SIZE];
        out_type acc_${i} = ${neutral};
        % endfor


        unsigned int lid = get_local_id(0);
        unsigned int i = get_group_id(0)*GROUP_SIZE*seq_count + lid;

        for (unsigned s = 0; s < seq_count; ++s)
        {
          if (i >= n)
            break;

        % for i in inds:
          acc_${i} = REDUCE(acc_${i}, READ_AND_MAP_${i}(i));
        % endfor

          i += GROUP_SIZE;
        }

        % for i in inds:
          ldata_${i}[lid] = acc_${i};
        % endfor


        <%
          cur_size = group_size
        %>
        % while cur_size > 1:
            barrier(CLK_LOCAL_MEM_FENCE);
            <%
            new_size = cur_size // 2
            assert new_size * 2 == cur_size
            %>
            if (lid < ${new_size})
            {

                % for i in inds:

                  ldata_${i}[lid] = REDUCE(
                  ldata_${i}[lid],
                  ldata_${i}[lid + ${new_size}]);

                % endfor

            }
            <% cur_size = new_size %>
        % endwhile
        if (lid == 0) {

           % for i in inds:
                out_${i}[get_group_id(0)] = ldata_${i}[0];

            % endfor


    }
    }
"""

KERNEL2 = r"""//CL//
    #define GROUP_SIZE ${group_size}
    % for i,m in map_exprs:
    #define READ_AND_MAP_${i}(i) (${m})
    % endfor
    #define REDUCE(a, b) (${reduce_expr})
    % if double_support:
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
    % endif
    #include <pyopencl-complex.h>
    ${preamble}
    typedef ${out_type} out_type;
    __kernel void ${name}(
      __global out_type *out__base,
      __global out_type *out__base2,
      long out__offset, ${arguments},
      unsigned int seq_count, unsigned int n)
    {
        __global out_type *out = (__global out_type *) (
            (__global char *) out__base + out__offset);
        __global out_type *out2 = (__global out_type *) (
            (__global char *) out__base2 + out__offset);

        ${arg_prep}
        __local out_type ldata[GROUP_SIZE];
        __local out_type ldata2[GROUP_SIZE];

        unsigned int lid = get_local_id(0);
        unsigned int i = get_group_id(0)*GROUP_SIZE*seq_count + lid;
        out_type acc = ${neutral};
        out_type acc2 = ${neutral};

        for (unsigned s = 0; s < seq_count; ++s)
        {
          if (i >= n)
            break;
          acc = REDUCE(acc, READ_AND_MAP(i));
          acc2 = REDUCE(acc2, READ_AND_MAP2(i));

          i += GROUP_SIZE;
        }
        ldata[lid] = acc;
        ldata2[lid] = acc2;

        <%
          cur_size = group_size
        %>
        % while cur_size > 1:
            barrier(CLK_LOCAL_MEM_FENCE);
            <%
            new_size = cur_size // 2
            assert new_size * 2 == cur_size
            %>
            if (lid < ${new_size})
            {
                ldata[lid] = REDUCE(
                  ldata[lid],
                  ldata[lid + ${new_size}]);
                ldata2[lid] = REDUCE(
                  ldata2[lid],
                  ldata2[lid + ${new_size}]);

            }
            <% cur_size = new_size %>
        % endwhile
        if (lid == 0) {
            out[get_group_id(0)] = ldata[0];
           out2[get_group_id(0)] = ldata2[0];
}
    }
    """


def _get_reduction_source(
        ctx, out_type, out_type_size,
        neutral, reduce_expr, map_exprs, parsed_args,
        name="reduce_kernel", preamble="", arg_prep="",
        device=None, max_group_size=None):


    if device is not None:
        devices = [device]
    else:
        devices = ctx.devices

    # {{{ compute group size

    def get_dev_group_size(device):
        # dirty fix for the RV770 boards
        max_work_group_size = device.max_work_group_size
        if "RV770" in device.name:
            max_work_group_size = 64

        # compute lmem limit
        from pytools import div_ceil
        lmem_wg_size = div_ceil(max_work_group_size, out_type_size)
        result = min(max_work_group_size, lmem_wg_size)

        # round down to power of 2
        from pyopencl.tools import bitlog2
        return 2**bitlog2(result)

    group_size = min(get_dev_group_size(dev) for dev in devices)



    if max_group_size is not None:
        group_size = min(max_group_size, group_size)

    # }}}


    from mako.template import Template
    from pytools import all
    from pyopencl.characterize import has_double_support
    src = str(Template(KERNEL).render(
        out_type=out_type,
        arguments=", ".join(arg.declarator() for arg in parsed_args),
        group_size=group_size,
        neutral=neutral,
        reduce_expr=_process_code_for_macro(reduce_expr),
        map_exprs=[_process_code_for_macro(m) for m in map_exprs],
        name=name,
        preamble=preamble,
        arg_prep=arg_prep,
        double_support=all(has_double_support(dev) for dev in devices),
    ))



    #    sys.exit()

    from pytools import Record

    class ReductionInfo(Record):
        pass

    return ReductionInfo(
        context=ctx,
        source=src,
        group_size=group_size)


def get_reduction_kernel(stage,
                         ctx, dtype_out,
                         neutral, reduce_expr, arguments=None,
                         name="reduce_kernel", preamble="",
                         map_exprs = [None],
                         device=None, options=[], max_group_size=None):

    for i, m in enumerate(map_exprs):
        if m is None:
            if stage==2:
                map_expr[i] = "pyopencl_reduction_inp[i]"
            else:
                map_expr[i] = "in[i]"


    from pyopencl.tools import (
        parse_arg_list, get_arg_list_scalar_arg_dtypes,
        get_arg_offset_adjuster_code, VectorArg)

    arg_prep = ""
    if stage==1 and arguments is not None:
        arguments = parse_arg_list(arguments, with_offset=True)
        arg_prep = get_arg_offset_adjuster_code(arguments)

    if stage==2 and arguments is not None:
        arguments = parse_arg_list(arguments)
        arguments = (
            [VectorArg(dtype_out, "pyopencl_reduction_inp")]
            +arguments)


    inf = _get_reduction_source(
        ctx, dtype_to_ctype(dtype_out), dtype_out.itemsize,
        neutral, reduce_expr, map_exprs, arguments,
        name, preamble, arg_prep, device, max_group_size)

    inf.program = cl.Program(ctx, inf.source)
    inf.program.build(options)
    inf.kernel = getattr(inf.program, name)

    inf.arg_types = arguments

    inf.kernel.set_scalar_arg_dtypes(
        [None, ]*len(map_exprs)+[np.int64]
        +get_arg_list_scalar_arg_dtypes(inf.arg_types)
        +[np.uint32]*2)

    return inf


# }}}


# {{{ main reduction kernel

class OCLReductionKernel2:
    """
    simultanins reduction of a weighted sum of two buffers

    example:

        k = OCLReductionKernel2(np.float32,
                neutral="0",reduce_expr="a+b",
                map_expr1 ="x[i]",
                map_expr2 ="x[i]*y[i]",
                arguments="__global float *x,__global float *y")

        k(a,b, out1 = out1, out2 = out2)



    """

    def __init__(self, dtype_out,
                 neutral, reduce_expr, arguments=None,
                 map_exprs=[None],
                 name="reduce_kernel", options=[], preamble=""):

        ctx = get_device().context
        dtype_out = self.dtype_out = np.dtype(dtype_out)

        max_group_size = None
        trip_count = 0

        self.n_exprs = len(map_exprs)
        assert self.n_exprs>0

        while True:
            self.stage_1_inf = get_reduction_kernel(1, ctx,
                                                    dtype_out,
                                                    neutral, reduce_expr, arguments,
                                                    name=name+"_stage1", options=options, preamble=preamble,
                                                    map_exprs=map_exprs,
                                                    max_group_size=max_group_size)


            kernel_max_wg_size = self.stage_1_inf.kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE,
                ctx.devices[0])

            if self.stage_1_inf.group_size<=kernel_max_wg_size:
                break
            else:
                max_group_size = kernel_max_wg_size

            trip_count += 1
            assert trip_count<=2

        self.stage_2_inf = get_reduction_kernel(2, ctx,
                                dtype_out,
                                neutral, reduce_expr, arguments=arguments,
                                name=name+"_stage2", options=options,
                                map_exprs=map_exprs,
                                preamble=preamble,
                                max_group_size=max_group_size)

        from pytools import any
        from pyopencl.tools import VectorArg
        assert any(
            isinstance(arg_tp, VectorArg)
            for arg_tp in self.stage_1_inf.arg_types), \
            "ReductionKernel can only be used with functions " \
            "that have at least one vector argument"

    def __call__(self, *args, **kwargs):
        MAX_GROUP_COUNT = 1024  # noqa
        SMALL_SEQ_COUNT = 4  # noqa

        from pyopencl.array import empty

        stage_inf = self.stage_1_inf

        queue = kwargs.pop("queue", None)
        wait_for = kwargs.pop("wait_for", None)
        return_event = kwargs.pop("return_event", False)

        outs = kwargs.pop("outs", [None]*self.n_exprs)

        if kwargs:
            raise TypeError("invalid keyword argument to reduction kernel")

        stage1_args = args


        while True:
            invocation_args = []
            vectors = []

            from pyopencl.tools import VectorArg
            for arg, arg_tp in zip(args, stage_inf.arg_types):
                if isinstance(arg_tp, VectorArg):
                    if not arg.flags.forc:
                        raise RuntimeError("ReductionKernel cannot "
                                           "deal with non-contiguous arrays")

                    vectors.append(arg)
                    invocation_args.append(arg.base_data)
                    if arg_tp.with_offset:
                        invocation_args.append(arg.offset)
                else:
                    invocation_args.append(arg)

            repr_vec = vectors[0]
            sz = repr_vec.size


            if queue is not None:
                use_queue = queue
            else:
                use_queue = repr_vec.queue

            if sz<=stage_inf.group_size*SMALL_SEQ_COUNT*MAX_GROUP_COUNT:
                total_group_size = SMALL_SEQ_COUNT*stage_inf.group_size
                group_count = (sz+total_group_size-1)//total_group_size
                seq_count = SMALL_SEQ_COUNT
            else:
                group_count = MAX_GROUP_COUNT
                macrogroup_size = group_count*stage_inf.group_size
                seq_count = (sz+macrogroup_size-1)//macrogroup_size

            if group_count==1:
                results = [empty(use_queue,
                                 (), self.dtype_out,
                                 allocator=repr_vec.allocator) if out is None else out for out in outs]
            else:
                results = [empty(use_queue,
                                 (), self.dtype_out,
                                 allocator=repr_vec.allocator) for out in outs]

            print seq_count, sz

            last_evt = stage_inf.kernel(
                use_queue,
                (group_count*stage_inf.group_size,),
                (stage_inf.group_size,),
                *([r.base_data for r in results]+[results[0].offset,]
                  +invocation_args+[seq_count, sz]),
                **dict(wait_for=wait_for))
            wait_for = [last_evt]

            if group_count==1:
                if return_event:
                    return results, last_evt
                else:
                    return results
            else:
                stage_inf = self.stage_2_inf
                #args = tuple(results)+stage1_args
                args = (results[0],)+stage1_args


if __name__=='__main__':
    from gputools import OCLArray


    def time_multi(N, nargs, niter =100):
        map_exprs=["%s*x%s[i]"%(i,i) for i in xrange(nargs)]
        arguments = ",".join("__global float *x%s"%i for i in xrange(nargs))

        k = OCLReductionKernel2(np.float32,
                            neutral="0", reduce_expr="a+b",
                            map_exprs=map_exprs,
                            arguments=arguments)

        ins = [OCLArray.from_array(np.ones(N,np.float32)) for _ in xrange(len(map_exprs))]
        outs = [OCLArray.empty(1,np.float32) for _ in xrange(len(map_exprs))]

        from time import time
        t = time()
        for _ in xrange(niter):
            k(*ins, outs = outs)
        get_device().queue.finish()
        t = (time()-t)/niter
        print "multi reduction: result =", [float(out.get()) for out in outs]
        print "multi reduction:\t\t%.2f ms"%(1000*t)
        return t



    def time_simple(N, nargs, niter =100):
        from gputools import OCLReductionKernel

        map_exprs=["%s*x[i]"%i for i in xrange(nargs)]


        ks = [OCLReductionKernel(np.float32,
                            neutral="0", reduce_expr="a+b",
                            map_expr="%s*x[i]"%i,
                            arguments="__global float *x") for i in xrange(len(map_exprs))]

        ins = [OCLArray.from_array(np.ones(N,np.float32)) for _ in xrange(len(map_exprs))]
        outs = [OCLArray.empty(1,np.float32) for _ in xrange(len(map_exprs))]

        from time import time
        t = time()
        for _ in xrange(niter):
            for k,inn,out in zip(ks,ins,outs):
                k(inn, out = out)
        get_device().queue.finish()
        t = (time()-t)/niter
        print "simple reduction: result =", [float(out.get()) for out in outs]
        print "simple reduction:\t\t%.2f ms"%(1000*t)
        return t



    from gputools import OCLReductionKernel
    k1 = OCLReductionKernel(np.float32,
                            neutral="0", reduce_expr="a+b",
                            map_expr="x[i]",
                            arguments="__global float *x")


    k2 = OCLReductionKernel2(np.float32,
                            neutral="0", reduce_expr="a+b",
                            map_exprs=["x[i]"],
                            arguments="__global float *x")

    a = OCLArray.from_array(np.ones((256,256),np.float32))


    # print k1(a)
    print k2(a)


    #
    # time_simple(512,5)
    # time_multi(512,5)
    #
    # ts1 = 1000*np.array([time_simple(512,n) for n in ns])
    # ts2 = 1000*np.array([time_multi(512,n) for n in ns])