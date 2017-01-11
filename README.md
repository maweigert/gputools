# gputools - OpenCL accelerated volume processing in Python

This package aims to provide GPU accelerated implementations of common volume processing algorithms to the python ecosystem. 

* convolutions 
* denoising
* deconvolution
* synthetic noise
* ffts 
* transforms

### Requirements 

- python 2.7 / 3.5+
- a working OpenCL environment (check with clinfo).

### Installation

```
pip install gputools
```
check if basic stuff is working:

```
python -m gputools
```

### Usage

Docs are still to be done ;)

Most of the methods work on both numpy arrays or GPU memory objects (gputools.OCLArrays/OCLImage). The latter saving the memory transfer (which e.g. for simple convolutions accounts for the main run time)

####Convolutions

* 2D-3D convolutions
* separable convolutions
* fft based convolution
* spatially varying convolutions

```python

import gputools

d = ones((128,200))
h = ones((17,17))
res = gputools.convolve.convolve(d,h)

```

```python
d = ones((128,128,128))
h = ones(17)
res = gputools.convolve.convolve_sep3(d,h)

```

####Denoising

bilateral filter, non local means

```python
...
res = gputools.denoise.nlm3(d,10.,3,4)
res = gputools.denoise.bilateral(d,3,10.)

```


####Deconvolution

richardson lucy deconvolution 

```python
...
res = gputools.deconv.deconv_rl(d,h,2)
```

####Perlin noise

fast 2d and 3d perlin noise calculations

```python
gputools.perlin3(size = (256,256,256), scale = (10.,10.,10.))
```


####Transforms
scaling, translate, rotate, affine...


```python
gputools.transforms.scale(d,.2)
gputools.transforms.rotate(d,(64,64,64),(1,0,0),pi/4)
gputools.transforms.translate(d,10,20,30)
...
```

####fft
wraps around reikna

```python
gputools.fft(d)
gputools.fft(d, inverse = True)
```

### Configuration

The default OpenCL platform and device can be changed in the config file "~/.gputools" (create it if necessary)  
```
#~/.gputools

id_platform = 0
id_device = 1
```

### Troubleshooting

#### pyopencl: _cffi.so ImportError
If you see a
```
ImportError: _cffi.so: undefined symbol: clSVMFree
```
after importing gputools, this is most likely a problem of pyopencl being installed with an incorrent OpenCL version. 
Check the OpenCL version for your GPU with clinfo (e.g. 1.2):

```
clinfo | grep Version
```

and install pyopencl manually while enforcing your opencl version:

```
# uninstall pyopencl
pip uninstall pyopencl cffi
  
# get pyopencl source
git clone https://github.com/pyopencl/pyopencl.git
cd pyopencl
python configure.py
	
# add in siteconf.py the line
# CL_PRETEND_VERSION = "1.2"
echo 'CL_PRETEND_VERSION = "1.2"' >> siteconf.py

pip install .
```
where e.g. "1.2" is your version of OpenCL.




