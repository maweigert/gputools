# gputools - OpenCL accelerated volume processing in Python

This package aims to provide GPU accelerated implementations of common volume processing algorithms to the python ecosystem, such as  

* convolutions 
* denoising
* synthetic noise
* ffts (simple wrapper around [reikna](https://github.com/fjarri/reikna))
* affine transforms

via OpenCL and the excellent [pyopencl](https://documen.tician.de/pyopencl/) bindings.

Some examples of processing tasks and their respective runtime (`tests/benchmark/benchmark.py`):

Task | Image Size/type | CPU[1] | GPU[2] | GPU (w/o transfer)[3]
----|----| ----| ---- | ----
Mean filter 7x7x7| (128, 1024, 1024) uint8 | 2627 ms | 99 ms | 24 ms
Median filter 3x3x3| (128, 1024, 1024) uint8 | 59750 ms | 346 ms | 252 ms
Gaussian filter 5x5x5| (128, 1024, 1024) float32 | 9594 ms | 416 ms | 101 ms
Zoom/Scale 2x2x2| (128, 1024, 1024) uint8 | 61829 ms | 466 ms | -
NLM denoising| (64, 256, 256) float32 | 52736 ms | 742 ms | -
FFT (pow2) | (128, 1024, 1024) complex64 | 13831 ms | 615 ms | 69 ms

	[1] Xeon(R) CPU E5-2630 v4 using numpy/scipy functions
	[2] NVidia Titan X using gputools
	[3] as [2] but without CPU->GPU->CPU transfer
	
### Requirements 

- python 2.7 / 3.5+
- a working OpenCL environment (check with clinfo).

### Installation

```
pip install gputools
```
Or the developmental version:

```
pip install git+https://github.com/maweigert/gputools@develop
```

Check if basic stuff is working:

```
python -m gputools
```

#### Troubleshooting 

If you experience installation issues in Windows, this might be due to `pyopencl` not  
being properly installed. 
1. Download the correct [pyopencl wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) for your platform
2. Install it via `pip install pyopencl‑2020.2.2+cl21‑cp38‑cp38‑win_amd64.whl` 

### Usage

Docs are still to be done ;)

Most of the methods work on both numpy arrays or GPU memory objects (gputools.OCLArrays/OCLImage). The latter saving the memory transfer (which e.g. for simple convolutions accounts for the main run time)

#### Convolutions

* 2D-3D convolutions
* separable convolutions
* fft based convolution
* spatially varying convolutions

```python

import gputools

d = np.zeros((128,128), np.float32)
d[64,64] = 1.
h = np.ones((17,17))
res = gputools.convolve(d,h)

```

```python
d = np.zeros((128,128,128), np.float32)
d[64,64,64] = 1.
hx,hy,hz = np.ones(7),np.ones(9),np.ones(11)
res = gputools.convolve_sep3(d,hx,hy,hz)

```

#### Denoising

bilateral filter, non local means

```python
...
d = np.zeros((128,128,128, np.float32))
d[50:78,50:78,50:78:2] = 4.
d = d+np.random.normal(0,1,d.shape)
res_nlm = gputools.denoise.nlm3(d,2.,2,3)
res_bilat = gputools.denoise.bilateral3(d,3,4.)

```

#### Perlin noise

fast 2d and 3d perlin noise calculations

```python
gputools.perlin3(size = (256,256,256), scale = (10.,10.,10.))
```


#### Transforms
scaling, translate, rotate, affine...


```python
gputools.transforms.scale(d,.2)
gputools.transforms.rotate(d,axis = (64,64,64),angle = .3)
gputools.transforms.shift(d,(10,20,30))
...
```

#### fft
wraps around reikna

```python
gputools.fft(d)
gputools.fft(d, inverse = True)
```

### Configuration

Some configuration data (e.g. the default OpenCL platform and devic) can be changed in the config file "~/.gputools" (create it if necessary)  
```
#~/.gputools

id_platform = 0
id_device = 1
```
See 
```python
gputools.config.defaults
```
for available keys and their defaults.

Alternatively, the used OpenCL Device can be set via the environment variables `gputools_id_device`,  `gputools_id_platform`, and `gputools_use_gpu` (variables present in the config file will take precendence, however).


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




