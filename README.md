# gputools

OpenCL acclerated volume processing in Python 

### Requirements 

- python 2 (yet)
- a working OpenCL environment (check with clinfo).

### Installation

```
pip install --user git+https://github.com/maweigert/gputools
```
check if basic stuff is working:

```
python -m gputools
```

### Usage

Docs are still to be done ;)

Most of the methods work on both numpy arrays or GPU memory objects (gputools.OCLArrays/OCLImage). The latter saving the memory transfer (which e.g. for simple convolutions accounts for the main run time)

####Convolutions

1D-3D convolutions/seperable convolutions/fft based convolution

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


####Transforms
scaling, translate, rotate, affine...


```python
gputools.transforms.scale(d,.2)
gputools.transforms.rotate(d,(64,64,64),(1,0,0),pi/4)
gputools.transforms.translate(d,10,20,30)
...
```

####fft
wraps around pyfft 

```python
gputools.fft(d)
gputools.fft(d, inverse = True)
```
