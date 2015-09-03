# gputools

OpenCL acclerated volume processing in Python 

### Requirements 

A working OpenCL environment (check with clinfo).

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

####Convolutions

```python

import gputools

d = ones((128,200))
h = ones((17,17))
res = gputools.convolve(d,h)

```

```python

import gputools

d = ones((128,128,128))
h = ones(17)
res = gputools.convolve_sep3(d,h)

```

####Denoising

```python
...
res = gputools.denoise.nlm3(d,10.,3,4)
res = gputools.denoise.bilateral(d,3,10.)

```


####Deconvolution


```python
...
res = gputools.deconv.deconv_rl(d,h,2)
```


####Transforms
```python
gputools.transforms.scale(d,.2)
gputools.transforms.rotate(d,(64,64,64),(1,0,0),pi/4)
gputools.transforms.translate(d,10,20,30)
...
```

####fft

```python
gputools.fft(d)
gputools.fft(d, inverse = True)
```
