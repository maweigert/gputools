
from setuptools import setup, find_packages

setup(name='gputools',
    version='0.1.1',
    description='',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=find_packages(),

    install_requires=["numpy>=1.10.0",
                      "pyopencl>=2016.1",
                      "pyfft"],

    package_data={"gputools":
                  ['core/kernels/*.cl',
                   'convolve/kernels/*.cl',
                   'denoise/kernels/*.cl',
                   'deconv/kernels/*.cl',
                   'noise/kernels/*.cl',
                   'transforms/kernels/*.cl',
                  ],
    },

    entry_points = {}
)
