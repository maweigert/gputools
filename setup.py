
from setuptools import setup, find_packages

exec(open('gputools/version.py').read())


setup(name='gputools',
    version=__version__,
    description='OpenCL accelerated volume processing',
    url='https://github.com/maweigert/gputools',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='BSD 3-Clause License',
    packages=find_packages(),

    install_requires=["numpy>=1.11.0",
                      "pyopencl>=2016.1",
                      "configparser",
                      "reikna>=0.6.7"],

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
