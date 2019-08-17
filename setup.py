from setuptools import setup, find_packages

exec (open('gputools/version.py').read())

setup(name='gputools',
      version=__version__,
      description='OpenCL accelerated volume processing',
      url='https://github.com/maweigert/gputools',
      author='Martin Weigert',
      author_email='mweigert@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='science image-processing ',

      install_requires=["scipy",
                        "numpy>=1.11.0",
                        "pyopencl>=2016.1",
                        "configparser",
                        "reikna>=0.6.7"],

      extras_require={
          ':python_version<"3.0"': ["scikit-tensor",
                                    "ConfigParser",
                                    ],
          ':python_version>="3.0"': ["configparser",
                                     "scikit-tensor-py3",
                                     ],
      },

      package_data={"gputools":
                        ['core/kernels/*.cl',
                         'convolve/kernels/*.cl',
                         'denoise/kernels/*.cl',
                         'deconv/kernels/*.cl',
                         'noise/kernels/*.cl',
                         'fft/kernels/*.cl',
                         'transforms/kernels/*.cl',
                         ],

                    },

      entry_points={}
      )
