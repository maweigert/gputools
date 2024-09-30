from setuptools import setup, find_packages

exec (open('gputools/version.py').read())

setup(name='gputools',
      version=__version__,
      description='OpenCL accelerated volume processing',
      url='https://github.com/maweigert/gputools',
      author='Martin Weigert',
      author_email='martin.weigert@epfl.ch',
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
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ],
      keywords='science image-processing ',

      install_requires=[
          "six",
          "scipy",
          "numpy",
          "pyopencl>=2016.1",
          "configparser",
          "reikna>=0.8.0"],

      extras_require={
          ':python_version<"3.0"': ["scikit-tensor",
                                    "ConfigParser",
                                    ],
          ':python_version>="3.0"': ["configparser",
                                    # "scikit-tensor-py3",
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
