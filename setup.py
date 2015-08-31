
from setuptools import setup, find_packages

setup(name='gputools',
    version='0.1.1',
    description='',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=find_packages(),
    install_requires=["numpy","pyopencl","pyfft"],
    package_data={},
    entry_points = {}
)
