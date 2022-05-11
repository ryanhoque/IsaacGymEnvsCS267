from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='myextension_cpp',
        ext_modules=[cpp_extension.CUDAExtension('myextension_cpp', ['myextension.cpp', 'myextensioncuda.cu'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
