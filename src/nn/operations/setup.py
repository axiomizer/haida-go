from distutils.core import setup, Extension
import numpy


setup(name="nnops_ext",
      description="Extension module for fast neural network operations",
      ext_modules=[Extension("nnops_ext", ["nnops_ext.c"], include_dirs=[numpy.get_include()])])
