from distutils.core import setup, Extension
import numpy


setup(name="nn_ext",
      description="Extension module for fast neural network operations",
      ext_modules=[Extension("nn_ext", ["nn_ext.c"], include_dirs=[numpy.get_include()])])
