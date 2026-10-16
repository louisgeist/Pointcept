#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Build in place: python setup.py build_ext --inplace
# (setuptools and numpy.get_include() replace the original distutils / numpy.distutils,
# which recent setuptools and numpy no longer support.)
import numpy
from setuptools import Extension, setup

SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "grid_subsampling/grid_subsampling.cpp",
    "fps_subsampling/fps_subsampling.cpp",
    "wrapper.cpp",
]

module = Extension(
    name="cpp_subsampling",
    sources=SOURCES,
    extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    include_dirs=[numpy.get_include()],
)

setup(ext_modules=[module])
