from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

ext_modules = [Extension("tree_k_means", ["tree_k_means.pyx"])]

setup(
  name = "Tree-K-means algorithm",
  author = "Anuar Yeraliyev",
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],        
  ext_modules = ext_modules
)
