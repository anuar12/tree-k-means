Tree K-means implementation of seeding before Lloyd's algorithm.

Written in Cython using Numpy dependency and compiled using distutils. You can change the dataset in k_tree_means.pyx file and modify compiler directives to Cython in setup.py.

Then one can compile it in command line *python setup.py build_ext --inplace* within the directory and your package is built.
