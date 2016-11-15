# Efficient K-means seeding using binary tree

Tree K-means implementation of seeding before Lloyd's algorithm.

Tree K-means implementation runs in *O(log(n)kd)* time in comparison to K-means++ which runs in *O(nkd)* while at the same time not requiring significant memory overhead. This is a huge computational advancement as seeding for large datasets is extremely computationally expensive.

Written in Cython using Numpy dependency and compiled using distutils. You can change the dataset in k_tree_means.pyx file and modify compiler directives to Cython in setup.py.

Then one can compile it in command line *python setup.py build_ext --inplace* within the directory and your package is built.

I also built the package and commited it, you can see it in *build* folder, just for the sake of an example.
