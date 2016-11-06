%%cython 
#!python
#cython: profile=True, linetrace=True, boundscheck=False

cimport cython
cimport numpy as np
import numpy as np
from sys import getsizeof

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.wraparound(False)
cdef euclidean_dists(np.ndarray[DTYPE_t, ndim=2, mode='c'] centers, 
                     np.ndarray[DTYPE_t, ndim=2, mode='c'] X):
    cdef:
        int N = X.shape[0]
        int D = X.shape[1]
        int K = centers.shape[0]
        double tmp, dist
        int n, k, d
        np.ndarray[DTYPE_t, ndim=2, mode='c'] Dists
    
    Dists = np.empty((K, N), dtype=np.float64, order='C')
    for k in xrange(K):
        for n in xrange(N):
            dist = 0.0
            for d in xrange(D):
                tmp = X[n, d] - centers[k, d]
                dist += tmp * tmp
            Dists[k, n] = dist
    return Dists

# Returns an index of the 2nd center that's d2-sampled 
@cython.wraparound(False)
cdef int d2_sample(np.ndarray[DTYPE_t, ndim=2, mode='c'] centers, 
                   np.ndarray[DTYPE_t, ndim=2, mode='c'] X) except? -1:
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] all_dists
    cdef int index
    all_dists = euclidean_dists(centers, X)
    # If sum is zero, the division yields zero too
    probs = np.divide(all_dists, all_dists.sum())
    probs = probs.reshape((probs.shape[1],), order='C')
    index = np.random.choice(np.arange(all_dists.shape[1], dtype=np.uint32), p=probs)
    return index

# Samples the next Voronoi cell to be split and 
# returns an index of the cell in centers
@cython.wraparound(False)
cdef np.intp_t sample_next_cell(int n_clusters, 
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] cell_q_errs):
    cdef np.intp_t center_ind
    cell_probs = np.divide(cell_q_errs, cell_q_errs.sum())
    cell_probs = cell_probs.reshape((cell_q_errs.shape[0],), order='C')
    center_ind = np.random.choice(np.arange(n_clusters, dtype=np.uint32), p=cell_probs)
    return center_ind

# Samples the next point which is a next center
# Returns an index of the point in X
@cython.wraparound(False)
cdef sample_next_center(np.ndarray[DTYPE_t, ndim=1, mode='c'] min_dists,
                         np.ndarray[np.int_t, ndim=1, mode='c'] argmin,
                         int next_cell_center_ind):
    cdef np.intp_t center_ind
    cdef np.ndarray[np.intp_t, ndim=2, mode='c'] next_cell_inds
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] next_cell_min_dists
    
    # Identifying points in a cell that is to be split
    next_cell_inds = np.asarray(np.where(argmin == next_cell_center_ind), 
                                      dtype=np.int, order='C')
    next_cell_min_dists = min_dists[next_cell_inds]

    # Sampling a point which will be a next center 
    probs = np.divide(next_cell_min_dists, next_cell_min_dists.sum())
    probs = probs.reshape((probs.shape[1],), order='C')
    center_ind = np.random.choice(np.arange(probs.shape[0]), p=probs)
    center_ind = next_cell_inds[0, center_ind]
    print "CENTER INDEX: ", center_ind
    return (center_ind, next_cell_inds)
    

# tree-k-means algorithm. Samples a new cell (i.e. centre) with d2-sampling 
# by quantization error.
# Additionally, computes a true (explicit) quantization error between 
# the points to the closest center (instead of the Voronoi cell), as well as
# computes number of distance evaluations.
'''
centers - k x d matrix of centers
min_dists - n, vector of distances to closest center
argmin - n, vector of the center that each point is closest to
cell_q_errs - k x 1 matrix of quantization errors of each cell
			  required for d^2-sampling of the next cell
cell_indices - matrix of indices used to find cell_q_errs
			   for the first 2 cells
new_dists - 2 x m matrix of distances from 2 centers in the operating cell
new_cell_inds - indices of points of the newly created cell
old_cell_inds - indices of points of the operating cell that is split
q_errs - implicit quantization error calculated from min_dists
true_q_error - true quantization error
num_dists_array - number of distance evaluations
'''
cdef ctree_means(n_clusters, X, out_folder):
    np.random.seed()
    cdef:
        int i
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        double init_q_err
        np.ndarray[DTYPE_t, ndim=2, mode='c'] centers = np.empty((n_clusters, n_features),
                                                                  dtype=np.float64, order='C')
        np.ndarray[DTYPE_t, ndim=1, mode='c'] min_dists
        np.ndarray[np.int_t, ndim=1, mode='c'] argmin
        np.ndarray[DTYPE_t, ndim=2, mode='c'] all_dists, q_errs,\
                                                true_q_error, cell_q_errs,\
                                                new_dists
        np.ndarray[np.int_t, ndim=2, mode='c'] num_dists_array, cell_indices,\
        									   new_cell_inds, old_cell_inds
        
    centers[0] = X[np.random.randint(n_samples)]
    centers[1] = X[d2_sample(centers[0].reshape(1, -1), X), :]
    
    all_dists = euclidean_dists(centers[0:2], X)   
    min_dists = np.min(all_dists, axis=0)   
    argmin = np.argmin(all_dists, axis=0)   
     
    init_q_err = min_dists.sum()
    q_errs = np.empty((n_clusters-1, 1), dtype=np.float64, order='C')
    q_errs[0] = init_q_err
    true_q_error = np.empty((n_clusters-1, 1), dtype=np.float64, order='C')
    true_q_error[0] = init_q_err
    num_dists_array = np.empty((n_clusters-1, 1), dtype=np.int, order='C')
    num_dists_array[0] = 2 * n_samples
    
    # Compute Cell Quantization Error for 2 initial cells
    cell_q_errs = np.zeros((n_clusters, 1), dtype=np.float64, order='C')
    cell_indices = np.asarray(np.where(argmin == 0), dtype=np.int, order='C')
    cell_q_errs[0] = min_dists[cell_indices].sum()
    cell_indices = np.asarray(np.where(argmin == 1), dtype=np.int, order='C')
    cell_q_errs[1] = min_dists[cell_indices].sum()

    
    print "=====" * 3, "START OF LOOP", "=====" * 3
    for i in range(2, n_clusters):
        print '|| i = ', i, '||',
        next_cell_center_ind = sample_next_cell(n_clusters, cell_q_errs)
        center_index, next_cell_inds = sample_next_center(min_dists,
                                                                argmin, next_cell_center_ind)
        centers[i] = X[center_index]
        
        # Updating existing variables for next iteration over the loop
        new_dists = euclidean_dists(np.vstack((centers[next_cell_center_ind], centers[i])),\
                                    np.take(X, next_cell_inds[0,:], axis=0))
        num_dists_array[i-1] = num_dists_array[i-2] + 2*next_cell_inds.shape[1]
        min_dists[next_cell_inds[0, :]] = np.min(new_dists, axis=0)
        old_cell_inds = np.take(next_cell_inds, np.where(np.argmin(new_dists, axis=0) == 0))
        new_cell_inds = np.take(next_cell_inds, np.where(np.argmin(new_dists, axis=0) == 1))
        argmin[new_cell_inds] = i
        
        q_errs[i-1] = min_dists.sum()
        all_dists = euclidean_dists(centers[0:i+1], X)
        true_q_error[i-1] = np.min(all_dists, axis=0).sum()
        cell_q_errs[next_cell_center_ind] = min_dists.take(old_cell_inds).sum()
        cell_q_errs[i] = min_dists.take(new_cell_inds).sum()
        print "True Quant Error:", true_q_error[i-1]
    np.save(folder + 'data/quant_errors', q_errs)
    np.save(folder + 'data/true_q_errors', true_q_error)
    np.save(folder + 'data/num_dists', num_dists_array)
    return centers

cpdef int main(k, input_file, out_folder): 
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] X
    X = np.load(input_file)
    print "X is", getsizeof(X) / 1000000.0, "MBs"
    Cs = ctree_means(k, X, out_folder)
    return 0
   
USGS = '/Users/Anuar_The_Great/Desktop/data/unsupervised/usgs/usgs.npy'
folder = '/Users/Anuar_The_Great/desktop/'
main(20, USGS, folder)