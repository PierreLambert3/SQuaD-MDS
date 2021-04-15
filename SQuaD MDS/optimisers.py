import numpy as np
from sklearn.manifold import _barnes_hut_tsne
from numpy import sqrt
from gradients import compute_quartet_grads
import numba

def precompile():
    print("\nprecompiling the JIT functions for SQuaD MDS  ...")
    hparams = {
        'metric': 'relative rbf distance',
        'n iter': 10,
        'LR': 550,
        'exaggerate D': False,
        'stop exaggeration': 0.85
    }
    SQuaD_MDS(hparams, np.random.uniform(size = 30*3).reshape((30,3)).astype(np.float64), 20*np.random.uniform(size = 30*2).reshape((30,2)).astype(np.float64))
    print('done\n')



# does the whole gradient descent process for the basic algorithm
# target dimension = 2
def SQuaD_MDS(hparams, Xhd, Xld):
    N, M = Xhd.shape

    relative_rbf_distance = hparams["metric"] == "relative rbf distance" # transform the distances nonlinearly with 1 - exp(- (Dhd - min(Dhd))/(2*std(Dhd)) ) as described in the paper
    n_iter                = hparams["n iter"]
    LR                    = hparams["LR"]
    decay = np.exp(np.log(1e-3) / n_iter) # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    squared_D  = False
    stop_D_exa = 0
    if hparams["exaggerate D"]: # exaggeration of HD distances by taking them squared
        stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration
        squared_D  = True

    perms         = np.arange(N)
    batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    grad_acc      = np.ones((N, 2))
    Dhd_quartet   = np.zeros((6,))
    for i in range(n_iter):
        LR *= decay
        if i == stop_D_exa:
            squared_D = False

        np.random.shuffle(perms)
        update = fast_distance_scaling_update(N, Xld, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, relative_rbf_distance)

@numba.jit(nopython=True, fastmath = True)
def fast_distance_scaling_update(N, X_LD, LR, perms, batches_idxes, grad_acc,  Xhd, squared_D, Dhd_quartet, relative_rbf_distance):
    grad_acc.fill(0.)

    for batch_idx in batches_idxes:
        quartet     = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if squared_D: # during exaggeration: dont take the square root of the distances
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

        if relative_rbf_distance:
            quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet))
        else:
            Dhd_quartet  /= np.sum(Dhd_quartet)
            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)
            # quartet_grads = grad_all_4_optimised(LD_points, Dhd_quartet)
            # quartet_grads = grad_all_4_optimised(LD_points, Dhd_quartet)

        grad_acc[quartet[0], 0] += quartet_grads[0]
        grad_acc[quartet[0], 1] += quartet_grads[1]
        grad_acc[quartet[1], 0] += quartet_grads[2]
        grad_acc[quartet[1], 1] += quartet_grads[3]
        grad_acc[quartet[2], 0] += quartet_grads[4]
        grad_acc[quartet[2], 1] += quartet_grads[5]
        grad_acc[quartet[3], 0] += quartet_grads[6]
        grad_acc[quartet[3], 1] += quartet_grads[7]

    X_LD -= LR*grad_acc

@numba.jit(nopython=True)
def relative_rbf_dists(Dhd_quartet):
    rel_dists = np.exp((Dhd_quartet-np.min(Dhd_quartet)) / (-2*np.std(Dhd_quartet)))
    rel_dists = 1 - rel_dists
    rel_dists /= np.sum(rel_dists)
    return rel_dists















'''
The second part of this file is about mixing t-SNE gradients to the quartet-based distance gradients
'''

# adds some t-SNE gradients to the distance-preservation gradients, results in better Rnx(K) curves with a loss in distance preservation
def SQuaD_MDS_tsne(hparams, Xhd, Xld):
    N, M = Xhd.shape
    n_iter = hparams["n iter"]
    LR     = hparams["LR"]
    decay  = np.exp(np.log(1e-3) / n_iter)

    P = compute_P(Xhd, hparams)
    tsne_LR_multiplier = hparams["tsne LR multiplier"]
    tsne_exa_stop      = min(300, int(0.3*n_iter))


    # exaggeration of HD distances by taking them squared
    squared_D  = False
    stop_D_exa = 0
    if hparams["exaggerate D"]:
        stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration on the distances
        squared_D  = True

    perms         = np.arange(N)
    batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    Dhd_quartet   = np.zeros((6,))
    grad_acc      = np.ones((N, 2))
    for i in range(n_iter):
        LR *= decay
        if i == stop_D_exa:
            squared_D = False
        if i == tsne_exa_stop:
            P /= hparams["tsne exa"]

        np.random.shuffle(perms)
        update = fast_distance_scaling_update_tsne(N, P, Xld, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, tsne_LR_multiplier)

def fast_distance_scaling_update_tsne(N, P, X_LD, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, tsne_LR_multiplier):
    grad_acc.fill(0.)

    # step 1 : distance scaling gradients using the quartet method
    for batch_idx in batches_idxes:
        quartet = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if squared_D: # during exaggeration: dont take the square root of the distances
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

        Dhd_quartet  /= np.sum(Dhd_quartet)
        quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)


        grad_acc[quartet[0], 0] += quartet_grads[0]
        grad_acc[quartet[0], 1] += quartet_grads[1]
        grad_acc[quartet[1], 0] += quartet_grads[2]
        grad_acc[quartet[1], 1] += quartet_grads[3]
        grad_acc[quartet[2], 0] += quartet_grads[4]
        grad_acc[quartet[2], 1] += quartet_grads[5]
        grad_acc[quartet[3], 0] += quartet_grads[6]
        grad_acc[quartet[3], 1] += quartet_grads[7]

    # step 2 : t-SNE gradients, the code is taken for scikit-learn's github
    _, tsne_grad = KL_divergeance_BH(X_LD.ravel(), P, 1, N, 2, 0, False)

    X_LD -= tsne_LR_multiplier*LR*tsne_grad.reshape((N, 2))
    X_LD -= LR*grad_acc

# t-SNE's joint P matrix, with possible exaggeration
def compute_P(X, hparams):
    P = joint_P(X, hparams["PP"], "barnes_hut", 4)
    if hparams["tsne exa"] > 1.:
        P *= hparams["tsne exa"]
    return P

# t-SNE's joint P matrix
def joint_P(X, PP, method, N_jobs):
    from sklearn.neighbors import NearestNeighbors
    N, dim = X.shape
    n_neighbors = min(N - 1, int(3.*PP + 1))
    knn = NearestNeighbors(algorithm='auto', n_jobs=N_jobs, n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    D_nn = knn.kneighbors_graph(mode='distance')
    D_nn.data **= 2
    del knn
    P = joint_probabilities_nn(D_nn, PP)
    return P

def joint_probabilities_nn(D, target_PP):
    from sklearn.manifold._utils import _binary_search_perplexity
    from scipy.sparse import csr_matrix
    D.sort_indices()
    N = D.shape[0]
    D_data = D.data.reshape(N, -1)
    D_data = D_data.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(D_data, target_PP, 0)
    assert np.all(np.isfinite(conditional_P))
    P = csr_matrix((conditional_P.ravel(), D.indices, D.indptr), shape=(N, N))
    P = P + P.T
    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P
    assert np.all(np.abs(P.data) <= 1.0)
    return P

def KL_divergeance_BH(flat_X_LD, P, degrees_of_freedom, n_samples, n_components,
                      skip_num_points, compute_error,
                      angle=0.5, verbose=False,  num_threads=1):

    flat_X_LD = flat_X_LD.astype(np.float32, copy=False)
    X_embedded = flat_X_LD.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad
