import numpy as np

def get_top_k(k, X):
    X = np.abs(X)
    return np.argpartition(X, -k, axis=1)[:, -k:]

def top_k_overall(k, x, y):
    # x and y are nxk arrays
    # we want to return a n-dimensional array where each entry is the consistency between x and y
    res = np.zeros([x.shape[0],k])
    for i in range(x.shape[0]):
        for j in range(k):
            if x[i,j] in y[i]:
                res[i,j] = 1
            else:
                res[i,j] = 0
    frac_right = np.sum(res, axis=1)/k
    return sum(frac_right)/len(frac_right)

def top_k_sa(k, x, y, signs_x, signs_y):
    # X and Y are nxk arrays
    # we want to return a n-dimensional array where n[i] is the frac. of x[i] and y[i] 
    # that agree and have same sign
    # step 1 just checks whether X's top-K features have the same sign in Y. If not, indices of x are set to 0
    #     so that in top_k_overall they will not be counted
    limited_sx = signs_x[np.arange(x.shape[0])[:,None], x]
    limited_sy = signs_y[np.arange(x.shape[0])[:,None], x]
    
    x = np.where(limited_sx == limited_sy, x, -1)
    return top_k_overall(k, x, y)

def top_k_cdc(k, x, y, signs_x, signs_y):
    ''' Returns CONSISTENT direction of contribution, i.e., 1 = total agreement '''
    limited_sx = signs_x[np.arange(x.shape[0])[:,None], x]
    limited_sy = signs_y[np.arange(x.shape[0])[:,None], x]
    x = (limited_sx == limited_sy)#  + (-1) * (limited_sx != limited_sy)
    limited_sx = signs_x[np.arange(y.shape[0])[:,None], y]
    limited_sy = signs_y[np.arange(y.shape[0])[:,None], y]
    y = (limited_sx == limited_sy)
    scores = np.all(x == 1, axis=1) & np.all(y == 1, axis=1)
    return sum(scores)/len(scores)

def top_k_ssa(k, x, y, signs_x, signs_y):
    ''' Returns Signed Set *Agreement* (i.e., 1 means perfect agreement)'''
    # First, we need to satisfy CDC, so check that first: 
    limited_sx = signs_x[np.arange(x.shape[0])[:,None], x]
    limited_sy = signs_y[np.arange(x.shape[0])[:,None], x]
    xeq = (limited_sx == limited_sy)#  + (-1) * (limited_sx != limited_sy)
    limited_sx = signs_x[np.arange(y.shape[0])[:,None], y]
    limited_sy = signs_y[np.arange(y.shape[0])[:,None], y]
    yeq = (limited_sx == limited_sy)
    cdc = np.logical_and(np.all(xeq == 1, axis=1), np.all(yeq == 1, axis=1))

    # Next, we need to know whether X and Y have the same top-k features
    res = np.zeros([x.shape[0],k])
    for i in range(x.shape[0]):
        for j in range(k):
            if x[i,j] in y[i]:
                res[i,j] = 1
            else:
                res[i,j] = 0
    frac_right = np.sum(res, axis=1)/k
    frac_right = np.where(frac_right == 1, 1, 0)

    # now, frac_right is 1 if x and y have the same top-K features, and cdc is 1 if all of X's top-K features have the same sign in Y
    # so, we need to return 1 if both are true and 0 otherwise
    scores = np.logical_and(frac_right, cdc)
    return sum(scores)/len(scores)