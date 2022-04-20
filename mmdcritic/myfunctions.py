import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def mmd_squared(z: np.ndarray, x: np.ndarray, metric = 'rbf' ):
    '''
    Calculate the (normalized) Squared Maximum Mean Discrepancy.
    Input:
        z: set of m prototypes as a numpy.array of shape (m, # of features)
        x: dataset of n_samples istances as a numpy.array of shape (n_samples, # of features)
        
    Output:
        - (normalize) MMD score
        - tuple of internal "energy" q1 and interaction q3
    '''
    
    assert z.shape[1] == x.shape[1]
    
    q1 = pairwise_kernels(z, z, metric = metric).mean()
    # q2 = pairwise_kernels(x, x, metric = metric).mean()
    q3 = pairwise_kernels(z, x, metric = metric).mean()
    
    return 2*q3 - q1, (q1,q3)

def witness(z:np.ndarray ,x: np.ndarray, w: np.ndarray, metric = 'rbf'):
    
    '''
    Given the set z of prototypes and the samples set x, this function calculate the witness score in a point w.
    Input:
        - z: numpy.array (n_prototypes, n_features), set of prototypes
        - x: numpy.array (n_samples, n_features) sample set
        - w: numpy.array (n_critics, n_features), set of critics
    Output:
        - witness: float
    '''
    
    assert (x.shape[1] == w.shape[1])&(x.shape[1] == w.shape[1])

    return np.abs(
        pairwise_kernels(x,w,metric = metric).mean(axis = 0) - pairwise_kernels(z,w,metric = metric).mean(axis = 0)
    ).sum()

def critic_score(z:np.ndarray ,x: np.ndarray, w: np.ndarray, metric = 'rbf', alpha = 1):
    
    '''
    Calculate the critic score given the set z of prototypes and the samples set x, this function calculate the witness score in a point w.
    Input:
        - z: numpy.array (n_prototypes, n_features), set of prototypes
        - x: numpy.array (n_samples, n_features) sample set
        - w: numpy.array (n_critics, n_features)
    Output:
        - critic_score: float, sum of witness and regularizer term given by the log-determinant of critics
    
    '''
    
    assert (x.shape[1] == w.shape[1])&(x.shape[1] == w.shape[1])
    
    witness_scores = witness(z,x,w,metric = metric)
    
    return witness_scores + alpha*np.linalg.slogdet(pairwise_kernels(w,w,metric = metric))[1]

def find_prototypes(x: np.ndarray,m: int, record = False, metric = 'rbf'):
    
    '''
    Grid Search to find the prototypes in the dataset
    Input:
        - x: np.ndarray, set of data istances
        - m: number of prototypes. If >= x.shape[0], then the function returns x
        - record: bool, if True return the scores obtained during the grid-search
    Output:
        - set of prototypes
        - scored mmd_squared with the final set of prototypes
        - if record = True, the tracked mmd_squared_scores
    '''
    if record:
            mmd_tracker = []
            
    if m >= x.shape[0]:
        if record:
            mmd_tracker.append((
                    mmd_squared(x,x, metric = metric)[0]
                ,
                    x.shape[0])                    
                )
            return x, mmd_squared(x,x, metric = metric)[0], mmd_tracker
        else:
            return x, mmd_squared(x,x, metric = metric)[0]
    else:

        z = np.empty(shape = (0,x.shape[1]), dtype = x.dtype)
        
        #loop until z has m elements
        while z.shape[0] < min(m,x.shape[0]):
            candidate = max(x, key = lambda v:
                          mmd_squared(
                              np.append(z,v.reshape(1,-1), axis = 0),
                              x, metric = metric)[0]
                         )
            z = np.append(z, candidate.reshape(1,-1), axis = 0)
            if record:
                mmd_tracker.append((
                    mmd_squared(z,x, metric = metric)[0]
                ,
                    z.shape[0])                    
                )
                
        if record:
            return z, mmd_squared(z,x, metric = metric)[0], mmd_tracker
        else:
            return z, mmd_squared(z,x, metric = metric)[0] 
        
def find_critics(x: np.ndarray, z: np.ndarray, n: int, record = False, metric = 'rbf', alpha = 1):
    '''
    Grid Search to find the prototypes in the dataset
    Input:
        - x: numpy.ndarray (n_samples, n_features), set of data istances
        - z: numpy.ndarray (n_prototypes, n_features), set of prototypes
        - n: number of critics. If n >= n_samples - n_prototypes, then the function returns x minus-set z
        - record: bool, if True return the scores obtained during the grid-search
    Output:
        - set of critics
        - critic_score w.r.t. the final set of prototypes
        - if record = True, the tracked critic score
    '''
    
    if record:
            critic_tracker = []
            
    if n >= (x.shape[0] - z.shape[0]):
        
        critics = x
        for prototype in z:
            critics = critics[np.all(~np.equal(critics, prototype), axis=1)]
        
        if record:
            critic_tracker.append((
                    critic_score(z,x, w,
                                metric = metric, alpha = alpha
                                )
                ,
                    (x.shape[0] - z.shape[0])
            )                    
                )
            return critics, critic_score(z,x, critics,
                                metric = metric, alpha = alpha
                                ), critic_tracker
        else:
            return critics, critic_score(z,x, critics,
                                metric = metric, alpha = alpha
                                )
    else:
        
        critics_universe = x
        for prototype in z:
            critics_universe = critics_universe[np.all(~np.equal(critics_universe, prototype), axis=1)]
            
        critics = np.empty(shape = (0,x.shape[1]), dtype = x.dtype)
        
        while critics.shape[0] < min(n,x.shape[0]):
            candidate = max(
                critics_universe, key = lambda v:
                critic_score(z ,x, np.append(critics,v.reshape(1,-1), axis = 0),
                            metric = metric, alpha = alpha
                            )
            )
            critics = np.append(critics, candidate.reshape(1,-1), axis = 0)
            
            if record:
                critic_tracker.append((
                    critic_score(z ,x,critics,
                                metric = metric, alpha = alpha
                                )
                ,
                    critics.shape[0])                    
                )
            
        if record:
            return critics, critic_score(z ,x,critics,
                                        metric = metric, alpha = alpha
                                        ), critic_tracker
        else:
            return critics, critic_score(z ,x,critics,
                                        metric = metric, alpha = alpha
                                        )
def mmd_critic(x: np.ndarray, n_prototypes: int, n_critics: int, record = False, metric = 'rbf', alpha = 1):
    
    '''
    Apply MMD Critic algorithm.
    Input:
        - x: numpy.array (n_samples, n_features), sample set
        - m: int, number of prototypes
        - n: int, number of critics
    Output:
        - set of prototypes (n_prototypes, n_features)
        - set of critics (n_critics, n_features)
    '''
    prototypes = find_prototypes(x, n_prototypes, record = record, metric = metric)
    critics = find_critics(x, prototypes[0], n_critics, record = record, metric = metric, alpha = alpha)
    
    return prototypes, critics