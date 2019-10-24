import numpy as np


def double_vec(q):
    q_d = np.zeros(len(q)*2)
    q_d[0::2] = q
    q_d[1::2] = q
    return q_d

def index_iter(n_dim, n_discr):
    """
    iterates through index tuples of n_dim dimentional cube with n_discr elements on each axis
    """
    powers = np.array([n_discr**i for i in range(n_dim)])
    powers_inc = n_discr*powers
    for i in range(n_discr**n_dim):
        yield ((i % powers_inc - i % powers)/powers).astype(np.int32)

        
def radial_fun(r,ri,sigma=0.15):
    return np.exp(-.5*(np.linalg.norm(r - ri)/sigma)**2)

class TaskSetup:
    def __init__(self, n_dim=3, n_discr=10, n_p=9, seed=42):
        """
        n_dim: dimentionality of the unit cube (equation domain), should be 3 for now
        n_discr: discretization of the unit cube
        n_p: number of radial functions
        """
        self.n_dim = n_dim
        self.n_discr = n_discr
        self.n_p = n_p
        self.discr_row = np.linspace(0, 1, n_discr)
        if seed is not None:
            np.random.seed(seed)
        
        # generate permability field
        self.ri = np.random.rand(n_p, n_dim) #centers of radial functions
        self.x = np.random.rand(n_p) #true parameters
        
        self.bi = np.zeros([n_p] + [n_discr]*n_dim) # radial functions 
        
        
        for i in range(n_p):
            for ind in index_iter(n_dim, n_discr):
                r = np.array([self.discr_row[i] for i in ind])
                val = radial_fun(r, self.ri[i])
                arr_ind = tuple(((k) for k in ind))
                self.bi[i][arr_ind] = val 
    
        self.K = self.eval_perm_field(self.x) # true permability field
        
        # generate sources
        self.Q = np.zeros([n_discr]*n_dim)
        for ind in index_iter(n_dim, n_discr):
            r = np.array([self.discr_row[i] for i in ind])
            val = 0
            for center in index_iter(n_dim, 2):
                center = center*0.4 + 0.3 # place sources in the corners 
                val += radial_fun(r, center, 0.05)
            arr_ind = tuple(((k) for k in ind))
            self.Q[arr_ind] = val 

        self.G = gradient_matrix(self.n_discr)
        self.L = laplacian_matrix(self.n_discr)
    
    def eval_perm_field(self, x):
        return np.tensordot(self.bi, x, ((0),(0)))
    
    def eval_u_x(self, x, u):
        K = self.eval_perm_field(x)
        phi = [K*grad for grad in np.gradient(u)]
        return sum([np.gradient(elem, axis=i) for i, elem in enumerate(phi)]) + self.Q
    
    def basis_projection_solution(self, x, U):
        """
        U supposed to be the list of previously calculated u values of shape (n_discr x n_discr x ...)
        """
        K = self.eval_perm_field(x)
        A = []
        for u in U:
            phi = [K*grad for grad in np.gradient(u)]
            new_vec = sum([np.gradient(elem, axis=i) for i, elem in enumerate(phi)])
            A.append(new_vec.flatten())
        U = np.vstack(U)
#         A = U.T @ np.vstack(A).T
#         b = U @ self.Q.flatten()
        A = np.vstack(A).T
        b = self.Q.flatten()
        coeffs = np.linalg.lstsq(A, b)
        return coeffs

def orthonormal(vectors):
    """
    turn list of vectors to basis vectors
    """
    Q = np.vstack(vectors).T
    Q, _ = np.linalg.qr(Q)
    vectors = (Q / np.linalg.norm(Q, axis=0)).T
    return vectors

def gradient_matrix(N):
    def to_one(i, j):
        return int(i*N + j)

    def to_two(k):
        return k//N, k%N

    ret = np.zeros((2*N**2, N**2))
    for k in range(N**2):
        i, j = to_two(k)

        x_grad_ind = k*2
        y_grad_ind = x_grad_ind+1

        if j < N-1:
            ret[x_grad_ind][to_one(i,j)] = -1
            ret[x_grad_ind][to_one(i,j+1)] = 1
        else:
            ret[x_grad_ind][to_one(i,j)] = 1
            ret[x_grad_ind][to_one(i,j-1)] = -1

        if i < N-1:
            ret[y_grad_ind][to_one(i,j)] = -1
            ret[y_grad_ind][to_one(i+1,j)] = 1
        else:
            ret[y_grad_ind][to_one(i,j)] = 1
            ret[y_grad_ind][to_one(i-1,j)] = -1
    return ret

def laplacian_matrix(N):
    def to_one(i, j):
        return int(i*N + j)

    def to_two(k):
        return k//N, k%N

    ret = np.zeros((N**2, N**2))

    for k in range(N**2):
        i, j = to_two(k)

        if j < N-1 and j > 0:
            ret[k][to_one(i,j)] += -2
            ret[k][to_one(i,j+1)] += 1
            ret[k][to_one(i,j-1)] += 1
        elif j==N-1:
            ret[k][to_one(i,j)] += 1
            ret[k][to_one(i,j-2)] += 1
        elif j==0:
            ret[k][to_one(i,j)] += 1
            ret[k][to_one(i,j+2)] += 1

        if i < N-1 and i > 0:
            ret[k][to_one(i,j)] += -2
            ret[k][to_one(i+1,j)] += 1
            ret[k][to_one(i-1,j)] += 1
        elif i==N-1:
            ret[k][to_one(i,j)] += 1
            ret[k][to_one(i-2,j)] += 1
        elif i==0:
            ret[k][to_one(i,j)] += 1
            ret[k][to_one(i+2,j)] += 1
    return ret

def boundary_matrices(N):
    def to_one(i, j):
        return int(i*N + j)

    B1 = np.zeros((4*N - 4, 2*N**2))

    B2 = np.zeros(N**2)

    k = 0
    for i in range(N):
        if 0 < i and i < N-1:
            B1[k][to_one(i, 0)*2] = -1
            B1[k+1][to_one(i, N-1)*2] = 1
            B2[to_one(i, 0)] = 1
            B2[to_one(i, N-1)] = 1
        elif i == 0:
            for j in range(N):
                B1[k][to_one(i, j)*2+1] = 1
                k += 1
                B2[to_one(i, j)] = 1
        elif i == N-1:
            for j in range(N):
                B1[k][to_one(i, j)*2+1] = -1
                k += 1
                B2[to_one(i, j)] = 1
    return B1, B2
    



        
