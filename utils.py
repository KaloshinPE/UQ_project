import numpy as np
from fenics import *
from dolfin import *
from scipy import sparse as sps
from scipy.sparse import linalg as sla
import time
from matplotlib import pyplot as plt


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

def get_rbf_expression(r_i):
    return Expression(
        "exp(-0.5 * (pow((x[0] - {}), 2) + pow((x[1] - {}), 2)) / pow(0.15, 2))".format(r_i[0], r_i[1]), degree=2
    )


def dolfin2py(obj):
    if isinstance(obj, cpp.la.Vector):
        return obj.gather_on_zero()
    if isinstance(obj, cpp.la.Matrix):
        m = obj.size(0)
        n = obj.size(1)
        ia = [0]
        ja = []
        v = []
        for i in range(m):
            rowj, rowv = obj.getrow(i)
            ja += list(rowj)
            v += list(rowv)
            ia.append(len(ja))
        return sps.csr_matrix((v, ja, ia), shape=(m, n))
    
def plot_vec(x, V, **kwargs):
    z = Function(V)
    z.vector()[:] = x
    return plot(z, **kwargs)


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
#         self.ri = np.random.rand(n_p, n_dim) #centers of radial functions
        self.ri = np.random.uniform(0, 1, (n_p, n_dim)) #centers of radial functions
        self.x = np.random.lognormal(mean=.0, sigma=2., size=self.n_p) #true parameters
#         self.x = np.array([1., 15., 16., 30., 16., 16., 16., 1., 1.]) #true parameters
        
        self.bi = np.zeros([n_p] + [n_discr]*n_dim) # radial functions 
        
        self.rbfs = [get_rbf_expression(self.ri[i]) for i in range(self.n_p)]
        
        for i in range(n_p):
            for ind in index_iter(n_dim, n_discr):
                r = np.array([self.discr_row[i] for i in ind])
                val = radial_fun(r, self.ri[i])
                arr_ind = tuple(((k) for k in ind))
                self.bi[i][arr_ind] = val 
    
        self.K = self.eval_perm_field(self.x) # true permability field
        
        # generate sources
        self.q_sigma = 0.05
        self.q_coeffs = np.array([-2, 3, 2, -3])
        self.Q = np.zeros([n_discr]*n_dim)
        self.q = self.q_coeffs[0]*Expression(
            f"exp(-0.5 * (pow((x[0] - 0.3), 2) + pow((x[1] - 0.3), 2)) / pow({self.q_sigma}, 2))", degree=2
        ) + self.q_coeffs[1]*Expression(
            f"exp(-0.5 * (pow((x[0] - 0.7), 2) + pow((x[1] - 0.3), 2)) / pow({self.q_sigma}, 2))", degree=2
        ) + self.q_coeffs[2]*Expression(
            f"exp(-0.5 * (pow((x[0] - 0.3), 2) + pow((x[1] - 0.3), 2)) / pow({self.q_sigma}, 2))", degree=2
        ) + self.q_coeffs[3]*Expression(
            f"exp(-0.5 * (pow((x[0] - 0.7), 2) + pow((x[1] - 0.7), 2)) / pow({self.q_sigma}, 2))", degree=2
        )

        for ind in index_iter(n_dim, n_discr):
            r = np.array([self.discr_row[i] for i in ind])
            val = 0
            for center, coeff in zip(index_iter(n_dim, 2), self.q_coeffs):
                center = center*0.4 + 0.3 # place sources in the corners 
                val += coeff*radial_fun(r, center, self.q_sigma)
            arr_ind = tuple(((k) for k in ind))
            self.Q[arr_ind] = val 

        self.G = gradient_matrix(self.n_discr)
        self.L = laplacian_matrix(self.n_discr)
#         self.u = self.evaluate_(self.x)[0]
        self.u, _ = self.evaluate(self.x)# + np.random.normal(0, .0, size=self.K.shape)
    
    def eval_perm_field(self, x):
        return np.tensordot(self.bi, x, ((0),(0)))
    
    def eval_u_x(self, x, u):
        K = self.eval_perm_field(x)
        phi = [K*grad for grad in np.gradient(u)]
        return sum([np.gradient(elem, axis=i) for i, elem in enumerate(phi)]) + self.Q
    
    def basis_projection_solution(self, x, U, u=None, get_time=True):
        """
        U supposed to be the list of previously calculated u values of shape (n_discr x n_discr x ...)
        """
#         K = self.eval_perm_field(x)
#         A = []
#         U = orthonormal(np.vstack([u.flatten() for u in U]))
#         U = [u.reshape(self.n_discr, self.n_discr) for u in U]
#         for u in U:
#             phi = [K*grad for grad in np.gradient(u)]
#             new_vec = np.sum([np.gradient(elem, axis=i) for i, elem in enumerate(phi)], axis=0)
#             A.append(new_vec.flatten())
#         U = np.vstack([u.flatten() for u in U]).T
#         A = U.T @ np.vstack(A).T
#         b = -U.T @ self.Q.flatten()
# #         A = np.vstack(A).T
# #         b = self.Q.flatten()
# #         coeffs, _, _, _ = np.linalg.lstsq(A, b)
#         coeffs = np.linalg.solve(A, b)
#         print(len(coeffs))
#         return (A @ coeffs).reshape(*U[0].shape)

        ret_time = -1
        if get_time:
#             ret_time = 0.1382
            ret_time = time.clock()
            G = self.G
            L = self.L
            k = self.eval_perm_field(x).flatten()
            K = np.diag(k)
            q = self.Q.flatten()
            A = (np.dot(G.T @ G, k) + K @ L)
            U_ = orthonormal(np.vstack([u.flatten() for u in U])).T

            A_m = U_.T @ A @ U_
            b_m = -U_.T @ q
            coeffs = np.linalg.solve(A_m, b_m)
#             u = self.evaluate(x)
            ret_time = time.clock() - ret_time
        
#             return u

        U = orthonormal(np.vstack([elem.flatten() for elem in U])).T
        if u is None:
            u , _ = self.evaluate(x)
        u = u.flatten()
        coeffs, _, _, _ = np.linalg.lstsq(U, u)
        return (U @ coeffs).reshape(self.n_discr, self.n_discr), ret_time
#         return np.sum([u*c for u, c in zip(U, coeffs)], axis=0)
#         U = orthonormal(np.vstack([elem.T.flatten() for elem in U])).T
# 
#         ret_time = time.clock()
#         
#         U = orthonormal(np.vstack([elem.T.flatten() for elem in U])).T
#         # add vector for lambda
#         U = np.pad(U, [[0, 1], [0, 1]], mode='constant', constant_values=0)
#         U[-1,-1] = 1
# 
#         mesh = UnitSquareMesh(self.n_discr-1, self.n_discr-1)
#         V = FunctionSpace(mesh, "CG", 1)
#         
#         k = sum([self.rbfs[i] * x[i] for i in range(self.n_p)])
#         
#         u = TrialFunction(V)
#         v = TestFunction(V)
#         A = k * dot(grad(u), grad(v)) * dx()
#         rhs = self.q * v * dx()
#         bc = u * ds()
#         
#         Amat = dolfin2py(assemble(A))
#         rhsvec = dolfin2py(assemble(rhs))
#         bcvec = dolfin2py(assemble(bc))
#         
#         n = Amat.shape[0]
#         print(type(sps.bmat([[Amat, bcvec.reshape(n, 1)], [bcvec.reshape(1, n), None]])))
#         A = sps.bmat([[Amat, bcvec.reshape(n, 1)], [bcvec.reshape(1, n), None]]).todense() @ U
#         b = np.hstack([rhsvec, [0]])
# #         coeffs = np.linalg.solve(A, b)
#         coeffs, _, _, _ = np.linalg.lstsq(A, b)
# 
#         ret_time = time.clock() - ret_time
#         print(A @ coeffs - b)
#         print(np.sum(coeffs))
# 
#         return (U @ coeffs)[:-1].reshape(self.n_discr, self.n_discr).T, ret_time
    
    def evaluate_(self, x):
        ret_time = time.clock()
#         mesh = UnitSquareMesh.create(self.n_discr, self.n_discr, CellType.Type.quadrilateral)
        mesh = UnitSquareMesh.create(self.n_discr-1, self.n_discr-1, CellType.Type.quadrilateral)
        
        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R = FiniteElement("Real", mesh.ufl_cell(), 0)
        W = FunctionSpace(mesh, P1 * R)
        
        k = sum([self.rbfs[i] * x[i] for i in range(self.n_p)])
        
        (u, c) = TrialFunction(W)
        (v, d) = TestFunctions(W)
        
        div(k * grad(u)) + q
        
        f = -q
        g = Expression("0", degree=2)
        
        a = (inner(k * grad(u), grad(v)) + c*v + u*d)*dx
        L = f*v*dx + g*v*ds
        
        w = Function(W)
        solve(a == L, w,)
        (u, c) = w.split()
        ret_time = time.clock() - ret_time
        return u.compute_vertex_values().reshape(self.n_discr, self.n_discr), ret_time
    
    
    def evaluate(self, x):
        ret_time = time.clock()
        
        mesh = UnitSquareMesh(self.n_discr-1, self.n_discr-1)
        V = FunctionSpace(mesh, "CG", 1)
        
        k = sum([self.rbfs[i] * x[i] for i in range(self.n_p)])
        
        u = TrialFunction(V)
        v = TestFunction(V)
        A = k * dot(grad(u), grad(v)) * dx()
        rhs = self.q * v * dx()
        bc = u * ds()
        
        Amat = dolfin2py(assemble(A))
        rhsvec = dolfin2py(assemble(rhs))
        bcvec = dolfin2py(assemble(bc))
        
        n = Amat.shape[0]
        B = sps.bmat([[Amat, bcvec.reshape(n, 1)], [bcvec.reshape(1, n), None]])
        ulam = sla.spsolve(B.tocsc(), np.hstack([rhsvec, [0]]))
        uh = ulam[:-1]
        lam = ulam[-1]
        
        ufunc = Function(V)
        ufunc.vector()[:] = uh
        
        probeX, probeY = np.meshgrid(np.linspace(0, 1, self.n_discr), np.linspace(0, 1, self.n_discr))
        probeX = probeX.reshape(-1)
        probeY = probeY.reshape(-1)
        u_values = np.array([ufunc(x, y) for (x, y) in zip(probeX, probeY)])

        ret_time = time.clock() - ret_time

        return u_values.reshape(self.n_discr, self.n_discr), ret_time


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
    
    def setval(k,n,val):
        try:
            ret[k,n] += val
        except IndexError:
            pass

    ret = np.zeros((2*N**2, N**2))
    for k in range(N**2):
        i, j = to_two(k)

        x_grad_ind = k*2
        y_grad_ind = x_grad_ind+1
        
        setval(x_grad_ind, to_one(i,j), -1)
        setval(x_grad_ind, to_one(i,j+1), 1)
        
        setval(y_grad_ind, to_one(i,j), -1)
        setval(y_grad_ind, to_one(i+1,j), 1)
        

#         if j < N-1:
#             ret[x_grad_ind][to_one(i,j)] = -1
#             ret[x_grad_ind][to_one(i,j+1)] = 1
#         else:
#             ret[x_grad_ind][to_one(i,j)] = 1
#             ret[x_grad_ind][to_one(i,j-1)] = -1

#         if i < N-1:
#             ret[y_grad_ind][to_one(i,j)] = -1
#             ret[y_grad_ind][to_one(i+1,j)] = 1
#         else:
#             ret[y_grad_ind][to_one(i,j)] = 1
#             ret[y_grad_ind][to_one(i-1,j)] = -1
    return ret

def laplacian_matrix(N):
    def to_one(i, j):
        return int(i*N + j)

    def to_two(k):
        return k//N, k%N

    ret = np.zeros((N**2, N**2))

    def setval(k,n,val):
        try:
            ret[k,n] += val
        except IndexError:
            pass
        
    for k in range(N**2):
        i, j = to_two(k)

#             if j < N-1 and j > 0:
#                 ret[k][to_one(i,j)] += -2
#                 ret[k][to_one(i,j+1)] += 1
#                 ret[k][to_one(i,j-1)] += 1
        setval(k,to_one(i,j),-2)
        setval(k,to_one(i,j+1),1)
        setval(k,to_one(i,j-1),1)
        setval(k,to_one(i,j),-2)
        setval(k,to_one(i+1,j),1)
        setval(k,to_one(i-1,j),1)
#         elif j==N-1:
#             ret[k][to_one(i,j)] += 1
#             ret[k][to_one(i,j-2)] += 1
#         elif j==0:
#             ret[k][to_one(i,j)] += 1
#             ret[k][to_one(i,j+2)] += 1

#         if i < N-1 and i > 0:
#             ret[k][to_one(i,j)] += -2
#             ret[k][to_one(i+1,j)] += 1
#             ret[k][to_one(i-1,j)] += 1
#         elif i==N-1:
#             ret[k][to_one(i,j)] += 1
#             ret[k][to_one(i-2,j)] += 1
#         elif i==0:
#             ret[k][to_one(i,j)] += 1
#             ret[k][to_one(i+2,j)] += 1
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
    



        
