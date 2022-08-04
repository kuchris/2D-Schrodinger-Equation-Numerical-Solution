import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from scipy import sparse
from numba import jit
import os
my_path = os.path.abspath(r"C:\Users\KuChris\Desktop\result") # Figures out the absolute path for you in case your working directory moves around.


##Create meshgrid for x and y
N = 100
L = 2
X,Y = np.meshgrid(np.linspace(-L/2,L/2,N, dtype=float),np.linspace(-L/2,L/2,N, dtype=float))
##potential m\deltax^2 unit
p = '2DHA'

#V = 0*X

# ISW
# V = np.zeros([N, N])

# SHO
#V = ((X)**2 + (Y)**2)/2

# SHO with Gaussian Barrier
# V = (400.0*((X/L)**2 + (Y/L)**2)/2.0 + 100*np.exp(-0.5*((X/L)**2 + (Y/L)**2)/0.1**2))

# Cone
# V = 0.5*(np.sqrt((X/L)**2 + (Y/L)**2))

# Circular Well
#from scipy.signal import square
#V = 200.0*(1.0 - square(2.1*np.pi*np.sqrt((X/L)**2 + (Y/L)**2)))

# Circular Well with Partitions
# from scipy.signal import square
# V = 200.0*(1.0 - square(2.0*np.pi*np.sqrt((X/L)**2 + (Y/L)**2)))
# V2 = np.zeros([N, N])
# V2[:, 15*N//32: 17*N//32] = 15.0
# V2[15*N//32: 17*N//32, :] = 15.0
# V += V2

#Four Square Wells
#V = np.zeros([N, N])
#height = 150.0
#V[:, 64*N//128: 65*N//128] = height
#V[64*N//128: 65*N//128, :] = height
#V[0: 4*N//32, :] = height
#V[:, 0: 4*N//32] = height
#V[N - 4*N//32: N, :] = height
#V[:, N - 4*N//32: N] = height

# Inverted Gaussian Well
#V = (1.0 - np.exp(-0.5*((X/L)**2 + (Y/L)**2)/0.25**2))

# Inverted Gaussian Wells
#s = 0.12
#V = 50.0*(1.0 - np.exp(-0.5*((X/L)**2 + (Y/L - 0.25)**2)/s**2)
#            - np.exp(-0.5*((X/L)**2 + (Y/L + 0.25)**2)/s**2)
#            - np.exp(-0.5*((X/L - 0.25)**2 + (Y/L)**2)/s**2)
#            - np.exp(-0.5*((X/L + 0.25)**2 + (Y/L)**2)/s**2))

# tanh potential
#V = 50.0*np.tanh((X/L)**2 + (Y/L)**2)

# 2d hydengen atom with convergence factor(yukawa potential)
esp = 150
V = -(np.exp(-esp*np.sqrt((X)**2 + (Y)**2)))/((X)**2 + (Y)**2)

##create matrix
diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0,1]), N, N)
##define energy
T = -1/2 * sparse.kronsum(D,D)
U = sparse.diags(V.reshape(N**2),(0))
H = T+U
##Solve for eigenvector and eigenvalue
eigenvalues , eigenvectors = eigsh(H, k=n+1, which='SM')

def get_e(n):
    return eigenvectors.T[n].reshape((N,N))
##number of state
for n in range (0,10):
    ##plot V
    plot0 = plt.figure(0,figsize=(8,6))
    cs = plt.contourf(X,Y,V,100)
    plt.colorbar()
    for c in cs.collections:
        c.set_rasterized(True)

    plt.title("Plot of V")
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.savefig(os.path.join(my_path, 'Figure_{}.{}.0.pdf'.format(p,n)))
    ##plot eigenvector
    plot1 = plt.figure(1,figsize=(8,6))
    cs = plt.contourf(X,Y, get_e(n),300)
    plt.colorbar()
    for c in cs.collections:
        c.set_rasterized(True)

    plt.title("Plot of Eigenfunction for {} state".format(n))
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.savefig(os.path.join(my_path, 'Figure_{}.{}.1.pdf'.format(p,n)))
    ##plot probability density
    plot2 = plt.figure(2, figsize=(8,6))
    cs = plt.contourf(X,Y, get_e(n)**2,300)
    plt.colorbar()
    for c in cs.collections:
        c.set_rasterized(True)

    plt.title("Plot of Probability Density for {} state".format(n))
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.savefig(os.path.join(my_path, 'Figure_{}.{}.2.pdf'.format(p,n)))
    ##plot eigenvalues
    plot3 = plt.figure(3)
    alpha = eigenvalues[0]/2
    E_a = eigenvalues/alpha
    b = np.arange(0, len(eigenvalues),1)
    plt.scatter(b, E_a, s=1444, marker="_", linewidth=2, zorder=3)
    plt.title("Plot of eigenvalues")
    plt.xlabel('$(n_{x})^2+(n_{y})^2$')
    plt.ylabel(r'$mE/\hbar^2$')

    c = ['$E_{}$'.format(i) for i in range(0,len(eigenvalues))]
    for i, txt in enumerate(c):
        plt.annotate(txt, (np.arange(0, len(eigenvalues),1)[i], E_a[i]), ha="center")
    plt.savefig(os.path.join(my_path, 'Figure_{}.{}.3.pdf'.format(p,n)))
    plt.show()
    plt.close('all')