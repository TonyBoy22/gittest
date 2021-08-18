import numpy as np
from scipy.linalg import cholesky # computes upper triangle by default, matches paper
import matplotlib.pyplot as plt


def sample(S, z_hat, m_FA, Gamma_Threshold=1.0):

    nz = S.shape[0]
    z_hat = z_hat.reshape(nz,1)

    X_Cnz = np.random.normal(size=(nz, m_FA))

    rss_array = np.sqrt(np.sum(np.square(X_Cnz),axis=0))
    kron_prod = np.kron( np.ones((nz,1)), rss_array)

    X_Cnz = X_Cnz / kron_prod       # Points uniformly distributed on hypersphere surface

    R = np.ones((nz,1))*( np.power( np.random.rand(1,m_FA), (1./nz)))

    unif_sph=R*X_Cnz;               # m_FA points within the hypersphere
    T = np.asmatrix(cholesky(S))    # Cholesky factorization of S => S=T’T


    unif_ell = T.H*unif_sph ; # Hypersphere to hyperellipsoid mapping

    # Translation and scaling about the center
    z_fa=(unif_ell * np.sqrt(Gamma_Threshold)+(z_hat * np.ones((1,m_FA))))

    return np.array(z_fa)


def rotation_mat(yaw):
    _rotation_mat = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
    return _rotation_mat

def random_point_generation(a, b, h, k, yaw):
    
    rho = np.random.rand(5000)
    phi = np.random.rand(5000)*2*np.pi
    x = np.sqrt(rho)*np.cos(phi)*a/2
    y = np.sqrt(rho)*np.sin(phi)*b/2
    
    r = rotation_mat(yaw)
    # rotation matrix
    dots = np.dot(r, np.array([x, y]))
    x = dots[0,:] + h
    y = dots[1,:] + k
    
    plt.scatter(x, y)
    plt.show()
    return None
random_point_generation(3,1,-5,2,np.pi/3)