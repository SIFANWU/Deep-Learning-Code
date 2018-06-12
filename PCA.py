import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sb  
from scipy.io import loadmat

def PCA(X):

    X = (X - X.mean()) / X.std()

    X = np.matrix(X)
    cov = np.dot(X.T,X) / X.shape[0]

    U,S,V = np.linalg.svd(cov)

    return U,S,V

def project_data(X, U, k):

    temp = np.dot(X,U[:,:k])

    return temp         # m*k

def recover_data(Z, U, k):

    temp = np.dot(Z,U[:,:k].T)

    return temp

if __name__ == '__main__':

    # data = loadmat('data/ex7data1.mat')  
    # X = data['X']

    # U,S,V = PCA(X)

    # X_pro = project_data(X, U, 1)

    # X_rec = np.array(recover_data(X_pro, U, 1))


    # fig, ax = plt.subplots(figsize=(12,8))  
    # ax.scatter(X_rec[:, 0], X_rec[:, 1])
    # plt.show()

    faces = loadmat('data/ex7faces.mat')  
    X = faces['X']

    face = np.reshape(X[3,:],(32,32))

    U,S,V = PCA(face)

    Z = project_data(face, U, 100)

    face = recover_data(Z, U, 100)

    plt.imshow(face)
    plt.show()








