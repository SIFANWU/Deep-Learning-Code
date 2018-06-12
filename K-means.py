import sys
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sb  
from scipy.io import loadmat

def init_centroids(X, k):

    m,n = X.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0,m,k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

def find_closest_centroids(X, centroids):

    m = X.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dis = 10000

        for j in range(centroids.shape[0]):

            X_temp = np.power(X[i,] - centroids[j,],2)
            distance = np.sum(X_temp)

            if distance < min_dis:
                min_dis = distance
                idx[i] = j
  
    return idx

def compute_centroids(X, idx, k):

    m,n = X.shape
    centroids = np.zeros((k,n))

    for i in range(k):

        indices = np.where(idx==i)[0]
        x = X[indices,:]
        x_1 = np.sum(x[:,0]) / x.shape[0]
        x_2 = np.sum(x[:,1]) / x.shape[0]
        x_3 = np.sum(x[:,2]) / x.shape[0]
        centroids[i,:] = x_1,x_2,x_3

    return centroids

def run_k_means(X, initial_centroids, max_iters):

    m,n = X.shape
    idx = np.zeros(m)
    k = initial_centroids.shape[0]
    centroids = initial_centroids

    for i in range(max_iters):

        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx,centroids

if __name__ == '__main__':

    # data = loadmat('data/ex7data2.mat')

    # X = data['X']       # 300*2

    # initial_centroids = init_centroids(X, 3)

    # idx, centroids = run_k_means(X, initial_centroids, 10)

    # cluster_1 = np.where(idx==0)[0]
    # cluster_2 = np.where(idx==1)[0]
    # cluster_3 = np.where(idx==2)[0]

    # c1 = X[cluster_1,:]
    # c2 = X[cluster_2,:]
    # c3 = X[cluster_3,:]

    # fig, ax = plt.subplots(figsize=(12,8))
    # ax.scatter(c1[:,0], c1[:,1], s=30, color='r', label='Cluster 1')  
    # ax.scatter(c2[:,0], c2[:,1], s=30, color='g', label='Cluster 2')  
    # ax.scatter(c3[:,0], c3[:,1], s=30, color='b', label='Cluster 3')  
    # ax.legend()
    # plt.show()

# image compression

    image_data = loadmat('data/bird_small.mat')

    A = image_data['A']     # 128*128*3
    
    A = A/255       # normalize value ranges

    X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))    # reshape the array

    initial_centroids = init_centroids(X, 16)

    idx, centroids = run_k_means(X, initial_centroids, 10)

    idx = find_closest_centroids(X, centroids)
    
    X_recovered = centroids[idx.astype(int),:]
    
    X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

    plt.imshow(X_recovered) 
    plt.show()
























