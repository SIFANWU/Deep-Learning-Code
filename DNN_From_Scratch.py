'''
Building your Deep Neural Network: Step by Step
6/Jun 2018
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

# Step 1 Initialize the parameters

def initialize_parameters_deep(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))   # 加上1是因为这样可以让向量变成(n,1)

    return parameters

# Step 2 Forward propagation module

def linear_forward(A_prev,W,b):

    Z = np.dot(W,A_prev) + b   

    cache = (A_prev,W,b)   # 计算导数的时候, A_prev与W有关, W与dA_prev有关, Z与激活函数有关

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):

    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    cache = (linear_cache,activation_cache)

    return A,cache

def L_model_forward(X, parameters):

    layer = len(parameters) // 2
    caches = []
    A_cache = [X]

    for l in range(1,layer):

        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]

        A,cache = linear_activation_forward(A_cache[-1], W, b, 'relu')
        A_cache.append(A)
        caches.append(cache)

    A_final,cache = linear_activation_forward(A_cache[-1],parameters['W'+str(layer)],parameters['b'+str(layer)],'sigmoid')
    caches.append(cache)

    return A_final,caches

# Step 3 Cost Function

def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL)))

    cost = np.squeeze(cost)

    return cost

# Step 4 Backward propagation

def linear_backward(dZ, cache):

    A_prev,W,b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ,axis=1,keepdims=True) / m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA, cache, activation):

    linear_cache,activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[-1]
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        grads['dA'+str(l+1)],grads['dW'+str(l+1)],grads['db'+str(l+1)] = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu')

    return grads

# Step 5 Update Parameters

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate * grads['db'+str(l+1)]

    return parameters

if __name__ == '__main__':
 
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))






















