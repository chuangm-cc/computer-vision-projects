import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    # W = np.zeros((in_size,out_size))
    b = np.zeros((out_size))
    W_size = (in_size,out_size)
    # according to formula in paper:
    high = np.sqrt(6)/np.sqrt(in_size+out_size)
    low = - high
    W = np.random.uniform(low,high,(W_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1/(1+np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    # print(X.shape,W.shape,b.shape)
    num = X.shape[0]
    out_size = b.shape[0]
    pre_act = np.zeros((num, out_size))
    for i in range(num):
        value = X[i] @ W + b
        pre_act[i, :] = value

    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    # print(x.shape)
    N, n = x.shape
    res = np.zeros((N, n))
    for i in range(N):
        # for each row
        # use the numerical stability trick in Q1.1 softmax
        c = - max(x[i, :])
        sum = 0
        for j in range(n):
            res[i, j] = np.exp(x[i, j] + c)
            sum += res[i, j]
        res[i, :] = res[i, :] / sum
    # print(res)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    # print(y)
    loss_matrix = y * np.log(probs)
    loss = -np.sum(loss_matrix)
    N = y.shape[0]
    correct_num = 0
    for i in range(N):
        prob_index = np.argmax(probs[i, :])
        y_index = np.argmax(y[i, :])
        if y_index == prob_index:
            correct_num += 1
    acc = correct_num / N

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################
    # do the derivative through activation
    action_deriv = activation_deriv(post_act)
    # derivative of cross-entropy:delta
    # print(action_deriv.shape)
    # print(delta.shape)
    # print(W.shape)
    # print(b.shape)
    # print(X.shape)
    deriv_f = action_deriv * delta
    # d(f(wx+b))/dw = x*f'
    grad_W = X.T @ deriv_f
    # d(f(wx+b))/db = I*f'
    N = deriv_f.shape[0]
    I = np.ones((1, N))
    grad_b = (I @ deriv_f).reshape(b.shape)
    # d(f(wx+b))/dx = w*f'
    grad_X = (W @ deriv_f.T).T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    # print(x.shape,y.shape)
    N = x.shape[0]
    num_per_batch = N // batch_size
    rand_index = np.random.permutation(N)
    start_index = 0
    # print(x)
    # print(rand_index)
    # the first n-1 batches
    for i in range(batch_size - 1):
        batch_index = rand_index[start_index:start_index + num_per_batch]
        start_index += num_per_batch
        batch_x = x[batch_index]
        batch_y = y[batch_index]
        batches.append((batch_x, batch_y))
    # the last batch(the rest)
    batch_index = rand_index[start_index:]
    batch_x = x[batch_index]
    batch_y = y[batch_index]
    # print(batch_x)
    batches.append((batch_x, batch_y))
    return batches
