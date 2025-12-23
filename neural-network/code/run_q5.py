import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(1024, 32, params, "layer1")
# zero-initialized momentum accumulators
params['Mw' + 'layer1'] = np.zeros((1024,32))
params['Mb' + 'layer1'] = np.zeros((32))

initialize_weights(32, 32, params, "layer2")
params['Mw' + 'layer2'] = np.zeros((32,32))
params['Mb' + 'layer2'] = np.zeros((32))

initialize_weights(32, 32, params, "layer3")
params['Mw' + 'layer3'] = np.zeros((32,32))
params['Mb' + 'layer3'] = np.zeros((32))

initialize_weights(32, 1024, params, "output")
params['Mw' + 'output'] = np.zeros((32,1024))
params['Mb' + 'output'] = np.zeros((1024))
# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        #pass
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        # forward
        h = forward(xb, params, 'layer1',relu)
        h = forward(h, params, 'layer2',relu)
        h = forward(h, params, 'layer3',relu)
        probs = forward(h, params, 'output', sigmoid)
        #  squared error for the output image compared to the input image
        loss = np.sum((xb-probs)**2)
        total_loss+=loss
        # backward
        delta1 = -2*(xb-probs)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer3', relu_deriv)
        delta4 = backwards(delta3, params, 'layer2', relu_deriv)
        delta5 = backwards(delta4, params, 'layer1', relu_deriv)
        # apply gradient
        # for layers
        for name in ('output','layer3','layer2', 'layer1'):
            grad_W = params['grad_W' + name]
            grad_b = params['grad_b' + name]
            params['Mw' + name] = 0.9 * params['Mw' + name] - learning_rate * grad_W
            params['W' + name] = params['W' + name] + params['Mw' + name]
            params['Mb' + name] = 0.9 * params['Mb' + name] - learning_rate * grad_b
            params['b' + name] = params['b' + name] + params['Mb' + name]
            # params['W' + name] -= learning_rate * grad_W
            # params['b' + name] -= learning_rate * grad_b
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
#plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
reconstructed_x = visualize_x
# TODO: name the output reconstructed_x
##########################
##### your code here #####
##########################
#print(visualize_x.shape)
h = forward(visualize_x, params, 'layer1',relu)
h = forward(h, params, 'layer2',relu)
h = forward(h, params, 'layer3',relu)
reconstructed_x = forward(h, params, 'output', sigmoid)


# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
h = forward(valid_x, params, 'layer1',relu)
h = forward(h, params, 'layer2',relu)
h = forward(h, params, 'layer3',relu)
result = forward(h, params, 'output', sigmoid)
rsnr_total=0
for i in range(len(valid_x)):
    rsnr_total+=peak_signal_noise_ratio(result[i],valid_x[i])
print("Average PSNR: "+str(rsnr_total/len(valid_x)))