import numpy as np
import random
import math
import argparse
import matplotlib.pyplot as plt
import pylab
import sys


#Preparation of the training data

nb_points = 50

blue_points1 = np.random.randn(nb_points/2,2) + np.tile(np.array([[1.8, 0]]),(nb_points/2,1))
blue_points2 = np.random.randn(nb_points/2,2) + 1
blue_points = np.concatenate((blue_points1, blue_points2), axis=0)
blue_points -= np.mean(blue_points, axis = 0)

red_points1 = np.random.random((nb_points/2,2)) + np.tile(np.array([[1, 0]]),(nb_points/2,1))
red_points2 = np.random.random((nb_points/2,2)) + np.tile(np.array([[0.8, 0.8]]),(nb_points/2,1))
red_points = np.concatenate((red_points1, red_points2), axis=0)
red_points -= np.mean(red_points, axis = 0)


X_train = np.concatenate((red_points, blue_points), axis=0)
Y_train = np.concatenate([np.ones(nb_points), np.zeros(nb_points)])


#Construction of the network

#Weights initialization
W = []  #Matrix of weights
b = []  #Bias vector


neurons_per_layer = input("Enter the structure of the network (ex: [2, 5, 3, 1]). It has to be a list finishing with 1 \n")
#In the last layer we have 1 neuron because it is a problem of binary classification -> the last neuron will output 0 or 1

if not isinstance(neurons_per_layer, list):
    print "Wrong input"
    sys.exit()
    
if neurons_per_layer[-1] != 1:
    print "Wrong last input"
    sys.exit()
numb_layers = len(neurons_per_layer)-1


#We initialize the weights and the bias with a standard normal distribution
for k in range(numb_layers):
    W.append(np.random.random((neurons_per_layer[k+1], neurons_per_layer[k])))
    b.append(0.0001*np.random.random(neurons_per_layer[k+1]))

#Definition of the activation function (here we use a simple sigmoid function)
activation_func = lambda x: 1.0/(1.0 + np.exp(-x)) 
derive_activation_func = lambda x: (1 - x)*x

#Cost function
def cost_function(pred, Y_train):
    return sum(0.5*np.power(np.array(pred-Y_train), 2))




# Training of the network
nb_iterations = 1000
learning_rate = 0.1
momentum = 0.9

#Forward pass
for k in range(nb_iterations):

    # We shuffle the training data set
    rand_idx = np.random.choice(2*nb_points, 2*nb_points, replace=False)
    X_train = X_train[rand_idx,:]
    Y_train = Y_train[rand_idx]
    final_output = []
    delta_W_tot = [0]*len(Y_train)

    #For loop on each training pair
    for n in range(len(Y_train)):  
        x_train = X_train[n,:]
        x_train_ini = X_train[n,:]
        y_train = Y_train[n]
        
        local_gradients = [1]*numb_layers
        outputs = []
        for layer in range(numb_layers):
            inter_output = np.dot(W[layer], x_train) + np.array(b[layer])
            output = activation_func(inter_output)   
            outputs.append(output)
                      
            # Output update
            x_train = output

        final_output.append(outputs[-1][0])

        #Back propagation and update of the weights
        delta_W = [0]*numb_layers
        for layer in reversed(range(numb_layers)):            
            if layer == numb_layers-1:
                erreur_constate = y_train - outputs[layer]
                local_gradient = erreur_constate*derive_activation_func(outputs[layer])
            else:
                local_gradient = derive_activation_func(outputs[layer])*np.dot(local_gradients[layer+1], W[layer+1])

            local_gradients[layer] = local_gradient

            
            if layer != 0:
                nb_neurones_layer = W[layer].shape[0]

                local_grad = local_gradients[layer]
                local_grad = local_grad.reshape((nb_neurones_layer, 1))
                x = outputs[layer-1].reshape((len(outputs[layer-1]),1))

                if n != 0:
                    W[layer] = W[layer] + learning_rate*local_grad*x.T + momentum*delta_W_tot[n-1][layer]
                else:
                    W[layer] = W[layer] + learning_rate*local_grad*x.T 

                delta_W[layer] = learning_rate*local_grad*x.T 

            else:
                nb_neurones_layer = W[layer].shape[0]
                local_grad = local_gradients[layer]
                local_grad = local_grad.reshape((nb_neurones_layer, 1))
                x = x_train_ini.reshape((len(x_train_ini),1))

                if n != 0:
                    W[layer] = W[layer] + learning_rate*local_grad*x.T + momentum*delta_W_tot[n-1][layer]
                else:
                    W[layer] = W[layer] + learning_rate*local_grad*x.T 

                delta_W[layer] = learning_rate*local_grad*x.T             

        delta_W_tot[n] = delta_W

    cost = cost_function(final_output, Y_train)

    for l in range(len(final_output)):
        if final_output[l] > 0.5:
            final_output[l] = 1
        else:
            final_output[l] = 0

    accuracy = sum(final_output==Y_train)/float(len(Y_train))*100.

    print "Iteration number ", k
    print "Value of the cost function: %f" % cost
    print "Accuracy: ", accuracy, " % \n"





# We have then the function learnt by the network

def network_function(x, y):
    # Forward pass
    X = np.array([x, y])
    for layer in range(len(W)):
        res = activation_func(np.dot(W[layer], X) + np.array(b[layer]))
        X = res
    return res


X_train = np.concatenate((red_points, blue_points), axis=0)

Y_train = np.concatenate([np.ones(nb_points), np.zeros(nb_points)])

blue_points_x = [k[0] for k in blue_points]
blue_points_y = [k[1] for k in blue_points]

red_points_x = [k[0] for k in red_points]
red_points_y = [k[1] for k in red_points]

plt.plot(red_points_x, red_points_y, 'ro', blue_points_x, blue_points_y, 'bo')
titre = "Network: " + str(neurons_per_layer) + " with learning rate: " + str(learning_rate) + " for " + str(nb_iterations) + " iterations " + "momentum = " + str(momentum)
plt.title(titre)

# define the grid over which the function should be plotted (xx and yy are matrices)
xx, yy = pylab.meshgrid(pylab.linspace(-3, 3, 101), pylab.linspace(-3, 4, 101))

# fill a matrix with the function values
zz = pylab.zeros(xx.shape)
print zz
print zz.shape
for i in range(xx.shape[0]):
    for j in range(xx.shape[0]):
        zz[i,j] = network_function(xx[i,j], yy[i,j])

# plot the calculated function values
pylab.pcolor(xx,yy,zz)

# and a color bar to show the correspondence between function value and color
pylab.colorbar()

pylab.show() 


