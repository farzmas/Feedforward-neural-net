import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Layer(object):
    """
    Class for single layer fully connected network. The layer can use 
    sigmoid or ReLU as activation function. 
    """
    def __init__(self, input_dim, output_dim, activation_type ):
        self.activation_type = activation_type # A string with two options('ReLU','sigmoid')
        # the pre-activation parameter
        self.W = np.random.normal(loc=0.0,scale = 0.1, size= (output_dim, input_dim))
        self.b = np.zeros((output_dim,1))
    
    def pre_activation(self,x):

        #############################################################################
        # TODO: Compute the layer pre-activation linear function.                   #
        #############################################################################
        # START OF YOUR CODE 
        z = None
        pass

        return z
    
    def activation(self):

        if self.activation_type == 'sigmoid':
        #############################################################################
        # TODO: Compute the layer sigmoid(activation) function.                     #
        #############################################################################
        # START OF YOUR CODE 
            a = None
            pass

            return a

        elif self.activation_type == 'ReLU':

        #############################################################################
        # TODO: Compute the layer ReLU(activation) function.                        #
        #############################################################################
        # START OF YOUR CODE 
            a = None
            pass

            return a

        else:

            raise NameError('activation type is not defined!')
    
    
    def forward_propagation(self, x):
        """
        Compute forward propagation step. 
        input: 
            x: a numpy vector input_dim*1. Input of the layer.
        output:
            a:  activation function output. 
        """
        self.z = self.pre_activation(x) # store the preactivation output for back propagation
        self.a = self.activation() # store the activation output for back propagation
        return self.a
    
    def back_propagation(self,x ,delta_l, next_layer = None):
        """
        Compute backward propagation step. Calculate gradients of layer parameters.
        input: 
            - x: Input of the layer. A numpy vector input_dim*1.
            - delta_l: loss propagated from next layer.
            - next_layer: next layer object, you will need it to update delta_l.
            - self.a and self.z: values stored during forward propagation. 
        output: 
            - delta_l for previous layer
        """
        #############################################################################
        # TODO: Compute back propagation.                                           #
        #############################################################################
        # START OF YOUR CODE 
        self.grad_W = None
        self.grad_b = None
        delta_l = None 
        pass 

        return delta_l

   
class MultiLayerNet(object):
    """
   A fully-connected neural network with an arbitrary number of hidden layers, each with
    ReLU or sigmoid activation function.
    """
    def __init__(self, layers, learning_rate = 0.001, epochs = 50):
        self.layers = layers # A list of Layer class objects. 
        self.learning_rate = learning_rate # float: SGD parameters update learning_rate
        self.epochs = epochs # integer: number of epochs
    

    
    def forward_propagation(self,x):
        #############################################################################
        # TODO: Compute forward propagation.                                       #
        #############################################################################
        # START OF YOUR CODE 
        y_pred = None
        pass
        return y_pred
    
    def back_propagation(self,x, y):

        #############################################################################
        # TODO: Compute back propagation.                                           #
        #############################################################################
        # START OF YOUR CODE 

        pass
    
    def update(self):
        """
        Make a single gradient update. This is called by fit function. 
        """
        #############################################################################
        # TODO: update the model parameters using SGD                               #
        #############################################################################
        # START OF YOUR CODE 

        pass
 
    def loss(self, y_hat, y):
        """
        cross-entropy loss function
        Input: 
            - y: response value.
            - y_hat: model predicted value of response. 
        Output:
            - cross entropy loss.
        """
        #############################################################################
        # TODO: Compute cross-entropy loss function                                 #
        #############################################################################
        # START OF YOUR CODE 
        
        loss = None
        pass

        return loss 


    def fit(self,X, Y, plot_loss = False):
        """
        Run optimization to train the model.
        Inputs:
        - X: A numpy array of shape n*9
        - Y: A numpy array of shape n*1
        - print_loss: A True/False value. It will print each epoch average loss if its True. 
        """
        index = list(range(X.shape[0]))
        if plot_loss:
            total_loss = list()
        for epoch in tqdm(range(self.epochs),'epoch:'):
            np.random.shuffle(index)
            if plot_loss:
                loss = list()
            for idx in index:
                x = np.matrix(X[idx]).T
                y = Y[idx]
                y_hat = self.forward_propagation(x)
                if plot_loss: 
                    loss.append(self.loss(y_hat,y))
                self.back_propagation(x, y)
                self.update()   
            if plot_loss:
                total_loss.append(np.round(np.mean(loss),4))   
        if plot_loss:
            plt.plot(range(self.epochs),total_loss)
            plt.xlabel('epoch')
            plt.ylabel('Accuracy')
            plt.title('loss')                   
        
    def predict(self, X):
        """
        Use the trained  network to predict response values for
        data points X. For each data point we predict score value between 0 and 1, 
        and assign each data point to the class 0 for scores between 0 and 0.5 and 1 
        otherwise.

        Inputs:
        - X: A numpy array of shape n*9

        outputs:
        - Y_hat: A numpy array of shape n*1 giving predicted labels for each of
          the elements of X. Y_pred element are either 0 or 1.
        """



        #############################################################################
        # TODO: Implement the predict function.                                     #
        #############################################################################
        # START OF YOUR CODE 
        Y_hat = None
        pass

        return Y_hat