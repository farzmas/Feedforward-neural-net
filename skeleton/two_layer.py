import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TwoLayerNet():
    """
    A two-layer fully-connected neural network.
    The network has a cross entropy loss function and. The input layer uses a ReLU or Sigmoid
    activation function. The output uses sigmoid only.   

    The outputs of the second fully-connected layer is a value between 0 and 1.
    """
    def __init__(self, input_dim, hidden_dim, activation_type = 'sigmoid' ,epochs = 50, learning_rate = 0.001):
        self.activation_type = activation_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_init(input_dim, hidden_dim)
    
    def random_init(self, input_dim, hidden_dim):
        """ Intialize the model parameters randomly """
        self.W1 = np.random.normal(loc=0.0,scale = 0.1, size= (hidden_dim, input_dim))
        self.b1 = np.zeros((hidden_dim,1))
        
        self.W2 = np.random.normal(loc=0.0,scale = 0.1, size= (1 , hidden_dim))
        self.b2 = np.zeros((1,1))
            
    
    def forward_propagation(self, x):
        """
        Compute forward propagation step. This is called by fit function. 
        input: 
            x: a numpy vector 9*1. Single data point.
        output:
            z1: input layer pre-activation function output.
            a1: input layer activation function output. 
            z2: output layer pre-activation function output.
            a2: output layer activation function. 
        """
        
        #############################################################################
        # TODO: Compute the input layer pre-activation linear function.             #
        #############################################################################
        # START OF YOUR CODE 
        z1 = None 
        pass

        
        #############################################################################
        # TODO: Compute the input layer activation function.                        #
        # It can be sigmoid or ReLU function.                                       #
        #############################################################################
        # START OF YOUR CODE 
        a1 = None
        pass

        
        #############################################################################
        # TODO: Compute the output layer pre-activation linear function.            #
        #############################################################################
        # START OF YOUR CODE 
        z2 = None
        pass

        
        #############################################################################
        # TODO: Compute the output layer activation linear function.                #
        #############################################################################
        # START OF YOUR CODE 

        a2 = None
        pass
    
        return z1, a1, z2, a2


    
    def back_propagation(self, x, y , z1, a1, z2, a2):
        """
        Compute backward propagation step. This is called by fit function. 
        input: 
            - x: a numpy vector 9*1. Single data point.
            - y: corresponding response value. 
            - z1: input layer pre-activation function output.
            - a1: input layer activation function output. 
            - z2: output layer pre-activation function output.
            - a2: output layer activation function. 
        output: 
            - gradients of model parameters 
        """
        #############################################################################
        # TODO: Compute W2 and b2 gradiants.                                        #
        #############################################################################
        # START OF YOUR CODE 
        
        grad_W2 = None
        grad_b2 = None
        pass 



        #############################################################################
        # TODO: Compute W1 and b1 gradiants.                                         #
        #############################################################################
        # START OF YOUR CODE         
        
        grad_W1 = None
        grad_b1 = None 
        pass 

        return grad_W1, grad_b1, grad_W2, grad_b2

   
    def update(self, grad_W1, grad_b1, grad_W2, grad_b2):
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
                z1, a1, z2, a2 = self.forward_propagation(x)
                if plot_loss: 
                    loss.append(self.loss(a2,y))
                grad_W1, grad_b1, grad_W2, grad_b2 = self.back_propagation(x, y , z1, a1, z2, a2)
                self.update(grad_W1, grad_b1, grad_W2, grad_b2)
            if plot_loss:
                total_loss.append(np.round(np.mean(loss),4))
        if plot_loss:
            plt.plot(range(self.epochs),total_loss)
            plt.xlabel('epoch')
            plt.ylabel('Accuracy')
            plt.title('loss')

        

    def predict(self, X):
        """
        Use the two-layer trained  network to predict response values for
        data points X. For each data point we predict score value between 0 and 1, 
        and assign each data point to the class 0 for scores between 0 and 0.5 and 1 otherwise.

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