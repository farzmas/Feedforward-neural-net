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
        
        z1 = np.dot(np.matrix(self.W1),np.matrix(x)) + np.matrix(self.b1)

        #############################################################################
        # TODO: Compute the input layer activation function.                        #
        # It can be sigmoid or ReLU function.                                       #
        #############################################################################
        # START OF YOUR CODE 

        if self.activation_type == 'sigmoid':
            a1 = np.divide(np.exp(z1),(np.exp(z1)+1.0))
        elif self.activation_type == 'ReLU':
            a1 = np.maximum(z1,0)

        #############################################################################
        # TODO: Compute the output layer pre-activation linear function.            #
        #############################################################################
        # START OF YOUR CODE 

        z2 = np.dot(np.matrix(self.W2),np.matrix(a1)) + np.matrix(self.b2)

        #############################################################################
        # TODO: Compute the output layer activation linear function.                #
        #############################################################################
        # START OF YOUR CODE 
        
        a2 = np.divide(np.exp(z2),(np.exp(z2)+1.0))
        
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
        
        delta_l = (a2-y)/(a2*(1-a2))
        grad_W2 = (a2-y)*a1.T
        grad_b2 = (a2-y)

        #############################################################################
        # TODO: Compute W1 and b1 gradiants.                                         #
        #############################################################################
        # START OF YOUR CODE         

        if self.activation_type == 'sigmoid':
            delta_l = delta_l*a2*(1-a2)*self.W2
            grad_W1 = np.multiply(delta_l.T,np.multiply(a1,(1-a1))).dot(x.T)
            grad_b1 = np.multiply(delta_l.T,np.multiply(a1,(1-a1)))
        elif self.activation_type == 'ReLU':
            delta_l = delta_l*self.d_ReLU(z2)*self.W2
            grad_W1 = np.multiply(delta_l.T,self.d_ReLU(z1)).dot(x.T)
            grad_b1 = np.multiply(delta_l.T,self.d_ReLU(z1)) 

        return grad_W1, grad_b1, grad_W2, grad_b2
    
    def d_ReLU(self,a):
        """ compute the derivative of ReLU function"""
        a[a>0]=1
        a[a<0]=0
        return(a)   
   
    def update(self, grad_W1, grad_b1, grad_W2, grad_b2):
        """
        Make a single gradient update. This is called by fit function. 
        """
        #############################################################################
        # TODO: update the model parameters using SGD                               #
        #############################################################################
        # START OF YOUR CODE 
        
        self.W1 -= self.learning_rate*grad_W1
        self.b1 -= self.learning_rate*grad_b1
        self.W2 -= self.learning_rate*grad_W2
        self.b2 -= self.learning_rate*grad_b2


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
             
        loss =  -y*np.log(y_hat)-(1-y)*np.log(1-y_hat)
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

        Y_pred = np.zeros(X.shape[0])
        loss = list()
        for idx in range(X.shape[0]):
            x = np.matrix(X[idx]).T
            _ , _ , _ , a = self.forward_propagation(x)
            Y_pred[idx] = np.round(a)
    
    
        return Y_pred