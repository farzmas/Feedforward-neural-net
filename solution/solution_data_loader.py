import numpy as np
from sklearn.model_selection import train_test_split
def load_data(file_path, test_size=0.5, random_seed=0):
    """
    Function to load the data and create train an test sets.
    Input:
        - input_file: path to input file.
        - test_size: a floating point number between 0.0 and 1.0, which represents the proportion of 
        the input data to be used as the test set.
        - random_seed:  an integer seed to be used by the random number generator. 
    Output:
        - X_train: a n*9 predictor matrix for training the classifier. n is training size. 
        - Y_train: a n*1 response matrix for training.
        - X_test: a m*9 predictor matrix for testing the classifier. m is test size size. 
        - Y_test: a m*1 response matrix for testing.
    """ 

    ################################################################################################
    # TODO: (1) load the input data given the file_path.                                           #
    # (2) Extract the  predictor  and outcome variable. The outcome should be converted to 0 and 1 #
    # (3) Create train and test set.                                                               #
    ################################################################################################
    # START OF YOUR CODE 

    
    f = open(file_path,'r')
    lines = f.readlines()
    X = list()
    Y = list()
    for ct, line_a in enumerate(lines):
        try:
            line = line_a.strip().split(',')
            X.append([ int(d) for d in line[1:-1]])
            if( int(line[-1])==2):
                Y.append(0)
            else:
                Y.append(1)
        except:
            pass
    X = np.array(X)
    Y = np.array(Y)
    np.random.seed(random_seed)
    np.random.shuffle(X)
    np.random.seed(random_seed)
    np.random.shuffle(Y)
    idx = round(X.shape[0]*0.5)
    return X[idx:], X[:idx], Y[idx:],Y[:idx]
