# CSE891 Mini Project 2: Classification with Neural Network
This is the second mini project in this class which accounts for 10\% fo your final grade. Similar to previous one you should do the project individually, without collaboration with any other students. The project requires implementing and evaluating a classification problem on Breast Cancer Wisconsin data set using multi-layer neural network.  You must implement your methods in Python using the provided template program.You are prohibited from using any Python libraries for Deep Learning (e.g., PyTorch  and TensorFlow) to do this project. You're expected to complete the project using numpy and the standard functions in Python. Please check with the instructor/TA first if you want to use other packages besides those provided by numpy. The project due date is December 6, 2019 (before midnight).
## Project Overview
	
The dataset for this project is a subset of the Breast Cancer Wisconsin Data Set. The original dataset contains 699 instances and 10 attributes. You can access to the original dataset from following link: \url{https://archive.ics.uci.edu/ml/datasets.php}. The original dataset includes some missing values. The version provided  in the class webpage on D2L has ignored all the instances with missing values. The attributes Information is provided in the  readme file. 	

## Tasks

The mini-project consists of following programming tasks to be completed:

- Data loading and creation.
- Classification using a two layers network.
- Multilayer Classifier

### Task 1: Data Loading and Creation

	The first task consists of the following steps:

		- Open and load the input data provided on the class webpage on D2L.
		- Extract the  predictor  and outcome variable.
		- Create train and test set.


### Task 2: A two layers classifier
**Input Layer:** is a fully connected layer with following activation functions options:

- **Sigmoid**: $\sigma(z) = \frac{1}{1+ e^{-z}}$
- **RelU**: 
$$
ReLU(z) = \begin{cases}
z & \textrm{if}\  z>0\\
0 & \textrm{o.w} \\ \end{cases}
$$


**Output Layer**  is a fully connected layer with sigmoid activation and a cross entropy loss function:
	$$Loss(y,\hat{y}) = -y log(\hat{y})-(1-y)log(1- \hat{y})$$
	

 You will implement a forward and a backward function for the classifier


- The forward function receive inputs and weights. It will return both the outputs of both layers.
- The backward function gets the output of forward layer and update the model parameters using stochastic gradient descent method.  


### Task 3: Multilayer Classifier
The last task requires you to implement fully-connected networks of arbitrary depth to solve the classification problem. For each layer you will implement a `forward` and a `backward` function separately. The `forward` function will receive inputs from previous layer, alongside with models parameters and will return an output for next layer and also store the data needed for the backward pass in a  cache object. The backward function receive  derivatives of the next layer and the cache object from the forward layer, and will return gradients with respect to the inputs and weights. Similar to previous task, the last layer is a fully connected layer with sigmoid activation and a cross entropy loss function. For the other layers  number of nodes and type of activation function are hyper parameter. You will report your results for networks with two to five layers.

