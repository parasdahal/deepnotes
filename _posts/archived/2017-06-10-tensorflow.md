---
layout: post
title: Introduction to TensorFlow and Computation Graph
comments: true
tags: tensorflow-computation-graph
excerpt: Tensorflow is very popular and powerful machine learning library from Google. It was developed by Google Brain Team for in-house research and later open sourced on November 2015. It has been widely adopted in research and production and has become one of the most popular library for Deep Learning.
categories: tensorflow
thumbnail: /public/images/tf_logo.jpg
---

Tensorflow is very popular and powerful machine learning library from Google. It was developed by Google Brain Team for in-house research and later open sourced on November 2015. It has been widely adopted in research and production and has become one of the most popular library for Deep Learning.

Let us discuss briefly about what makes Tensorflow so successful.

**Computation Graph**

Tensorflow approaches series of computations as a flow of data through a graph with nodes being computation units and edges being flow of Tensors (multidimensional arrays). 

Tensorflow builds the computation graph before it starts execution, so the computations are scheduled only when it is absolutely necessary (lazy programming). The graph is not actually executed when the nodes are defined. After the graph is assembled, it is deployed and executed in a *Session*, which is a run-time environment that binds the hardware it is going to run in.

The notion of computation graph also makes machine learning problems very intuitive and easy to visualize and debug. TensorFlow comes with awesome [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to visualize the computation graph.

![Tensorflow Graph](/public/images/tf.gif)



**Low Level Library**

TensorFlow is a low-level computation library, which allows us to use simple operators, such as ‘add’ (element-wise addition of two matrices) and ‘matmul’ (matrix multiplication), in order to implement an algorithm. It provides an extensive suite of functions and classes that allow users to build various models from scratch. 

This is why TensorFlow is more comparable to Numpy than libraries like scikit-learn, which are high level libraries with already implemented machine learning algorithms.

![Numpy & Tensorflow](/public/images/numpy-tf.png)



**Auto Differentiation**

Tensorflow comes with Automatic Differentiation, which as the name suggests, automatically calculates derivatives. As the program is broken down into small, different pieces, TensorFlow efficiently calculates derivatives from the computation graph by using chain rule.

Every node in TensorFlow has an attached gradient operations which calculates derivatives of input with respect to output. Then the gradients with respect to parameters are calculated automatically during backpropagation.

Automatic differentiation is very important because you don’t want to have to hand-code a new variation of backpropagation every time you’re experimenting with a new arrangement of neural networks. This makes TensorFlow a helpful tool to use in research and allows to iterate quickly without having to worry about implementation errors.


**Portability and Flexibility**

TensorFlow models can be trained on both CPU and GPUs, and can run anywhere from mobile to server farms with a single API. This makes TensorFlow a great candidate for both research and production and has become one of the mostly used library in Deep Learning.

-----

## Understanding the Computation Graph

Essentially, TensorFlow computation graph contains the following parts:

1. **Placeholders**, variables used in place of inputs to feed to the graph
2. **Variables**, model variables that are going to be optimized to make model perform better
3. **Model**, a mathematical function that calculates output based on placeholder and model variables
4. **Loss Measure**, guide for optimization of model variables
5. **Optimization Method**, update method for tuning model variables

Let us try to understand the coding paradigm of TensorFlow by building the pieces of our computation graph step by step. 

Note that  

1. While defining the graph, we are not manipulating any data, only building the nodes and symbols inside our graph. 
2. We don't create the graph structure explicitly in our programs. New nodes are automatically built into the underlying graph. We can use ```tf.get_default_graph().get_operations()``` to see all the nodes in the default graph. 


We will implement a linear classifier to classify handwritten digits from MNIST dataset. Let us first import, extract and load the dataset:

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)
```


**Placeholder**

Placeholders are nodes whose values are fed in at the execution time. They are generally used for feeding in inputs and the labels. While declaring Placeholders, we only assign datatype and shape of the tensor. 

Our inputs are images of size $$28\times28$$, but we flatten the images to vectors of size $$784$$. Let us create a placeholder variable to store arbitrary number of these image vectors and name the node as "X".

```python
X = tf.placeholder(tf.float32,[None,784],name="X")
```

Here ```X```  represents a tensor of datatype ```float32``` and will store any number of vector with size 780.


Now let us create another placeholder variable to store one hot encoded vector of labels for our images. 

````python
y_true = tf.placeholder(tf.float32,[None,10],name="y_true")
````

We can store the scalar labels as another placeholder variable ```y_true_cls```.

```python
y_true_cls = tf.argmax(y_true,dimension=1,name="y_true_cls")
```


**Variables**

Variables are stateful nodes that are used for storing model parameters. The state of the Variables are retained across multiple executions of a graph. They can be saved, stored and restored during training which makes experimentation with new models convenient.

To make predictions with our linear classifier, we need to learn the weights and biases. These will be our model variables. 

```weights``` variable is a 2 dimensional tensor of size *input vector size* by *output vector size* ($$784\times10$$). We initialize the tensor to have random numbers from Gaussian distribution.

```python
weights = tf.Variable(tf.random_uniform([784,10],-1,1),name="weights")
```

```Bias``` is a 1 dimensional vector of size *output vector*, which is $$10$$. We initialize it to zeros.

```python
biases = tf.Variable(tf.zeros[10],name="biases")
```

**Model**

Model is a mathematical function that maps the inputs to outputs using the model variables. For our classifier, we use a very simple matrix multiplication model:

```python
logits = tf.matmul(X,weights) + biases
```

In the above line where we define our linear model, we have defined two computation nodes; ```matmul``` and ```add```. 

The output tensor of these computation are stored in Python variable ```logits```. The tensor is of size *number of image vectors* by *number of classes* which is $$784\times10$$.

Now to convert output scores to a probability distribution, we apply softmax function to the output.

```python
y_pred = tf.nn.softmax(logits)
```

Since our output is a probability distribution, we pick out the class with highest probability for our final prediction and store it separately.

```python
y_pred_cls = tf.argmax(y_pred,dimension=1)
```

**Loss Measure**

We will be using Cross Entropy Loss as our loss measure for optimization as our output are a probability distribution. Our goal is to minimize the Cross Entropy Loss as much as possible. We create this loss node using labels and softmax predictions.

```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y_true)
```

Note that our cross entropy function calculates softmax internally, so we are passing ```logits``` and not ```y_pred```.

Now we calculate the mean of cross entropy loss to change it single scalar value.

```
loss = tf.reduce_mean(cross_entropy)
```

**Optimization Method**

We are going to use Gradient Descent optimizer to optimize our model parameters. We create our optimizer object ```tf.train.GradientDescentOptimizer``` and add an optimization operation on the graph by calling minimize method on the optimizer object. 

```
train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
```

```minimize(loss)``` does two things:

1. Computes the gradient of our argument (loss node) with respect to to all the variables (weights and biases)
2. Applies gradient updates to all those variables.

---

## Running the graph

We have constructed the necessary ingredients of our computation graph. Now we are going to execute the graph with a Session. 

Session is a binding to a particular execution environment (CPU or GPU). A Session object creates a runtime where operation nodes are executed and tensors are evaluated. 

To execute our graph, we must initialize a Session object, and call its run method.

**Create Session object**

```python
sess = tf.Session()
```

**Initialize the variables**

```python
sess.run(tf.global_variables_initializer())
```

We need to initialize the variables as we only assign real values to the variables once we are in a runtime i.e. after a session has been created.

**The run method**

The run method of Session object evaluates the output of a node of the graph. It takes in two parameters fetches and feeds as shown below:

```python
sess.run(fetches,feeds)
```

```fetches``` is a list of graph nodes that are to be evaluated. The Session returns the output of these nodes.

```feeds``` are Dictionary mappings from graph nodes to concrete values. Feed Dictionaries specify the value of each Placeholder required by the node to manipulate the data. The name of the keys of Feed Dictionaries should be the same as the Placeholders.

**SGD**

For each iteration of SGD we create batches of input and labels. We then create a feed dictionary from these batches and call the run method on train_step node. We also run the loss node every 50 iterations and print current loss value of the model.

```python
for i in range(num_iter):
    x_batch,y_batch = data.train.next_batch(100)
    feed_dict = {x: x_batch,y_true: y_batch}
    summary = sess.run(train_step,feed_dict)
    if i % 50 == 0:
        loss_val = sess.run(loss,feed_dict)
        print("Loss: ",loss_val)
```

**Accuracy**

We also calculate the classification accuracy of our model on test data after training.

```python
x_test,y_test = data.test.next_batch(10000)    
print("Accuracy of model: ",accuracy.eval(feed_dict = {x:x_test,y_true:y_test}))
```

**Training Results**

```
Loss:  0.696295
Loss:  0.720417
Loss:  0.406744
Loss:  0.694278
Loss:  0.402604
Loss:  0.836929
Loss:  0.425581
Loss:  0.719999
Loss:  0.395444
Loss:  0.57637
Accuracy of model:  0.8569
```

The above linear classifier was trained for 1000 iterations and gave classification error on test data of 85.69%. Not bad!

---

## Visualization using TensorBoard

Thanks to TensorBoard, we can visualize the computation graph and monitor the scalars like loss, accuracy and parameters like weights and biases.

**Creating FileWriter**

To enable this visualization, we first create a FileWriter object. As TensorBoard essentially visualizes the training logs, we need to store them before we can use TensorBoard.

```python
writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
```

Here we are storing our logs to ```./graphs``` directory.

**Logging Dynamic Values**

In order to log loss, accuracy and parameters, we need to create Summary nodes that can be executed inside a session.

We log scalars as below:

```python
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
```

For our parameters, we can create histogram summaries:

```python
tf.summary.histogram("weights", weights)
tf.summary.histogram("biases", biases)
```

These histograms display the occurrence of number of values relative to each other value. They are very helpful in studying the distribution of parameter values over time.

**Merging Summary Operations**

We merge these summary operations so that they can be executed as a single operation inside the session.

```python
summary_op = tf.summary.merge_all()
```

**Running and logging the summary operation**

Now we can run our Summary operation inside the session, and write its output to our FileWriter. Note that ```i``` is the iteration index inside the training loop.

```python
_,summary = sess.run([train_step,summary_op],feed_dict)
writer.add_summary(summary,i)
```
![Scalars](/public/images/scalars.png)

![Histograms](/public/images/histograms.png)

**Making the Graph Readable**

By default our graphs look very messy. We can clean them up by adding a name scope to the nodes.

```python
with tf.name_scope('LinearModel'):
    logits = tf.matmul(x,weights) + biases
    y_pred = tf.nn.softmax(logits)
```

This annotates the graph nodes and makes our graphs readable. We can always click on the plus sign expand and view the full underlying graph.

We can also name our placeholder and variable nodes, as discussed already.

```python
x = tf.placeholder(tf.float32,[None,784],name="x")
y_true = tf.placeholder(tf.float32,[None,10],name="labels")

weights = tf.Variable(tf.random_uniform([784,10],-1,1),name="weights")
biases = tf.Variable(tf.zeros([10]),name="biases")
```

![Graph](/public/images/linear_graph.png)

---

## The Source Code

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Download, extract and load MNIST dataset
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define Placeholders for data and labels
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32,[None,784],name="x")
    y_true = tf.placeholder(tf.float32,[None,10],name="labels")

# Define model variables
with tf.name_scope('Weights'):
    weights = tf.Variable(tf.random_uniform([784,10],-1,1),name="weights")
with tf.name_scope('Biases'):
    biases = tf.Variable(tf.zeros([10]),name="biases")

# Define the model
with tf.name_scope('LinearModel'):
    logits = tf.matmul(x,weights) + biases
    y_pred = tf.nn.softmax(logits)

# Define cost measure
with tf.name_scope('CrossEntropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
    loss = tf.reduce_mean(cross_entropy)

# create optimizer
with tf.name_scope('GDOptimizer'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)

with tf.name_scope('Accuracy'):
    y_pred_cls = tf.argmax(y_pred,dimension=1)
    y_true_cls = tf.argmax(y_true,dimension=1)
    correct_pred = tf.equal(y_pred_cls,y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# adding summary
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("weights", weights)
tf.summary.histogram("biases", biases)
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
    num_iter = 1000
    sess.run(tf.global_variables_initializer())

    for i in range(num_iter):
        x_batch,y_batch = data.train.next_batch(100)
        feed_dict = {x: x_batch,y_true: y_batch}
        _,summary = sess.run([train_step,summary_op],feed_dict)
        writer.add_summary(summary,i)
        if i % 50 == 0:
            loss_val = sess.run(loss,feed_dict)
            print("Loss: ",loss_val)
    
    x_test,y_test = data.test.next_batch(10000)    
    print("Accuracy of model: ",accuracy.eval(feed_dict = {x:x_test,y_true:y_test}))
```
