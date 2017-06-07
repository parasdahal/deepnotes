---
layout: post
title: Putting it all together and Classifying MNIST dataset
comments: true
categories: cnn-series
---
{% include series.html %}

## The Neural Network class

Now that we have our layers and optimizers ready, we need to bring them together for us to train. To do this, we implement a ```NeuralNet``` class to encapsulate our group of layers as a single network.

The class stores the list of layer objects and parameters for each layer and implements the ```forward``` and ```backward``` methods to flow the output and gradients through these layers. It also implements ```train_step``` method, which is used by our solver to perform forward and backward pass and get loss and gradients for each iteration.

Here is the implementation of ```NeuralNet``` class.

```python
class NeuralNet:

    def __init__(self,layers,loss_func=SoftmaxLoss):
        self.layers = layers
        self.params = [layer.params for layer in self.layers]
        self.loss_func = loss_func

    def forward(self,X):
        for layer in self.layers:
            X=layer.forward(X)
        return X

    def backward(self,dout):
        grads = []
        for layer in reversed(self.layers):
            dout,grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self,X,y):
        out = self.forward(X)
        loss,dout = self.loss_func(out,y)
        loss += l2_regularization(self.layers)
        grads = self.backward(dout)
        grads = delta_l2_regularization(self.layers,grads)
        return loss,grads
    
    def predict(self,X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)
```

## Classifying MNIST Dataset



![MNIST dataset](/public/images/mnist.png)

MNIST dataset is a collection of $$28\times28$$ grayscale images of handwritten digits with integer labels collected by [Yaan LeCun et al.](http://yann.lecun.com/exdb/mnist/) It consists of 60,000 training images and 10,000 test images. MNIST dataset is used a benchmark test for performance of computer vision and machine learning algorithms.

We will now try to use our CNN to test what level of accuracy we can achieve on MNIST dataset. First, let define a function to load the dataset into training and test numpy arrays:

```python
def load_mnist(path,num_training=50000,num_test=10000,cnn=True,one_hot=False):
    f = gzip.open(path,'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='iso-8859-1')
    f.close()
    X_train, y_train = training_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data
    if cnn:
        shape = (-1,1,28,28)
        X_train = X_train.reshape(shape)
        X_validation = X_validation.reshape(shape)
        X_test = X_test.reshape(shape)
    if one_hot:
        y_train = one_hot_encode(y_train,10)
        y_validation = one_hot_encode(y_validation,10)
        y_test = one_hot_encode(y_test,10)
    X_train,y_train = X_train[range(num_training)],y_train[range(num_training)]
    X_test,y_test = X_test[range(num_test)],y_test[range(num_test)]
    return (X_train, y_train),(X_test, y_test)
```

**Network Architecture**

We will be using following network architecture:

1. First Convolution layer with $$16$$ filters of size $$5\times5$$, stride $$1$$ and padding $$2$$ followed by ReLU layer
2. First Maxpool layer with size and stride set to $$2$$
3. Second Convolution layer with $$32$$ filters of size $$5\times5$$, stride $$1$$ and padding $$2$$ followed by ReLU layer
4. Second Maxpool layer with size and stride set to $$2$$
5. Fully Connected Layer with 120 output neurons followed by ReLU layer
6. Final Fully Connected Layer with 10 output neurons

```python
def make_mnist_cnn(X_dim,num_class):
    conv = Conv(X_dim,n_filter=16,h_filter=5,w_filter=5,stride=1,padding=2)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim,size=2,stride=2)
    conv2 = Conv(maxpool.out_dim,n_filter=32,h_filter=5,w_filter=5,stride=1,padding=2)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim,size=2,stride=2)
    flat = Flatten()
    fc1 = FullyConnected(np.prod(maxpool2.out_dim),120)
    relu3 = ReLU()
    fc2 = FullyConnected(120,num_class)
    return [conv,relu,maxpool,conv2,relu2,maxpool2,flat,fc1,relu3,fc2]
```

**Training**

For training our network we use *SGD with Nesterov's Momentum* with minibatch size $$50$$ and learning rate $$0.1$$. We train the network for $$20$$ epochs.

```python
training_set , test_set = load_mnist('data/mnist.pkl.gz',num_training=50000,num_test=10000)
X,y = training_set
X_test,y_test = test_set
mnist_dims = (1,28,28)
cnn = NeuralNet( make_mnist_cnn(mnist_dims,num_class=10) )
cnn = sgd_momentum(cnn,X,y,minibatch_size=50,epoch=100,learning_rate=0.1,X_test=X_test,y_test = y_test,nesterov=True)
```

**Results**

Our network was able to achieve $$98.5$$% accuracy on the test dataset in 20 epochs of training. The output of our training process is shown below:

```
Epoch 1
Loss = 0.6158082986959508 | Training Accuracy = 0.92482 | Test Accuracy = 0.9214
Epoch 2
Loss = 0.48516234663789304 | Training Accuracy = 0.9631 | Test Accuracy = 0.9629
Epoch 3
Loss = 0.43413816209655653 | Training Accuracy = 0.974 | Test Accuracy = 0.9727
Epoch 4
Loss = 0.37336248647019726 | Training Accuracy = 0.98186 | Test Accuracy = 0.98
Epoch 5
Loss = 0.33226090509325357 | Training Accuracy = 0.98148 | Test Accuracy = 0.9784
Epoch 6
Loss = 0.287616026292885 | Training Accuracy = 0.97884 | Test Accuracy = 0.9768
Epoch 7
Loss = 0.25498999493125074 | Training Accuracy = 0.9812 | Test Accuracy = 0.978
Epoch 8
Loss = 0.21029055612821101 | Training Accuracy = 0.98768 | Test Accuracy = 0.9833
Epoch 9
Loss = 0.18531799576774766 | Training Accuracy = 0.9868 | Test Accuracy = 0.9831
Epoch 10
Loss = 0.16083971459715343 | Training Accuracy = 0.9869 | Test Accuracy = 0.9824
Epoch 11
Loss = 0.14779162308404176 | Training Accuracy = 0.98832 | Test Accuracy = 0.984
Epoch 12
Loss = 0.13155044609279076 | Training Accuracy = 0.98776 | Test Accuracy = 0.9834
Epoch 13
Loss = 0.11915613861899203 | Training Accuracy = 0.98792 | Test Accuracy = 0.9834
Epoch 14
Loss = 0.10997964782538788 | Training Accuracy = 0.98672 | Test Accuracy = 0.9835
Epoch 15
Loss = 0.09872268428766803 | Training Accuracy = 0.98642 | Test Accuracy = 0.9831
Epoch 16
Loss = 0.09759947557975868 | Training Accuracy = 0.98636 | Test Accuracy = 0.9832
Epoch 17
Loss = 0.09519812639043623 | Training Accuracy = 0.98704 | Test Accuracy = 0.9835
Epoch 18
Loss = 0.08813418409136559 | Training Accuracy = 0.98606 | Test Accuracy = 0.9826
Epoch 19
Loss = 0.08229978339663813 | Training Accuracy = 0.9885 | Test Accuracy = 0.9845
Epoch 20
Loss = 0.08087128565835669 | Training Accuracy = 0.98874 | Test Accuracy = 0.985
```

**Conclusion**

The current state of the art for classifying MNIST dataset is $$99.79$$% as we can see [here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). The test accuracy $$98.5$$% that we have achieved using our network is not exceptional, but it's not very bad either. We can improve the performance of our network by 

1. Introducing BatchNorm layers after every convolution and fully connected layers
2. Introducing Dropout on fully connected layers
3. Applying Data Augmentation techniques to increase dataset size

