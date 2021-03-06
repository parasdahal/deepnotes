---
layout: post
title: Solving the model - SGD, Momentum and Adaptive Learning Rate
excerpt: Thanks to active research, we are much better equipped with various optimization algorithms than just vanilla Gradient Descent. Lets discuss two more different approaches to Gradient Descent - Momentum and Adaptive Learning Rate.
comments: true
categories: cnn-series
thumbnail: /public/images/updates.gif
---
{% include series.html %}

*Note: Complete source code can be found here [https://github.com/parasdahal/deepnet](https://github.com/parasdahal/deepnet)*

Once we have the model of our neural network, we need to find the best set of parameters to minimize the training/test loss and maximize the accuracy of the model. Model solver brings together training data, the model and the optimization algorithms to train the model. A model solver has:

1. Training and Test dataset
2. Reference to the model
3. Different type of optimizers like SGD, Adam
4. Record of loss and accuracy for the training phase for each epoch
5. Optimized parameters for the model

Let us start developing the ideas around building our model solver with a brief review of the backbone of all popular optimizers, Gradient Descent.

---

### Gradient Descent

Gradient Descent is the most common optimization algorithm used in Machine Learning. It uses gradient of loss function to find the global minima by taking one step at a time toward the negative of the gradient (as we wish to minimize the loss function). 

If $$\Delta w$$ and $$\Delta b$$ are the changes we need to find in  direction of $$w$$ and $$b$$ respectively of our Loss Vs. Parameters curve, then the gradient of the loss function L is defined as,


$$
\Delta L = \left(\frac{\partial L}{\partial w},\frac{\partial L}{\partial b}\right)
$$


The way gradient descent works is to now repeatedly compute the gradient $$\Delta L$$ and then move in the opposite direction, "falling down" the slope of the valley.


$$
\begin{align}
w_k \leftarrow w_k - \alpha \frac{\partial L}{\partial w_k} \\
b_k \leftarrow b_k - \alpha \frac{\partial L}{\partial b_k} \\
\end{align}
$$


where $$\alpha$$ is known as Learning Rate, which represents the size of the step taken at each iteration by Gradient Descent.

**Why Gradient Descent?**

Gradient Descent along with Backpropagation algorithm has become the de-facto learning algorithm of neural networks.

In other optimizing algorithm like Newton's Method and BFGS, we need to calculate second order partial derivative matrix or Hessian matrix. Although the algorithm makes efficient updates and doesn't require learning rate parameter, we have to calculate second order partial derivatives matrix for every parameter with respect to every other parameter,  which makes it very computationally costly and highly ineffective in terms of memory.  

Gradient Descent requires only first order derivatives of parameters with respect to the loss function, which is efficiently calculated by Backpropagation, and so it shines above these techniques because of its simplicity and efficiency.

**Variations of Gradient Descent**

There are other variations of Gradient Descent with few ideas added to allow faster convergence to the optimum. The most popular algorithms are:

1. Stochastic Gradient Descent (SGD)
2. SGD with Momentum 
3. Nestorov's Accelerated Gradient (NAG)
4. Adaptive gradient (AdaGrad)
5. RMSprop
6. Adam

---

### Stochastic Gradient Descent

When training input is very large, gradient descent is quite slow to converge. Stochastic Gradient Descent is the preferred variation of gradient descent which estimates the gradient from a small sample of randomly chosen training input in each iteration called minibatches.

**Minibatches**

Minibatches are generated by shuffling the training data and randomly selecting certain number of training samples. This number of samples is called **minibatch size** and is a parameter to SGD. Here is the code:

```python
def get_minibatches(X,y,minibatch_size):
    m = X.shape[0]
    minibatches = []
    X,y = shuffle(X,y)
    for i in range (0,m,minibatch_size):
        X_batch = X[i:i+minibatch_size,:,:,:]
        y_batch = y[i:i+minibatch_size,]
        minibatches.append((X_batch,y_batch))
    return minibatches
```

**Update Rule**

SGD uses a very simple update rule to change the parameters along the negative gradient. Assume we have a list learnable parameters for each layer in order ``params`` and a similar list for gradients ``grads`` calculated by backward pass with tuples of gradients for each learnable parameter, our simple update rule would be:

```python
def vanilla_update(params,grads,learning_rate=0.01):
    for param,grad in zip(params,reversed(grads)): # grads are in opposite order of params
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]
```

**SGD**

Every complete exposure of the training dataset is called **epoch**. The SGD algorithm iterates for a given number of epochs. It uses the above ```get_minibatches``` and ```vanilla_update``` functions to brings together the requirements for our model solver.

```python
def sgd(nnet,X_train,y_train,minibatch_size,epoch,learning_rate,verbose=True,\
        X_test=None,y_test=None):
    minibatches = get_minibatches(X_train,y_train,minibatch_size)
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i+1))
        for X_mini, y_mini in minibatches: 
            loss,grads = nnet.train_step(X_mini,y_mini)
            vanilla_update(nnet.params,grads,learning_rate = learning_rate)
        if verbose:
            train_acc = accuracy(y_train,nnet.predict(X_train))
            test_acc = accuracy(y_test,nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".\
                  format(loss,train_acc,test_acc))
    return nnet
```



---

### Momentum

Momentum technique is an approach which provides an update rule that is motivated from the physical perspective of optimization. Imagine a ball in a hilly terrain is trying to reach the deepest valley. When the slope of the hill is very high, the ball gains a lot of momentum and is able to pass through slight hills in its way. As the slope decreases the momentum and speed of the ball decreases, eventually coming to rest in the deepest position of valley.

This technique modifies the standard SGD by introducing *velocity* $$v$$ , which is the parameter we are trying to optimize, and *friction* $$\mu$$, which tries to control the velocity and prevents overshooting the valley while allowing faster descent. The gradient only has direct influence on the velocity, which in turn has an effect on the position. Mathematically,




$$
\begin{align}
v &= \mu v - \alpha \Delta L \\
w &= w + v
\end{align}
$$


Which translates to code as

```python
def momentum_update(velocity,params,grads,learning_rate=0.01,mu=0.9):
    for v,param,grad, in zip(velocity,params,reversed(grads)):
        for i in range(len(grad)):
            v[i] = mu*v[i] + learning_rate * grad[i]
            param[i] -= v[i]
```



The advantage of momentum is that it makes very small change to SGD but provides a big boost to speed of learning. We need to store the velocity for all the parameters, and use this velocity for making the updates. Here is the modified function for SGD which uses the above momentum update rule.

```python
def sgd_momentum(nnet,X_train,y_train,minibatch_size,epoch,learning_rate,mu = 0.9,\
                verbose=True,X_test=None,y_test=None):
    minibatches = get_minibatches(X_train,y_train,minibatch_size)
    for i in range(epoch):
        loss = 0
        velocity = []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        if verbose:
            print("Epoch {0}".format(i+1))
        for X_mini, y_mini in minibatches:
            loss,grads = nnet.train_step(X_mini,y_mini)
            momentum_update(velocity,nnet.params,grads,learning_rate=learning_rate,mu=mu)
        if verbose:
            train_acc = accuracy(y_train,nnet.predict(X_train))
            test_acc = accuracy(y_test,nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".\
                  format(loss,train_acc,test_acc))
    return nnet
```



**Nesterov's Accelerated Gradient**

Nesterov's Accelerated Gradient is a clever variation of momentum that works slightly better than standard momentum. The idea behind Nesterov's momentum is that instead of calculating the gradient at the current position, we calculate the gradient at a position that we know our momentum is about to take us, called as "look ahead" position. From physical perspective, it makes sense to make judgements about our final position based on the position that we know we are going to be in a short while.

The implementation makes a slight modification to standard SGD Momentum by nudging our parameters slightly in the direction of the velocity and calculating the gradients there. Here is the code:

```python
def sgd_momentum(nnet,X_train,y_train,minibatch_size,epoch,learning_rate,mu = 0.9,\
                verbose=True,X_test=None,y_test=None,nesterov = False):
    
    minibatches = get_minibatches(X_train,y_train,minibatch_size)
    
    for i in range(epoch):
        loss = 0
        velocity = []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        if verbose:
            print("Epoch {0}".format(i+1))
        
        for X_mini, y_mini in minibatches:
			# if nesterov is enabled, nudge the params forward by momentum
            # to calculate the gradients in a "look ahead" position
            if nesterov:
                for param,ve in zip(nnet.params,velocity):
                    for i in range(len(param)):
                        param[i] += mu*ve[i]
           
            loss,grads = nnet.train_step(X_mini,y_mini) 
            momentum_update(velocity,nnet.params,grads,learning_rate=learning_rate,mu=mu)
        
        if verbose:
            train_acc = accuracy(y_train,nnet.predict(X_train))
            test_acc = accuracy(y_test,nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(loss,train_acc,test_acc))
    return nnet
```



---

### Adaptive Learning Rate

Until now we have used a global and equal learning rate for all our parameters. So all of our parameters are being updated with constant factor. But what if we could speed up or slow down this factor, even for each parameter, as the training progresses? We could adaptively tune the learning throughout the training phases and know which direction to accelerate and which to decelerate. Several methods that use such adaptive learning rates have been proposed, most notably AdaGrad, RMSprop and ADAM.

**AdaGrad** 

AdaGrad ([original paper](http://jmlr.org/papers/v12/duchi11a.html)) keeps track of per parameter sum of squared gradient and normalizes parameter update step. The idea is that parameters which receive big updates will have their effective learning rate reduced, while parameters which receive small updates will have their effective learning rate increased. This way we can accelerate the convergence by accelerating per parameter learning.

```python
def adagrad_update(cache,params,grads,learning_rate=0.01):
    for c,param,grad, in zip(cache,params,reversed(grads)):
        for i in range(len(grad)):
            cache[i] += grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i])+1e-8) # for preventing divide by 0
            
```

**RMSprop**

A disadvantage of AdaGrad is that ```cache[i] += grad[i]**2``` part of the update is monotonically increasing. This can pose problems because the learning rate can steadily decrease to the point where it stops the learning altogether. RMSprop (unpublished, [citation here](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)) combats this problem by decaying the past squared gradient by a factor ```decay_rate``` to control the aggressive learning rates. Here ```decay_rate``` is a hyperparameter with typical values like 0.9,0.99 and so on.

```python
def rmsprop_update(cache,params,grads,learning_rate=0.01,decay_rate=0.9):
    for c,param,grad, in zip(cache,params,reversed(grads)):
        for i in range(len(grad)):
            cache[i] = decay_rate * cache[i] + (1-decay_rate) * grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i])+1e-4)
```

**Adam**

Adam ([original paper](http://arxiv.org/abs/1412.6980)) is a recently proposed and currently state of the art first order optimization algorithm. It is an improvement upon RMSprop by adding momentum to the update rule, combining best of the both momentum and adaptive learning worlds. We introduce two more parameters ```beta1``` and ```beta2``` with recommended values 0.9 and 0.999 respectively.

Another thing to note is that Adam includes *bias correction* mechanism, which compensates for first few iterations when both cache and velocity are biased at zero as they are initialized to zero. 

Here is the full implementation of Adam:

```python

def adam(nnet,X_train,y_train,minibatch_size,epoch,learning_rate,verbose=True,\
        X_test=None,y_test=None,beta1=0.9,beta2=0.999):
    
    minibatches = get_minibatches(X_train,y_train,minibatch_size)
    for i in range(epoch):
        loss = 0
        velocity,cache = [],[]
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)
            cache.append(p)
        if verbose:
            print("Epoch {0}".format(i+1))
        t = 1
        for X_mini, y_mini in minibatches: 
            loss,grads = nnet.train_step(X_mini,y_mini)
            for c,v,param,grad, in zip(cache,velocity,nnet.params,reversed(grads)):
                for i in range(len(grad)):
                    c[i] = beta1 * c[i] + (1-beta1) * grad[i]
                    mt = c[i] / (1 - beta1**t)
                    v[i] = beta2 * v[i] + (1-beta2) * (grad[i]**2)
                    vt = v[i] / (1 - beta2**t)
                    print(vt)
                    param[i] += - learning_rate * mt / (np.sqrt(vt) + 1e-4)
            t+=1

        if verbose:
            train_acc = accuracy(y_train,nnet.predict(X_train))
            test_acc = accuracy(y_test,nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".\
                  format(loss,train_acc,test_acc))
    return nnet
```







