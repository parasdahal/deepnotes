---
layout: post
title: BatchNorm Layer - Understanding and eliminating Internal Covariance Shift
comments: true
categories: cnn-series
---
{% include series.html %}
We know that feature scaling makes the job of gradient descent easy and allows it to converge faster. Feature scaling is performed as a pre-processing task on the dataset. But once the normalized input is fed to the deep network, as each layer is affected by parameters in all the input layer, even a small change in the network parameter is amplified and leads to the input distribution being changed in the internal layers of the network. This is known as internal covariance shift.

Batch Normalization is an idea introduced by Ioffe & Szegedy in 2015 ([original paper](http://arxiv.org/pdf/1502.03167v3.pdf)) of normalizing activations of every fully connected and convolution layer with unit standard deviation and zero mean during training, as a part of the network architecture itself. It allows us to use much higher learning rates and be less careful about network initialization.

It is implemented as a layer (with trainable parameters) and normalizes the activations of the previous layer. Backpropagation allows the network to learn if they want the activations to be normalized and upto what extent. It is inserted immediately after fully connected or convolutional layers and before nonlinearities. It effectively reduces the internal covariance shift in deep networks. 

**Advantages of BatchNorm**

1. Improves gradient flow through very deep networks
2. Reduces dependency on careful initialization
3. Allows higher learning rates
4. Provides regularization and reduces dependency on dropout

### Forward Propagation

In the forward pass, we calculate the mean and variance of the batch, normalize the input to have unit Gaussian distribution and scale and shift it with the learnable parameters $$\gamma$$ and $$\beta $$, respectively.


$$
\begin{align}
\mu_B &= \frac{1}{m}\sum_{i=1}^{m} x_i \\
\sigma_B^2 &= \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2 \\
\hat{x_i} &= \frac{x_i - \mu_B}{\sqrt{ \sigma_B^2 + \epsilon }} \\
y_i &= \gamma x_i + \beta
\end{align}
$$


The implementation is very simple and straightforward:

```python
n_X,c_X,h_X,w_X = X.shape
X_flat = X.reshape(n_X,c_X*h_X*w_X)

mu = np.mean(X_flat,axis=0)
var = np.var(X_flat, axis=0)
X_norm = (X_flat - mu)/np.sqrt(var + 1e-8)

out = gamma * X_norm + beta
```

### Backward Propagation

For our backward pass, we need to find gradients $$\frac{\partial C}{\partial x_i}$$,  $$\frac{\partial C}{\partial \gamma}$$ and  $$\frac{\partial C}{\partial \beta}$$. We calculate the intermediate gradients from top to bottom in the computational graph to get these gradients.


$$
\begin{align}
\frac{\partial C}{\partial \gamma \hat{x_i}} &= \frac{\partial C}{\partial y_i} \times \frac{\partial y_i}{\partial \gamma x_i} \\
&= \frac{\partial C}{\partial y_i} \times \frac{\partial (\gamma x_i + \beta)}{\partial \gamma x_i} \\
&= \frac{\partial C}{\partial y_i}
\end{align}
$$


$$
\begin{align}
\frac{\partial C}{\partial \beta} &= \frac{\partial C}{\partial y_i} \times \frac{\partial y_i}{\partial \beta} \\
&= \frac{\partial C}{\partial y_i} \times \frac{\partial(\gamma x_i + \beta )}{\partial \beta} \\
&= \sum_{i=1}^m \frac{\partial C}{\partial y_i}
\end{align}
$$


$$
\begin{align}
\frac{\partial C}{\partial \gamma} &= \frac{\partial C}{\partial \gamma \hat{x_i}} \times  \frac{\partial \gamma \hat{x_i}}{\partial \gamma} \\
&= \sum_{i=1}^m \frac{\partial C }{\partial y_i} \times \hat{x_i}
\end{align}
$$


Now we have gradients for both the learnable parameters. Now for input gradient,

$$
\begin{align}
\frac{\partial C}{\partial \hat{x_i}} &= \frac{\partial C}{\partial \gamma x_i} \times \frac{\partial \gamma x_i}{\partial x_i} \\
&= \frac{\partial C}{\partial y_i} \times \gamma \\
\end{align}
$$


$$
\begin{align}
\frac{\partial C}{\partial \sigma_B^2} &= \frac{\partial C}{\partial \hat{x_i}} \times \frac{\partial \hat{x_i} }{\partial \sigma_B^2} \\
&= \frac{\partial C}{\partial \hat{x_i}} \times \frac{\partial \left ( \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \right ) }{\partial \sigma_B^2 } \\
&= \sum_{i=1}^m  \frac{\partial C}{\partial \hat{x_i}} \times (x_i - \mu_B) \times \frac{\partial (\sigma_B^2 + \epsilon)^{-1/2}}{\partial \sigma_B^2} \\
&= \sum_{i=1}^m  \frac{\partial C}{\partial \hat{x_i}} \times (x_i - \mu_B) \times  -\frac{1}{2} \times (\sigma_B^2 + \epsilon)^{-3/2}
\end{align}
$$

We can see from the computation graph, $$\mu_B$$ is on two nodes, so we need to add up gradients on both nodes.

$$
\begin{align}
\frac{\partial C}{\partial \mu_B} &= \frac{\partial C}{\partial \hat{x_i}} \times \frac{\partial \hat{x_i}}{\partial \mu_b} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{\partial \sigma_B^2}{\partial \mu_B} \\
&= \frac{\partial C}{\partial \hat{x_i} } \times \frac{\partial \left ( \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \right ) }{\partial \mu_B} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{\partial \left (\frac{1}{m} \sum_{i=0}^m (x_i - \mu_B) \right)^2 }{\partial \mu_B} \\
&= \sum_{i=1}^m \frac{\partial C}{\partial \hat{x_i}}\times \frac{-1}{\sqrt{\sigma_B^2}+\beta} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{1}{m}\sum_{i=1}^m 2(x_i - \mu_B)
\end{align}
$$

Now we have all the intermediate gradients to calculate input gradient. Since $$x_i$$ is in three nodes, we add up the gradients on each of those nodes.

$$
\begin{align}
\frac{\partial C}{\partial x_i} &= \frac{\partial C}{\partial \hat{x_i}} \times \frac{\partial \hat{x_i}}{\partial x_i} + \frac{\partial C}{\partial \mu_B} \times \frac{\partial \mu_B}{\partial x_i} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{\partial \sigma_B^2}{\partial x_i} \\
&= \frac{\partial C}{\partial \hat{x_i}} \times \frac{1}{\sqrt{\sigma_B^2+ \beta}} + \frac{\partial C}{\partial \mu_B} \times \frac{\partial \frac{1}{m}\sum_{i=1}^m x_i} {\partial \mu_B} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{2}{m}(x_i - \mu_B) \\
&= \frac{\partial C}{\partial \hat{x_i}} \times \frac{1}{\sqrt{\sigma_B^2+ \beta}} + \frac{\partial C}{\partial \mu_B} \times \frac{1}{m} + \frac{\partial C}{\partial \sigma_B^2} \times \frac{2}{m}(x_i - \mu_B) \\
\end{align}
$$

Translating the gradient expressions in python, we have our implementation of backprop through the BatchNorm layer:

```python
n_X,c_X,h_X,w_X = X.shape
# flatten the inputs and dout
X_flat = X.reshape(n_X,c_X*h_X*w_X)
dout = dout.reshape(n_X,c_X*h_X*w_X)

X_mu = X_flat - mu
var_inv = 1./np.sqrt(var + 1e-8)
        
dX_norm = dout * gamma
dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (var + 1e-8)**(-3/2)
dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/n_X * np.sum(-2.* X_mu, axis=0)

dX = (dX_norm * var_inv) + (dmu / n_X) + (dvar * 2/n_X * X_mu)
dbeta = np.sum(dout,axis=0)
dgamma = dout * X_norm
```

### Source code

Here is the source code for BatchNorm layer with forward and backward API implemented.

```python
class Batchnorm():

    def __init__(self,X_dim):
        self.d_X, self.h_X, self.w_X = X_dim
        self.gamma = np.ones((1, int(np.prod(X_dim)) ))
        self.beta = np.zeros((1, int(np.prod(X_dim))))
        self.params = [self.gamma,self.beta]

    def forward(self,X):
        self.n_X = X.shape[0]
        self.X_shape = X.shape
        
        self.X_flat = X.ravel().reshape(self.n_X,-1)
        self.mu = np.mean(self.X_flat,axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta
        
        return out.reshape(self.X_shape)

    def backward(self,dout):

        dout = dout.ravel().reshape(dout.shape[0],-1)
        X_mu = self.X_flat - self.mu
        var_inv = 1./np.sqrt(self.var + 1e-8)
        
        dbeta = np.sum(dout,axis=0)
        dgamma = dout * self.X_norm

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (self.var + 1e-8)**(-3/2)
        dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/self.n_X * np.sum(-2.* X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / self.n_X) + (dvar * 2/self.n_X * X_mu)
        
        dX = dX.reshape(self.X_shape)
        return dX, [dgamma, dbeta]
```

