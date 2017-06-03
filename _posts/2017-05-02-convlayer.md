---
layout: post
title: Convolution Layer - The core idea behind CNNs
comments: true
categories: cnn-series
---
Convolutional operation takes a patch of the image, and applies a filter by performing a dot product on it. The convolution layer is similar to fully connected layer, but performs convolution operation on input rather than matrix multiplication.

The convolutional layer takes an input volume of:

1. Number of input $$N$$
2. The depth of input $$C$$
3. Height of the input $$H$$
4. Width of the input $$W$$

These hyperparameters control the size of output volume:

1. Number of filters $$K$$
2. Spatial Extent $$F$$
3. Stride length $$S$$
4. Zero Padding $$P$$

The spatial size of output is given by $$(H-F+2P)/S+1 \times (W-F+2P)/S+1$$

**Note:** When $$S=1,P = (F-1)/2$$ preserves the input volume size.

### Forward Propagation

As stated earlier, convolutional layer replaces the matrix multiplication with convolution operation. To compute the pre non linearity for $$i,j^{th}$$ neuron on $$l$$ layer, we have:


$$
\begin{align}
Z_{ij}^{l} &= \sum_{a=0}^{m-1}\sum_{b=0}^{m-1}W_{ab}a_{(i+a)(j+b)}^{l-1}
\end{align}
$$


Naively, for doing our convolutional operation we loop over each image, over each channel and take a dot product at each $$F \times F$$ location for each of our filters. For the sake of efficiency and computational simplicity, what we need to do is gather all the locations that we need to do the convolution operations and get the dot product at each of these locations. Lets examine this with a simple example.

Suppose we have a single image of size $$1\times 1 \times 4 \times 4$$ and a single filter $$1 \times 1 \times \times 2 \times 2$$ and are using $$S = 1$$ and $$P=1$$. After padding the shape of our image is $$1 \times 1 \times 6 \times 6$$. 


$$
\begin{align}
X_{pad} = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 2 & 3  & 0\\
0 &4 & 5 & 6 & 7 & 0\\
0 &8 & 9 & 10 & 11 & 0\\
0 & 12 & 13 & 14 & 15 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
\end{bmatrix}
_{6\times 6}
\end{align}
$$


Now we have $$4-2/1+1=5$$ locations along both width and height, so $$25$$ possible locations to do our convolution. Locations for top edges are


$$
\begin{align}
X_0 =
\begin{bmatrix}
0&0\\
0&0\\
\end{bmatrix}
_{2\times 2}
X_1 =
\begin{bmatrix}
0&0\\
0&1\\
\end{bmatrix}
_{2\times 2}
X_2 =
\begin{bmatrix}
0&0\\
1&2\\
\end{bmatrix}
_{2\times 2}
X_3 =
\begin{bmatrix}
0&0\\
3&0\\
\end{bmatrix}
_{2\times 2}
\end{align}
$$


For all the $$25$$ locations we have a $$1\times 2 \times 2$$ filter, which we stretch out to $$4 \times 1$$ column vector. Thus we have 25 of these column vectors, or $$4 \times 25$$ matrix of all the stretched out receptive fields. 


$$
\begin{align}
X_{col} = 
\begin{bmatrix}
0&0&0&0&0&0&0&1&2&3&0&4&5&6&7&0&8&9&10&11&0&12&13&14&15\\
0&0&0&0&0&0&1&2&3&0&4&5&6&7&0&8&9&10&11&0&12&13&14&15&0\\
0&0&1&2&3&0&4&5&6&7&0&8&9&10&11&0&12&13&14&15&0&0&0&0&0\\
0&1&2&3&0&4&5&6&7&0&8&9&10&11&0&12&13&14&15&0&0&0&0&0&0\\
\end{bmatrix}
_{4\times 25}
\end{align}
$$


Similarly the weights are also stretched out. If we have 3 filters of size $$1\times 2\times 2$$ then we have matrix $$W_{row}$$ of size $$3\times 4$$. 

The result of convolution is now equivalent of performing a single matrix multiplication ```np.dot(W_row,X_col)``` which has the dot product of every filter with every receptive field, giving the result $$3\times25$$ which can be reshaped back to get output volume of size $$1\times 3\times 5\times 5$$ .

We use [im2col](http://cs231n.stanford.edu/assignments/2016/winter1516_assignment2.zip) utility to perform the reshaping of input X to X_col.

```python
# Create a matrix of size (h_filter*w_filter) by n_X * ((h_X-h_filter+2p)/stride + 1)**2
# suppose X is 5x3x10x10 with 3x3x3 filter and padding and stride = 1
# X_col will be 27x500
X_col = im2col_indices(X,h_filter,w_filter,padding,stride)
# suppose we have 10 filter of size 10x3x3x3, W_col will be 10x27
W_col = W.reshape(n_filter,c_filter*h_filter*w_filter)
# output will be 10x500
output = np.dot(W_col,X_col) + b
# reshape to get 10x10x10x5 and then transpose the axes 5x10x10x10
output = output.reshape(n_filter,h_out,w_out,n_X).transpose(3,0,1,2)
```


### Backward Propagation

We know the output error for the current layer $$\partial out$$ which in our case is $$\frac{\partial C}{\partial Z^l_{ij}}$$ as our layer is only computing pre non linearity output $$Z$$ . We need to find the gradient $$\frac{\partial C}{\partial W_{ab}^{l}}$$ for each weight .


$$
\frac{\partial C}{\partial W_{ab}^{l}} = \sum_{i=0}^{N-m}\sum_{j=0}^{N-m} \frac {\partial C}{\partial Z_{ij^{l}}} \times \frac {\partial Z_{ij}^{l}}{\partial W_{ab^{l}}} \\
 = \sum_{i=0}^{N-m}\sum_{j=0}^{N-m} \frac {\partial C}{\partial Z_{ij^{l}}} \times a^{l-1}_{(i+a)(j+b)}
$$


Notice that $$\frac {\partial Z_{ij}^{l}}{\partial W_{ab^{l}}} = a^{l-1}_{(i+a)(j+b)} $$ is from the forward propagation above, where $$a^{l-1}$$ is the output of the previous layer and input to our current layer.

```python
# from 5x10x10x10 to 10x10x10x5 and 10x500
dout_flat = dout.transpose(1,2,3,0).reshape(n_filter,-1)
# calculate dot product 10x500 . 500x27 = 10x27
dW = np.dot(dout_flat,X_col.T)
# reshape back to 10x3x3x3
dW = dW.reshape(W.shape)
```



For bias gradient, we simply accumulate the gradient as with backpropagation for fully connected layers. So,


$$
\frac{ \partial C}{\partial b^l} = \sum_{i=0}^{N-m}\sum_{j=0}^{N-m} \frac{\partial C}{\partial Z_{ij}^{l}}
$$


```python
db = np.sum(dout,axis=(0,2,3))
db = db.reshape(n_filter,-1)
```



Now to backpropagate the errors back to the previous layer, we need to compute the input gradient $$\partial X$$ which in our case is $$\frac{\partial C}{\partial a^{l-1}_{ij}}$$.

$$
\frac{\partial C}{\partial a_{ij}^{l-1}} = \sum_{a=0}^{m-1}\sum_{b=0}^{m-1} \frac{\partial C}{\partial Z_{(i-a)(j-b)}}\times \frac{\partial Z^l_{(i-a)(j-b)}}{\partial a_{ij}^{l-1}} \\
 = \sum_{a=0}^{m-1}\sum_{b=0}^{m-1} \frac{\partial C}{\partial Z_{(i-a)(j-b)}}\times W_{ab}
$$

Notice this looks similar to our convolution operation from forward propagation step but instead of $$Z_{(i+a)(j+b)}$$ we have $$Z_{(i-a)(j-b)}$$, which is simply  a convolution using $$W$$ which has been flipped along both the axes.
```python
# from 10x3x3x3 to 10x9
W_flat = W.reshape(n_filter,-1)
# dot product 9x10 . 10x500 = 9x500
dX_col = np.dot(W_flat.T,dout_flat)
# get the gradients for real image from the stretched image.
# from the stretched out image to real image i.e. 9x500 to 5x3x10x10
dX = col2im_indices(dX_col,X.shape,h_filter,w_filter,padding,stride) 
```

### Source Code

Here is the source code for convolutional layer with forward and backward API implemented.

```python
class Conv():

    def __init__(self,X_dim,n_filter,h_filter,w_filter,stride,padding):

        self.d_X,self.h_X,self.w_X = X_dim

        self.n_filter,self.h_filter,self.w_filter = n_filter,h_filter,w_filter
        self.stride,self.padding = stride,padding

        self.W = np.random.randn(n_filter,self.d_X,h_filter,w_filter) / np.sqrt(n_filter/2.)
        self.b = np.zeros((self.n_filter,1))
        self.params = [self.W,self.b]

        self.h_out = (self.h_X - h_filter + 2*padding)/ stride + 1
        self.w_out = (self.w_X - w_filter + 2*padding)/ stride + 1
        

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out,self.w_out  = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter,self.h_out,self.w_out)

    def forward(self,X):
        
        self.n_X = X.shape[0]

        self.X_col = im2col_indices(X,self.h_filter,self.w_filter,stride=self.stride,padding=self.padding)        
        W_row = self.W.reshape(self.n_filter,-1)

        out = W_row @ self.X_col + self.b
        out = out.reshape(self.n_filter,self.h_out,self.w_out,self.n_X)
        out = out.transpose(3,0,1,2)
        return out

    def backward(self,dout):

        dout_flat = dout.transpose(1,2,3,0).reshape(self.n_filter,-1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout,axis=(0,2,3)).reshape(self.n_filter,-1)

        W_flat = self.W.reshape(self.n_filter,-1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X,self.d_X,self.h_X,self.w_X)
        dX = col2im_indices(dX_col,shape,self.h_filter,self.w_filter,self.padding,self.stride)

        return dX, [dW, db]
```
