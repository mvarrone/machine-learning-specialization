# Vectorization (optional)

## 1. How neural networks are implemented efficiently

One of the reasons that deep learning researchers have been able to scale up neural networks, and thought really large neural networks over the last decade, is because neural networks can be vectorized.

They can be implemented very efficiently using matrix multiplications and it turns out that parallel computing hardware, including GPUs but also some CPU functions, are very good at doing very large matrix multiplications.

In this video, we'll take a look at how these vectorized implementations of neural networks work. Without these ideas, I don't think deep learning would be anywhere near a success and scale today

## Vectorization of code (using np.matmul())

![alt text](./images_for_10/image1.png)

### Non-vectorized implementation (See *1-non-vectorized-code.ipynb*)

On the left, is the code we developed previously of how to implement forward propagation in a single layer

For this code, X is the input, W the weights of the 1st, 2nd and 3rd neurons and we have the parameters b. Result of this implementation is [1, 0, 1]

### Vectorized implementation (See *2-vectorized-code.ipynb*)

On the right, is a vectorized implementation of this code.

For this new code, X is a 2D array like in TensorFlow, W is the same as before, B is also a 1x3 2D array and all of the functions steps can be replaced with just a couple of lines of code. 

Now, we can use $np.matmul()$ Numpy function to carry out matrix multiplication. Result of this implementation is [[1, 0, 1]]

## Recap 

This turns out to be a very efficient implementation of one step of forward propagation through a dense layer in the neural network. 

This is code for a vectorized implementation of forward prop in a neural network

## 2. Matrix multiplication
