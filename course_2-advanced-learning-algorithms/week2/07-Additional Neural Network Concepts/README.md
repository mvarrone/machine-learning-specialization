## Advanced Optimization

Gradient descent is an optimization algorithm that is widely used in machine learning and was the foundation of many algorithms like Linear Regression and Logistic /Regression and early implementations of neural networks. 

But it turns out that there are now some other optimization algorithms for minimizing the cost function that are even better than Gradient descent. 

We will take a look at an algorithm that can help to train a neural network much faster than Gradient descent: It is called *Adam algorithm*

### Recalling Gradient descent algoritm

![alt text](./img/image1.png)

Let's recall that the expression for one step of Gradient descent for the parameter $w$:

$$w_j = w_j - \alpha \frac{\partial }{\partial w_j}J(\vec{w}, b)$$

How can we make this work even better? In this example, we have plotted the Cost function J using a contour plot comprising these ellipsis. We know the minimum of this cost function is at the center of this ellipsis. 

Now, if you were to start Gradient descent down here (marked on graph as "start"), if $\alpha$ is small then one step of gradient descent may take you a little bit in that direction. Then another step, then another step, then another step, then another step, and you notice that every single step of gradient descent is pretty much going in the same direction. In that case, if you see this you might wonder, why don't we make $\alpha$ bigger?

Can we have an algorithm to automatically increase the value of $\alpha$? They just make it take bigger steps and get to the minimum faster. There's an algorithm called the Adam algorithm that can do that. 

Depending on how gradient descent is proceeding, sometimes you wish you had a bigger learning rate $\alpha$ and sometimes you wish you had a smaller learning rate $\alpha$.

The Adam algorithm can adjust the learning rate $\alpha$ automatically.

### Adam algorithm

Adam stands for **Ada**ptive **m**oment estimation

![alt text](./img/image2.png)

### Increasing $\alpha$
If it sees that the learning rate $\alpha$ is too small and we are just taking tiny little steps in a similar direction over and over, we should just make the learning rate $\alpha$ bigger. 

### Decreasing $\alpha$
In contrast taking a look at the same cost function J if we were starting here and have a relatively big learning rate $\alpha$, then maybe one step of gradient descent takes us here, in the second step takes us here, third step, and the fourth step, and the fifth step, and the sixth step, and if you see Gradient descent doing this, is oscillating back and forth. You'd be tempted to say, well, why don't we make the learning rate $\alpha$ smaller? The Adam algorithm can also do that automatically, and with a smaller learning rate, you can then take a more smooth path toward the minimum of the cost function. 

![alt text](./img/image3.png)

But interestingly, the Adam algorithm does not use a single global learning rate $\alpha$: Instead, it uses a different learning rate for **every single parameter** of your model. 

If you have parameters $w_1$ through $w_{10}$ as well as b, then it actually has 11 learning rate parameters

It would have $\alpha_1$, $\alpha_2$, all the way through $\alpha_{10}$ for $w_1$ to $w_{10}$, as well as $\alpha_{11}$ for the parameter $b$.

$$w_1 = w_1 - \alpha_1 \frac{\partial }{\partial w_1}J(\vec{w}, b)$$
$$...$$
$$w_{10} = w_{10} - \alpha_{10} \frac{\partial }{\partial w_{10}}J(\vec{w}, b)$$
$$b = b - \alpha_{11} \frac{\partial }{\partial b}J(\vec{w}, b)$$

## Adam algorithm's intuition

![alt text](./img/image4.png)

The intuition behind the Adam algorithm is:

- If a parameter $w_j$ or $b$ seems to keep on moving in roughly the same direction, let's increase the learning rate $\alpha$ for that parameter. Let's go faster in that direction. 

- Conversely, if a parameter keeps oscillating back and forth, then let's not have it keep on oscillating or bouncing back and forth. Let's reduce $\alpha_j$ for that parameter a little bit. 

The details of how Adam does this is a bit complicated and beyond the scope of this course, but if you take some more advanced deep learning courses later, you may learn more about the details of this Adam algorithm but in code this is how you implement it.

![alt text](./img/image5.png)

The model is exactly the same as before and the way you compile the model is very similar to what we had before except that we now added one extra argument to the *compile* function. This is the place where we specify that the **optimizer** you want to use is Adam by using: $$tf.keras.optimizers.Adam()$$ 

```python
...

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=...)

...
```

The Adam optimization algorithm needs some initial default value for the learning rate $\alpha$

## Piece of advice for choosing this global learning rate value

In this example, I've set that initial learning rate to be $\alpha=10^{-3}$ but when using the Adam algorithm in practice, it's worth trying a few values for this default global learning rate. Try some larger and some smaller values to see what gives you the fastest learning performance.

Compared to the original gradient descent algorithm that you had learned in the previous course though, the Adam algorithm is more robust to the exact choice of learning rate that you pick because it can adapt the learning rate a bit automatically. Though there's still way tuning this parameter little bit to see if you can get somewhat faster learning.

It typically works much faster than gradient descent, and it has become a de-facto standard in how practitioners train their neural networks. If you're trying to decide what learning algorithm to use, what optimization algorithm to use to train your neural network a safe choice would be to just use the Adam optimization algorithm, and most practitioners today will use Adam rather than the optional gradient descent algorithm
