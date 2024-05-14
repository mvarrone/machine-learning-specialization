import math, copy

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("./deeplearning.mplstyle")

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

import time

# Load our data set

x_train = np.array([1.0, 2.0])  # features

y_train = np.array([300.0, 500.0])  # target value

# Function to calculate the cost


def compute_cost(x, y, w, b):

    m = x.shape[0]

    cost = 0

    for i in range(m):

        f_wb = w * x[i] + b

        cost = cost + (f_wb - y[i]) ** 2

    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    """

    Computes the gradient for linear regression

    Args:

      x (ndarray (m,)): Data, m examples

      y (ndarray (m,)): target values

      w,b (scalar)    : model parameters

    Returns

      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w

      dj_db (scalar): The gradient of the cost w.r.t. the parameter b

    """

    # Number of training examples

    m = x.shape[0]

    dj_dw = 0

    dj_db = 0

    for i in range(m):

        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]

        dj_db_i = f_wb - y[i]

        dj_db += dj_db_i

        dj_dw += dj_dw_i

    dj_dw = dj_dw / m

    dj_db = dj_db / m

    return dj_dw, dj_db


plt_gradients(x_train, y_train, compute_cost, compute_gradient)

plt.show()


def gradient_descent(
    x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function
):
    """

    Performs gradient descent to fit w,b. Updates w,b by taking

    num_iters gradient steps with learning rate alpha



    Args:

      x (ndarray (m,))  : Data, m examples

      y (ndarray (m,))  : Target values

      w_in,b_in (scalar): Initial values of model parameters

      alpha (float)     : Learning rate

      num_iters (int)   : Number of iterations to run gradient descent

      cost_function     : Function to call to produce cost

      gradient_function : Function to call to produce gradient



    Returns:

      w (scalar)      : Updated value of parameter after running gradient descent

      b (scalar)      : Updated value of parameter after running gradient descent

      J_history (List): History of cost values

      p_history (list): History of parameters [w,b]

    """

    # An array to store cost J and w's at each iteration primarily for graphing later

    J_history = []

    p_history = []

    b = b_in

    w = w_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters using gradient_function

        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above

        w = w - alpha * dj_dw

        b = b - alpha * dj_db

        # Save cost J at each iteration

        if i < 100000:  # prevent resource exhaustion

            J_history.append(cost_function(x, y, w, b))

            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10

        if i % math.ceil(num_iters / 10) == 0:

            print(
                f"Iteration {i:4}: Cost J: {J_history[-1]:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}",
            )

    return w, b, J_history, p_history  # return w and J,w history for graphing


def measure_elapsed_time(start_time):

    total_time = round(time.time() - start_time, 2)

    if total_time < 1:

        unit = "ms"

        total_time = total_time * 1000

    else:

        unit = "s"

    return total_time, unit


# initialize parameters

w_init = 0

b_init = 0


# some gradient descent settings

iterations = 10000

tmp_alpha = 1.0e-2


start_time = time.time()

# run gradient descent

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient,
)

total_time, unit = measure_elapsed_time(start_time)

print(f"\nElapsed time: {total_time} {unit}")

print(f"(w,b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")

print(f"\nfw_b(x) = wx + b = {w_final:8.4f}x + {b_final:8.4f}")

# plot cost versus iteration

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

ax1.plot(J_hist[:100])

ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])

ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")

ax1.set_ylabel("Cost")
ax2.set_ylabel("Cost")

ax1.set_xlabel("iteration step")
ax2.set_xlabel("iteration step")

plt.show()

# Test data

x_test = np.array([1.2, 1.8, 2.4])

for value in x_test:

    print(
        f"{value * 1000} sqft house prediction = {w_final*value + b_final:0.1f} Thousand dollars"
    )


# print(f"\n{w_final = }")

# print(f"{b_final = }")


# Predict prices for test data

y_test = []

for value in x_test:

    prediction = w_final * value + b_final

    y_test.append(prediction)


# Convert y_test to numpy array

y_test = np.array(y_test)


# Print predictions and data types

print("\nPredictions for test data:")

print(f"{x_test = }")

print(f"{y_test = }")

# print("Data type of y_test:", type(y_test))


# Plot the data points

# print(f"{x_train = }")

# print(f"{y_train = }")

# print(f"{w_final = }")

# print(f"{b_final = }")

# print(f"{x_test = }")

# print(f"{y_test = }")


# Plot features

plt.scatter(x_train, y_train, label="Actual data", c="b")


# Plot predictions

plt.scatter(x_test, y_test, label="Predictions", c="k")


# Plot f(x)

x_line = np.linspace(min(x_train), max(x_test), 100)

y_line = w_final * x_line + b_final

plt.plot(x_line, y_line, color="g", label=f"f(x) = {w_final:.2f}*x + {b_final:.2f}")


# Set labels and title

plt.xlabel("Size (in sqft)")

plt.ylabel("Price (in 1000s of dollars)")

plt.title("Plotting actual data, f(x) and predictions")


# Add legend

plt.legend()


# Show grid and then plot

plt.grid(True)

plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

plt_contour_wgrad(x_train, y_train, p_hist, ax)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

plt_contour_wgrad(
    x_train,
    y_train,
    p_hist,
    ax,
    w_range=[180, 220, 0.5],
    b_range=[80, 120, 0.5],
    contours=[1, 5, 10, 20],
    resolution=0.5,
)

# initialize parameters

w_init = 0

b_init = 0


iterations = 10


# set alpha to a large value

tmp_alpha = 8.0e-1


# run gradient descent

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient,
)

plt_divergence(p_hist, J_hist, x_train, y_train)

plt.show()
