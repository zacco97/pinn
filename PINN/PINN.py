import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Sequential, optimizers


# oscilator for underdumped state (we know the equation)


def oscilator(d, w0, x):

    assert d < w0
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = tf.cos(phi+w*x)
    sin = tf.sin(phi+w*x)
    exp = tf.exp(-d*x)
    y = exp*2*A*cos
    return y


def nn(n_input, n_output, n_hidden, n_layers):
    model = Sequential()
    model.add(layers.Dense(n_input, input_shape=(1, 1)))
    for i in range(n_layers):
        model.add(layers.Dense(n_hidden, activation="tanh",
                  name="dense{}".format(i)))
    model.add(layers.Dense(n_output, name="output"))
    return model


def get_r(model, mu, k, x):

    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        yhp = tf.reshape(model(x), (-1, 1))
        dx = tf.cast(g.gradient(yhp, x), dtype=tf.float32)
    dx2 = tf.cast(g.gradient(dx, x), dtype=tf.float32)

    phisics = dx2 + mu*dx + k*yhp
    del g

    return phisics


def compute_loss(model, mu, k, x1, x_data, y_data):
    r = get_r(model, mu, k, x1)
    loss2 = tf.reduce_mean(tf.square(r))
    y_pred = tf.reshape(model(x_data), (-1, 1))
    loss1 = tf.reduce_mean((y_pred-y_data)**2)

    # print(loss1.dtype, loss2.dtype, y_pred.dtype)
    # print(loss1 , loss2)
    loss = loss1 + loss2
    return loss


def get_grad(model, mu, k, x1, x_data, y_data):
    with tf.GradientTape(persistent=True) as g:
        g.watch(model.trainable_variables)
        loss = compute_loss(model, mu, k, x1, x_data, y_data)
    tape = g.gradient(loss, model.trainable_variables)
    del g
    return loss, tape


def train_step(model, mu, k, x1, x_data, y_data, adam):
    loss, grad = get_grad(model, mu, k, x1, x_data, y_data)

    adam.apply_gradients(zip(grad, model.trainable_variables))
    return loss


def trainNN():
    d, w0 = 2, 20
    n_input, n_output, n_hidden, n_layers = 1, 1, 32, 3
    lr_sch = 10**-3

    x = tf.linspace(0, 1, 500, dtype=tf.float32)
    x = tf.reshape(x, (-1, 1))
    y = oscilator(d, w0, x)
    y = tf.reshape(y, (-1, 1))

    x_data = x[0:400:20]
    y_data = y[0:400:20]

    model = nn(n_input, n_output, n_hidden, n_layers)
    model.summary()
    adam = optimizers.Adam(learning_rate=lr_sch)
    model.compile(loss="mse", optimizer=adam)
    history = model.fit(x_data, y_data, epochs=2000)
    pred = model.predict(x)
    pred = tf.reshape(pred, (-1, 1))
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, linewidth=2, label="exact solution")
    axs.plot(x, pred, color="r", linewidth=2, label="prediction")
    axs.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    d, w0 = 2, 20
    mu, k = 2*d, w0**2
    hist = []
    lr_sch = 10**-3
    N = 10000

    x = tf.linspace(0, 1, 500)
    x = tf.cast(tf.reshape(x, (-1, 1)), dtype=tf.float32)

    y = oscilator(d, w0, x)
    y = tf.cast(tf.reshape(y, (-1, 1)), dtype=tf.float32)

    x_data = x[0:400:20]
    y_data = y[0:400:20]

    # print(x_data.dtype)

    x1 = tf.linspace(0, 1, 30)
    x1 = tf.reshape(x1, (-1, 1))

    n_input, n_output, n_hidden, n_layers = 1, 1, 32, 3
    model = nn(n_input, n_output, n_hidden, n_layers)
    #get_r(model, mu, k, x1)
    adam = optimizers.Adam(learning_rate=lr_sch)

    for i in range(N+1):
        loss = train_step(model, mu, k, x1, x_data, y_data, adam)
        hist.append(loss.numpy())
        if i % 50 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(i, loss))

    pred = model.predict(x)
    pred = tf.reshape(pred, (-1, 1))
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, linewidth=2, label="exact solution")
    axs.plot(x, pred, color="r", linewidth=2, label="prediction")
    axs.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()
