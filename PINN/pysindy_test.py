import matplotlib.pyplot as plt 
import numpy as np 
import pysindy as ps 
import tensorflow as tf

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


if __name__ == "__main__":

    d, w0 = 2, 20
    mu, k = 2*d, w0**2

    x = tf.linspace(0, 1, 500).numpy()
    # x1 = tf.cast(tf.reshape(x, (-1, 1)), dtype=tf.float32)

    y = oscilator(d, w0, x).numpy()

    differentiation_method = ps.FiniteDifference(order=2)
    feature_library = ps.PolynomialLibrary(degree=2)
    optimizer = ps.STLSQ()

    model = ps.SINDy(differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,feature_names=["y"])

    t = np.linspace(0, 1, 500)
    #print(t)
    model.fit(y, t=t)
    model.print()
    x0, t0 = 1, 0
    sim = model.simulate([x0], t=t)
    #print(sim)
    plt.plot(t0, x0, "ro", label="Initial condition", alpha=0.6, markersize=8)
    plt.plot(x, y, "b", label="Exact solution", alpha=0.4, linewidth=4)
    plt.plot(sim[:, 0], t, "k--", label="SINDy model", linewidth=3)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.show()