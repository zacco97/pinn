import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

dt = 0.1
t_train = np.arange(0, 10, dt)
x_train = np.sin(t_train)
seed = 1
np.random.seed(seed)
x_noise = x_train + .1*np.random.randn(len(t_train))

# threshold is parameter called lambda
# (if it's too big => coefficients will be eliminated)

opt = ps.optimizers.SINDyPI(threshold=0.1)

# library_functions = [
#     #lambda x : np.exp(x),
#     #lambda x : 1. / x,
#     #lambda x : x,
#     #lambda x : np.sin(x),
#     lambda x : np.cos(x)
# ]
# library_function_names = [
#     #lambda x : 'exp(' + x + ')',
#     #lambda x : '1/' + x,
#     #lambda x : x,
#     #lambda x : 'sin(' + x + ')',
#     lambda x : 'cos(' + x + ')'
# ]


# custom_library = ps.CustomLibrary(
#     library_functions=library_functions, function_names=library_function_names
# )

#library = ps.PolynomialLibrary(degree=4)
library = ps.FourierLibrary(n_frequencies=2)
opt = ps.STLSQ(threshold=0.02, fit_intercept=True)

model = ps.SINDy(differentiation_method= ps.SINDyDerivative(kind="spectral") ,feature_library=library, optimizer=opt)

model.fit(x_noise, t=dt)
model.print()


x_model = model.simulate([0], t = t_train)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(t_train, x_noise, color="red", label="noisy")
ax1.plot(t_train, x_train, color="blue", label="noiseless")
ax1.plot(t_train, x_model[:,0], color="black", label="sindy")
ax1.legend()
plt.show()
