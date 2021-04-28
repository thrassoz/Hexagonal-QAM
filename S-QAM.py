import numpy as np
import matplotlib.pyplot as plt

x_axis = np.array([-0.5, 0.5])
y_axis = np.array([-1.5, -0.5, 0.5, 1.5])

symbols = np.array(np.random.choice(x_axis, 100)) + 1j*np.array(np.random.choice(y_axis, 100))
n = (np.random.randn(100) + 1j*np.random.randn(100))/np.sqrt(2) # AWGN with unity power
noise_power = 0.001
r = symbols + n * np.sqrt(noise_power)
plt.plot(np.real(r), np.imag(r), '.')
plt.grid(True)
plt.show()