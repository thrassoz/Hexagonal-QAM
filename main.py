import numpy as np
import matplotlib.pyplot as plt

num_symbols = 1000

x_int = np.random.randint(0, 4, num_symbols)
x_degrees = x_int*360/4 + 45
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_amplitude = np.array([2, 3])
x_symbols1 = np.cos(x_radians)*x_amplitude[0] + 1j*np.sin(x_radians)*x_amplitude[0]# this produces our QPSK complex symbols
x_symbols2 = np.cos(x_radians)*x_amplitude[1] + 1j*np.sin(x_radians)*x_amplitude[1]

x_symbols = np.concatenate((x_symbols1, x_symbols2))

n = (np.random.randn(num_symbols*2) + 1j*np.random.randn(num_symbols*2))/np.sqrt(2) # AWGN with unity power
noise_power = 0.005
r = x_symbols + n * np.sqrt(noise_power)
plt.plot(np.real(r), np.imag(r), '.')
plt.grid(True)
plt.show()

