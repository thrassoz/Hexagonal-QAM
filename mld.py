import numpy as np
import matplotlib.pyplot as plt
from HQAM import hqam
import sorting
import timeit

constellation = hqam(16, 2)

if __name__ == '__main__':
    idx =  0
    errors = 0
    while idx < 1000:
        n = (np.random.randn(1) + 1j*np.random.randn(1))/np.sqrt(2)  # AWGN with unity power No = 1
        noise_power = 0.5  # SNR parameter
        send_symbol = np.random.choice(constellation, 1) #symbol that we sent
        r = send_symbol + n*np.sqrt(noise_power)  # received symbol

        argmax = float('-inf') # argmax {dotproduct(r,s) - energy(s)/2}
        candidate = 0 + 0j
        for symbol in constellation:
            if np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2 > argmax:
                argmax = np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2
                canditate = symbol
        
        if canditate != send_symbol:
            errors += 1
            print(send_symbol, r)
        idx += 1
    print(errors)
        # plt.plot(np.real(r), np.imag(r), '.')
        # plt.grid(True)
        # plt.show()
