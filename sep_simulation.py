import numpy as np
import matplotlib.pyplot as plt
from HQAM import hqam,IrrHQAM
import sorting
import timeit
import math
from scipy.stats import gamma

def proposal(dmin, No):
    R = dmin/2
    p = 1 - gamma.cdf(R**2,1,0,No) #cdf(x, a, loc=0, scale=1)
    return p

def existing(dmin, No, K, Kc): # K, Kc are unique numbers for every M-Hqam
    sigma = math.sqrt(No/2)
    return K*(0.5 - math.erf(dmin/(2*sigma)/math.sqrt(2))/2) + (2/3)*Kc*(0.5 - math.erf(dmin/(sigma*math.sqrt(6))/math.sqrt(2))/2)**2 - 2*Kc*(0.5 - math.erf(dmin/(2*sigma)/math.sqrt(2))/2)*(0.5 - math.erf(dmin/(2*math.sqrt(3)*sigma)/math.sqrt(2))/2)

def existing2(dmin, No, K):
    sigma = math.sqrt(No/2)
    return K*(0.5 - math.erf(dmin/(2*sigma)/math.sqrt(2))/2)

noise = np.arange(0.1, 1, 0.02)
y1 = np.array([])
y2 = np.array([])
y3 = np.array([])
for i in noise:
    temp = np.array([proposal(2, i)])
    temp2 = np.array([existing(2, i, 33/8, 27/8)])
    temp3 = np.array([existing2(2, i, 33/8)])
    y1 = np.concatenate((y1,temp))
    y2 = np.concatenate((y2,temp2))
    y3 = np.concatenate((y3,temp3))
# constellation = IrrHQAM(16, 2)  # Generate Constellation with nn_dist = 2
# sim = np.array([])
# for i in noise:
#     idx =  0
#     errors = 0
#     while idx < 10000000:
#         n = np.random.normal(0,i,1) + 1j*np.random.normal(0,i,1)  # AWGN np.random.normal(mean, sigma, size)
#         #by changing sigma we change noise power and because energy per symol is fixed due to standard nn_dist we can 
#         #change SNR parameter
#         send_symbol = np.random.choice(constellation, 1)
#         r = send_symbol + n  # received symbol
#         #MLD
#         argmax = float('-inf') # argmax {dotproduct(r,s) - energy(s)/2}
#         candidate = 0 + 0j
#         for symbol in constellation:
#             if np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2 > argmax:
#                 argmax = np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2
#                 canditate = symbol
        
#         if canditate != send_symbol:
#             errors += 1
#         idx += 1
#     temp = np.array([errors/idx])
#     sim = np.concatenate((sim, temp))
    

# plt.plot(noise, sim, '-')
# plt.gca().set_yscale('log')
plt.plot(noise, y1, 's')
plt.gca().set_yscale('log')
plt.plot(noise, y2, 'h')
plt.gca().set_yscale('log')
plt.plot(noise, y3, '-')
plt.gca().set_yscale('log')
plt.legend(["proposed", "existing", "existing2"])
plt.ylabel('SEP of 16-HQAM')
plt.xlabel("No")
plt.show()