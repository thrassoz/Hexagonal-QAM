import numpy as np
import matplotlib.pyplot as plt
from HQAM import hqam
import sorting
import timeit
import math
from scipy.stats import gamma


# K = 33/8
# a = 8/35
# Kc = 27/8
# gs = 18
# snr = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #Es/No
# li = []
# print(K*(0.5 - math.erf(math.sqrt(a*gs)/math.sqrt(2))/2) + (2/3)*Kc*(0.5 - math.erf(math.sqrt(2*a*gs/3)/math.sqrt(2))/2)**2 - 2*Kc*(0.5 - math.erf(math.sqrt(a*gs)/math.sqrt(2))/2)*(0.5 - math.erf(math.sqrt(a*gs/3)/math.sqrt(2))/2))
# for gs in snr:
#     p = K*(0.5 - math.erf(math.sqrt(a*gs)/math.sqrt(2))/2) + (2/3)*Kc*(0.5 - math.erf(math.sqrt(2*a*gs/3)/math.sqrt(2))/2)**2 - 2*Kc*(0.5 - math.erf(math.sqrt(a*gs)/math.sqrt(2))/2)*(0.5 - math.erf(math.sqrt(a*gs/3)/math.sqrt(2))/2)
#     li.append(p)

# plt.plot(snr, li, '.')
# plt.gca().set_yscale('log')
# plt.grid(True)
# plt.show()

K = 33/8
Kc = 27/8
dmin = 2
No = 0.1
sigma = math.sqrt(No/2)
print(K*(0.5 - math.erf(dmin/(2*sigma)/math.sqrt(2))/2) + (2/3)*Kc*(0.5 - math.erf(dmin/(sigma*math.sqrt(6))/math.sqrt(2))/2)**2 - 2*Kc*(0.5 - math.erf(dmin/(2*sigma)/math.sqrt(2))/2)*(0.5 - math.erf(dmin/(2*math.sqrt(3)*sigma)/math.sqrt(2))/2))