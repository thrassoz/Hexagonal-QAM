import numpy as np
import matplotlib.pyplot as plt
from HQAM import hqam
import sorting
import timeit

#create the constellation and set it as global variable
constellation = hqam(16, 2)

def initialize_x_values():
    x = []
    for symbol in constellation:
        if np.real(symbol) not in x:
            x.append(np.real(symbol))
    x = sorting.merge(x)
    return x

def initialize_x_map():
    x_axis = initialize_x_values()
    x_map = {

    }
    for x in x_axis:
        x_map[x] = []
    for symbol in constellation:
        x_map[np.real(symbol)].append(symbol)
    return x_map

def find_nearest(array, r):
    dist = float('inf')
    nearest = 0
    nearest_index = 0
    temp = 0
    for x in array:
        if abs(r-x) < dist:
            dist = abs(r-x)
            nearest = x
            nearest_index = temp
        temp += 1
    return nearest_index
   


if __name__ == '__main__':
    x_map = initialize_x_map()
    x_values = []
    for item in x_map:
        x_values.append(item)

    idx = 0
    errors = 0
    start = timeit.default_timer()
    while idx<1000:
        #create a received symbol with AWGN
        n = (np.random.randn(1) + 1j*np.random.randn(1))/np.sqrt(2)  # AWGN with unity power No = 1
        noise_power = 0.5 # SNR parameter
        send_symbol = np.random.choice(constellation, 1) #symbol that we sent
        r = send_symbol + n*np.sqrt(noise_power)  # received symbol
        #detection algo
        x_neighbour = find_nearest(x_values, np.real(r)) #index of nearest x_value
        x_region = [float('-inf'), float('inf')]
        #i have to change if arguments because sometimes it gets out of bounds
        if x_neighbour == 0 and np.real(r) > x_values[x_neighbour]:
            x_region[0] = x_values[x_neighbour]
            x_region[1] = x_values[x_neighbour+1]
        elif x_neighbour == 0 and np.real(r) < x_values[x_neighbour]:
            x_region[1] = x_values[x_neighbour]
        elif x_neighbour == len(x_values)-1 and np.real(r) > x_values[x_neighbour]:
            x_region[0] = x_values[x_neighbour]
        elif x_neighbour == len(x_values)-1 and np.real(r) < x_values[x_neighbour]:
            x_region[0] = x_values[x_neighbour-1]
            x_region[1] = x_values[x_neighbour]
        elif np.real(r) == x_values[x_neighbour]:
            x_region[0] = x_values[x_neighbour]
            x_region[1] = x_values[x_neighbour]
        elif np.real(r) < x_values[x_neighbour] and x_neighbour != len(x_values)-1 and x_neighbour != len(x_values)-1:
            x_region[0] = x_values[x_neighbour-1]
            x_region[1] = x_values[x_neighbour]
        elif np.real(r) > x_values[x_neighbour] and x_neighbour != len(x_values)-1 and x_neighbour != len(x_values)-1:
            x_region[0] = x_values[x_neighbour]
            x_region[1] = x_values[x_neighbour+1]
        
        argmax = float('-inf') # argmax {dotproduct(r,s) - energy(s)/2}
        canditate = 0 + 0j
        if x_region[0] != float('-inf') and x_region[0] != float('inf'):
            for symbol in x_map[x_region[0]]:
                if (np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2 > argmax):
                    argmax = np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2
                    canditate = symbol
        if x_region[1] != float('-inf') and x_region[1] != float('inf'):
            for symbol in x_map[x_region[1]]:
                if (np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2 > argmax):
                    argmax = np.real(r)*np.real(symbol)+np.imag(r)*np.imag(symbol) - (np.real(symbol)**2 + np.imag(symbol)**2)/2
                    canditate = symbol
        
        if canditate != send_symbol:
            errors += 1
            
        idx += 1
    stop = timeit.default_timer()
    print(errors, stop-start)
    