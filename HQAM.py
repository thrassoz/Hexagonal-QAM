import numpy as np
import matplotlib.pyplot as plt
import sorting



def hqam(M, nn_dist=2):
    # M is the order of constellation, nn_dist is the nearest neighbour dist
    # The idea behind this algo is that we will create an M-ary constellation with a center point at(0,-y_1)
    # and then we will shift the constellation backwards in the x-axis in order to achieve the symmetry
    # of the constellation around the origin for a regular HQAM
    # We have to parametrize the func in order to generate different types of HQAM
    # depending on M and nn_dist
    # First we check if M is the square of an even integer
    even = 0  # even number that is even^2=M or the first even number that is even^2>M
    for i in range(2, M, 2):
        if i*i == M:
            even = i
            break
        elif i*i > M:
            even = i
            break

    diff = even**2 - M
    symbols = np.array([])

    # this works only for nn_dist = 2, something went wrong with listing comprehension below (check the bounds)
    # We can observe that we have two possible arrays for the x_axis values
    if diff == 0:
        pos_x_axis1 = np.array([x for x in range(nn_dist//2, even+1) if x % 2 == 1])  # for 64-HQAM
        neg_x_axis1 = np.array([-x for x in range(nn_dist//2, even+1) if x % 2 == 1])
        pos_x_axis2 = np.array([x for x in range(nn_dist//2, even+1) if x % 2 == 0])
        neg_x_axis2 = np.array([-x for x in range(0, even) if x % 2 == 0])  # for 64-HQAM
        # We can observe that we have only one array for y_axis values
        y_unity = np.sqrt(3*(nn_dist**2)/4)  # the height of the basic equilateral triangle
        pos_y_axis = np.array([y_unity/2 + y_unity*i for i in range(0, even//2)])  # for 64-HQAM not exactly xd
        neg_y_axis = np.array([-y_unity/2 - y_unity*(even//2-i-1) for i in range(0, even//2)])

        # build 1st quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even//4).astype(int))*pos_x_axis1[cnt1]  # the real part of the symbol
                temp = temp + 1j*np.array([pos_y_axis[i] for i in range(0, even//2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even//4).astype(int)) * pos_x_axis2[cnt2]
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even//2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # build 2nd quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even//4).astype(int))*neg_x_axis1[cnt1]  # the ceil and astype(int) are added
                # because of the posibility that we want to produce 32-HQAM and therefore even == 6
                temp = temp + 1j*np.array([pos_y_axis[i] for i in range(0, even//2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even//4).astype(int)) * neg_x_axis2[cnt2]
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even//2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # build 3rd quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even//4).astype(int))*neg_x_axis1[cnt1]
                temp = temp + 1j*np.array([neg_y_axis[i] for i in range(0, even//2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even//4).astype(int)) * neg_x_axis2[cnt2]
                temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even//2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # build 4th quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even//4).astype(int))*pos_x_axis1[cnt1]
                temp = temp + 1j*np.array([neg_y_axis[i] for i in range(0, even//2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even//4).astype(int)) * pos_x_axis2[cnt2]
                temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even//2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # Now we will shift the constellation in x-axis direction
        shift = np.sqrt((nn_dist/2)**2 - (y_unity/2)**2)  # calculated by pythagorean theorem
        for k in range(0, len(symbols)):
            symbols[k] -= shift  # by this move we achieve the symmetry
        return symbols
    else:  # diff !=0
        # We will assume that we can create the constellation with even * even points and then we will erase
        # diff points from the constellation, we will erase them depending on their energy

        # First, we have created an even^2 constellation,
        # symbols is an already constructed np.array with points on the complex plane
        # Now , we are going to remove the diff points from the symbols
        pos_x_axis1 = np.array([x for x in range(nn_dist // 2, even + 1) if x % 2 == 1])  # for 64-HQAM
        neg_x_axis1 = np.array([-x for x in range(nn_dist // 2, even + 1) if x % 2 == 1])
        pos_x_axis2 = np.array([x for x in range(nn_dist // 2, even + 1) if x % 2 == 0])
        neg_x_axis2 = np.array([-x for x in range(0, even) if x % 2 == 0])  # for 64-HQAM
        # We can observe that we have only one array for y_axis values
        y_unity = np.sqrt(3 * (nn_dist ** 2) / 4)  # the height of the basic equilateral triangle
        pos_y_axis = np.array([y_unity / 2 + y_unity * i for i in range(0, even // 2)])  # for 64-HQAM not exactly xd
        neg_y_axis = np.array([-y_unity / 2 - y_unity * (even // 2 - i - 1) for i in range(0, even // 2)])

        # build 1st quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis1[cnt1]  # the real part of the symbol
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis2[cnt2]
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # build 2nd quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if column % 2 == 0:
                temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis1[cnt1]
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                symbols = np.concatenate((symbols, temp))
                cnt1 += 1
            else:
                temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis2[cnt2]
                temp = temp + 1j * np.array([pos_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                symbols = np.concatenate((symbols, temp))
                cnt2 += 1
        # build 3rd quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if (even//2) % 2 == 0:
                if column % 2 == 0:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis1[cnt1]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt1 += 1
                else:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis2[cnt2]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt2 += 1
            else:  # exception when we have 32-HQAM even = 6 and even//2 = 3
                if column % 2 == 0:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis2[cnt1]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt1 += 1
                else:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * neg_x_axis1[cnt2]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt2 += 1

        # build 4th quadrant
        cnt1 = 0
        cnt2 = 0
        for column in range(0, even, 1):
            if (even // 2) % 2 == 0:
                if column % 2 == 0:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis1[cnt1]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt1 += 1
                else:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis2[cnt2]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt2 += 1
            else:  # exception when we have 32-HQAM even = 6 and even//2 = 3
                if column % 2 == 0:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis2[cnt1]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 == 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt1 += 1
                else:
                    temp = np.ones(np.ceil(even // 4).astype(int)) * pos_x_axis1[cnt2]
                    temp = temp + 1j * np.array([neg_y_axis[i] for i in range(0, even // 2) if i % 2 != 0])
                    symbols = np.concatenate((symbols, temp))
                    cnt2 += 1

        shift = np.sqrt((nn_dist / 2) ** 2 - (y_unity / 2) ** 2)
        for k in range(0, len(symbols)):
            symbols[k] -= shift  # by this move we achieve the symmetry

        energy = [np.real(symbol)**2 + np.imag(symbol)**2 for symbol in symbols]
        energy = sorting.merge(energy)  # sorted in ascending order O(n*log n) algorithm complexity
        erased = []
        removed = 0
        for k in range(0, len(symbols)):
            symbol_energy = np.real(symbols[k])**2 + np.imag(symbols[k])**2  # calculate energy per symbol while traversing symbols array
            # compare it to the diff-biggest elements of the energy array and if it is inside delete it
            for j in range(len(energy)-1, len(energy)-diff, -1):
                if energy[j] == symbol_energy:
                    removed += 1
                    erased.append(k)
                    break
            if removed == diff:
                break

        print(len(erased))
        symbols = np.delete(symbols, erased)
        print(len(symbols))
        return symbols


def IrrHQAM(M, nn_dist=2):
    # we define as center point (nn_dist/2 , 0)
    regular = hqam(M, 2)  # a python list that contains the symbols of the regular M-HQAM
    y_unity = np.sqrt(3 * (nn_dist ** 2) / 4)
    x_shift = np.sqrt((nn_dist / 2) ** 2 - (y_unity / 2) ** 2)
    for i in range(0, len(regular)):
        regular[i] += x_shift  # by this move we break the symmetry of the regular HQAM and we shift the
        # constellation on the x_axis
        regular[i] -= 1j*y_unity/2
        # by this move we create a constellation row on the y = 0 line

    # after we have made the above shifts on the x and y axis we will insert an extra row and an extra column
    # extra column on -max{np.real(regular)} and extra row on max{abs(np.imag(regular))}
    max_real_part = 0  # will be positive
    min_imag_part = 0  # will be negative
    for i in range(0, len(regular)):
        if np.real(regular[i]) > max_real_part:
            max_real_part = np.real(regular[i])
        if np.imag(regular[i]) < min_imag_part:
            min_imag_part = np.imag(regular[i])
    temp = []
    for i in range(0, len(regular)):
        if np.real(regular[i]) == max_real_part:
            temp.append(-max_real_part + 1j*np.imag(regular[i]))
        if np.imag(regular[i]) == min_imag_part:
            temp.append(np.real(regular[i]) + 1j*abs(min_imag_part))
    temp = np.array(temp)
    temp = np.concatenate((regular, temp))
    #now we will create a hash map between symbols index and its energy
    hash_map = {

    }
    for i in range(len(temp)):
        hash_map[i] = np.real(temp[i])**2 + np.imag(temp[i])**2
    hash_map = sorted(hash_map.items(), key=lambda x: x[1])  # we sort the hash_map with respect to energy
    idx = 0
    irregular = []
    while idx < M:
        irregular.append(temp[hash_map[idx][0]])  # we create the irregular constelation by selecting the
        # the lowest M energies from the hash_map
        idx += 1

    irregular = np.array(irregular)
    return irregular

if __name__ == '__main__':
    constellation = hqam(16, 2)  # Generate Constellation with nn_dist = 2
    n = (np.random.randn(10000) + 1j*np.random.randn(10000))/np.sqrt(2)  # AWGN with unity power No = 1
    noise_power = 0.5  # SNR parameter
    r = np.random.choice(constellation, 10000) + n*np.sqrt(noise_power)  # received symbol
    #energy = [np.real(symbol)**2 + np.imag(symbol)**2 for symbol in constellation]
    #total = 0
    #for k in energy:
        #total += k
    #print(total/64)
    plt.plot(np.real(r), np.imag(r), '.')
    plt.grid(True)
    plt.show()
    
