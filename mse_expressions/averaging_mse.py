import numpy as np
def averaging_mse(thetas, source_powers, noise_power, coarray_length, number_of_snapshots, Jdict, element_locations, channel):
    L = element_locations.shape[0]
    repetitions = np.kron(element_locations, np.ones(L, ))
    edict = {}
    for key, J in Jdict.items():
        edict[key] = err_n(thetas, source_powers, noise_power, J, repetitions, number_of_snapshots, channel)
    err = 0
    for m in range(1, coarray_length+1):
        for n in range(1 - m, coarray_length-m+1):
            err += edict[n]
    return err[0]

def err_n(thetas, source_powers, noise_power, J, repetitions, number_of_snapshots, channel):
    J = list(J)
    e = 2 * noise_power * sum(source_powers) + noise_power ** 2
    e = e / len(J)
    for i in J:
        for j in J:
            z = zvector(thetas, repetitions, i, j, channel)
            e += np.abs(z[:, None].T @ source_powers) ** 2 / len(J) ** 2
    e = e / number_of_snapshots
    return e

def omega(repetitions, i, j):
    return repetitions[i]-repetitions[j]

def zvector(thetas, repetitions, i, j, channel):
    (carrier_frequency, propagation_speed) = channel
    K = thetas.shape[0]
    const = 2 * np.pi * carrier_frequency / propagation_speed
    z = np.zeros(K, ) + 1j * np.zeros(K, )
    om = omega(repetitions, i, j)
    for k in range(K):
        z[k] = np.exp( - 1j * const * np.sin(thetas[k]) * om)
    return z