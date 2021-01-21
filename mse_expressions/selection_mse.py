def selection_mse(source_powers, noise_power, coarray_length, number_of_snapshots):
    e = (sum(source_powers) + noise_power) ** 2 / number_of_snapshots
    return coarray_length ** 2 * e