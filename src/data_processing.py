import numpy as np




def process_input(data: dict, multiplier: float, meta_data: dict):
    tensor = data['tensor']
    tensor = np.asarray(tensor, dtype=np.float64)
    normalized_tensor = normalize_and_remove_outliers(tensor, multiplier, meta_data)
    return normalized_tensor


# We first remove outliers based on the new dataset.
# However, we normalize based on the original training data.
# This is to make sure we're consistent in values fed into the network.
def normalize_and_remove_outliers(data: np.ndarray, multiplier: float, meta_data: dict):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    dimensions = meta_data['io_dimension']
    norm_boundaries = meta_data['norm_boundaries']
    for i in range(dimensions):
        for j in range(len(data[:, i])):
            max_delta = mean[i] - multiplier * sd[i]
            if not (data[j, i] > max_delta):
                print('clipped {} for being too far above the mean.'.format(str(data[j, i])))
                data[j, i] = max_delta
            elif not (-data[j, i] > max_delta):
                print('clipped {} for being too far below the mean.'.format(str(data[j, i])))
                data[j, i] = -max_delta
    for i in range(dimensions):
        numerator = np.subtract(data[:, i], norm_boundaries[i]['min'])
        data[:, i] = np.divide(numerator, norm_boundaries[i]['max'] - norm_boundaries[i]['min'])

    return data



