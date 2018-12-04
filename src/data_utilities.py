import numpy as np
import torch



def process_input(data: dict, meta_data: dict):
    tensor = data['tensor']
    tensor = np.asarray(tensor, dtype=np.float64)
    normalized_tensor = normalize_and_remove_outliers(tensor, meta_data)
    return normalized_tensor


# We first remove outliers based on the new dataset.
# However, we normalize based on the original training data.
# This is to make sure we're consistent in values fed into the network.
def normalize_and_remove_outliers(data: np.ndarray, meta_data: dict):
    dimensions = meta_data['io_dimension']
    norm_boundaries = meta_data['norm_boundaries']
    for i in range(dimensions):
        numerator = np.subtract(data[:, i], norm_boundaries[i]['min'])
        data[:, i] = np.divide(numerator, norm_boundaries[i]['max'] - norm_boundaries[i]['min'])
    return data


def select_key_variables(key_variables, tensor: torch.Tensor):
    if key_variables:
        filtered_tensors = []
        if len(tensor.shape) == 3:
            for feature in key_variables:
                index = feature['index']
                filtered_tensors.append(tensor[:, :, index])
            if isinstance(tensor, torch.Tensor):
                filtered_tensor = torch.stack(filtered_tensors, dim=2)
            else:
                filtered_tensor = np.stack(filtered_tensors, axis=2)
        else:
            for feature in key_variables:
                index = feature['index']
                filtered_tensors.append(tensor[:, index])
            if isinstance(tensor, torch.Tensor):
                filtered_tensor = torch.stack(filtered_tensors, dim=1)
            else:
                filtered_tensor = np.stack(filtered_tensors, axis=1)
    else:
        filtered_tensor = tensor
    return filtered_tensor
