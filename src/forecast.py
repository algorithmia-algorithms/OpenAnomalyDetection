
import math
from ergonomics import serialization
from src.algo_processing import get_file
import torch
from torch.autograd import Variable
from time import perf_counter
import numpy as np
def execute(dataframe, model_path, forecast_range, step_size, calibration_percentage, zero_state):
    local_model = get_file(model_path)
    max_bound = dataframe.shape[0] - forecast_range * 2 - step_size * 2
    net = serialization.load_portable(local_model)
    if zero_state:
        net.zero()
    data = Variable(torch.from_numpy(dataframe), requires_grad=False).float()
    if calibration_percentage > 0:
        min_bound = math.ceil(calibration_percentage*data.shape[0])
        cal_data = data[0:min_bound, :]
        anom_data = data[min_bound:, :]
        net.forward(input=cal_data)
    else:
        anom_data = data
    max_bound = anom_data.shape[0] - forecast_range * 2 - step_size * 2
    forecast_outputs = []
    start = perf_counter()
    # net.forward(data[0:min_bound, :])
    for i in range(0, max_bound, step_size):
        next_point = anom_data[i:i + step_size, :]
        net.forward(input=next_point)
        lock_state = net.get_state()
        target = anom_data.data[i + step_size:i + step_size + forecast_range, :]
        pred = net.forecast(future=forecast_range)
        pred = revert_normalization(pred, lock_state)
        net.load_mutable_state(lock_state)
        total_error = 0
        for j in range(anom_data.shape[1]):
            dim_pred = pred[:, j]
            dim_target = target[:, j]
            error = criterion(dim_pred, dim_target)
            total_error += error
        total_error /= anom_data.shape[1]
        intermediate = {
            'error': total_error,
            'index': i + step_size
        }
        print("completed: {}, error was {}".format(str(i), str(total_error)))
        forecast_outputs.append(intermediate)
    result = process_output_advanced(forecast_outputs)
    complete = perf_counter()
    print("total compute time: {}s".format(str(complete-start)))
    return result


def process_output_advanced(results):
    metadata = {}
    metadata['error'] = {}
    metadata['error']['mean'] = 0
    metadata['error']['std'] = None
    metadata['error']['max'] = {'value': 0, 'id': None}
    metadata['error']['min'] = {'value': -1, 'id': None}
    errors = []
    for result in results:
        errors.append(result['error'])
        metadata['error']['mean'] += result['error']
        if result['error'] > metadata['error']['max']['value']:
            metadata['error']['max']['value'] = result['error']
            metadata['error']['max']['id'] = result['index']
        if result['error'] < metadata['error']['min']['value'] or metadata['error']['min']['value'] == -1:
            metadata['error']['min']['value'] = result['error']
            metadata['error']['min']['id'] = result['index']
    metadata['error']['mean'] /= len(results)
    metadata['error']['std'] = np.std(np.asarray(errors))
    output = {
        'info': results,
        'summary': metadata
    }
    return output


def revert_normalization(data, state):
    norm_boundaries = state['norm_boundaries']
    io_shape = state['io_width']
    output = np.empty(data.shape, float)
    for i in range(io_shape):
        min = norm_boundaries[i]['min']
        max = norm_boundaries[i]['max']
        multiplier = max-min
        intermediate = np.multiply(data.data[:, i], multiplier)
        result = np.add(intermediate, min)
        output[:, i] = result
    return output

def criterion(prediction, target):
    error_rate = 0
    for j in range(len(prediction)):
        out = math.pow(prediction[j] - target[j], 2)
        error_rate += out
    error_rate /= len(prediction)
    return error_rate
