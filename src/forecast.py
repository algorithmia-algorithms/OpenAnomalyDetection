
import math
import torch
from torch.autograd import Variable
from torch import from_numpy
from time import perf_counter
import numpy as np


class Model:


    def __init__(self, meta_data, network=None):
        self.residual_shape = meta_data['tensor_shape']['residual']
        self.memory_shape = meta_data['tensor_shape']['memory']
        self.data_dimensionality = meta_data['io_dimension']
        self.key_variables = meta_data['key_variables']
        self.forecast_length = meta_data['forecast_length']
        self.training_time = meta_data['training_time']
        self.network = network

    def execute(self, data: np.ndarray, calibration_percentage: float):
        tensor = convert_to_torch_tensor(data)
        init_residual = generate_state(self.residual_shape)
        init_memory = generate_state(self.memory_shape)

        if calibration_percentage > 0:
            min_bound = math.ceil(calibration_percentage*tensor.shape[0])
            cal_data = tensor[0:min_bound, :]
            anom_data = tensor[min_bound:, :]
            _, working_residual, working_memory = self.update(cal_data, init_residual, init_memory)
        else:
            anom_data = data
            working_residual = init_residual
            working_memory = init_memory
        max_bound = anom_data.shape[0] - self.forecast_length * 2
        forecast_outputs = []

        # for i in range(0, max_bound):
        #     next_point = anom_data[i:i+1, :]
        #     _, working_residual, working_memory = self.network(next_point, working_residual, working_memory)
        #     lock_state = net.get_state()
        #     target = anom_data.data[i + step_size:i + step_size + forecast_range, :]
        #     pred = net.forecast(future=forecast_range)
        #     pred = revert_normalization(pred, lock_state)
        #     net.load_mutable_state(lock_state)
        #     total_error = 0
        #     for j in range(anom_data.shape[1]):
        #         dim_pred = pred[:, j]
        #         dim_target = target[:, j]
        #         error = criterion(dim_pred, dim_target)
        #         total_error += error
        #     total_error /= anom_data.shape[1]
        #     intermediate = {
        #         'error': total_error,
        #         'index': i + step_size
        #     }
        #     print("completed: {}, error was {}".format(str(i), str(total_error)))
        #     forecast_outputs.append(intermediate)
        anorm_subset = anom_data[0:max_bound, :]
        x, y = self.segment_data(anorm_subset)
        h = self.forecast_every_step(x, working_residual, working_memory)
        y_f = self.select_key_variables(y)
        h_f = self.select_key_variables(h)
        deviations = criterion(y_f, h_f)
        for i in range(deviations.shape[0]):
            for j in range(deviations.shape[1]):
                deviation = deviations[i, j]
                intermediate = {'error': deviation, 'index': i, 'dimension': j}
                forecast_outputs.append(intermediate)
        results = process_output_advanced(forecast_outputs)
        return results

    def update(self, x: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor):
        h_t = None
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            h_t, residual, memory = self.network(x_t, residual, memory)
        return h_t, residual, memory

    def forecast_step(self, residual: torch.Tensor, memory: torch.Tensor, last_step: torch.Tensor):
        residual_forecast = residual.clone()
        memory_forecast = memory.clone()
        forecast_tensor = torch.zeros(self.forecast_length, self.data_dimensionality)
        forecast_tensor[0] = last_step
        for i in range(1, self.forecast_length):
            x_t = forecast_tensor[i - 1]
            next_step, residual_forecast, memory_forecast = self.network(x_t, residual_forecast, memory_forecast)
            forecast_tensor[i] = next_step
        return forecast_tensor

    def forecast_every_step(self, x, residual, memory):
        h = []
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            h_t, residual, memory = self.network(x_t, residual, memory)
            h_n = self.forecast_step(residual, memory, h_t[-1])
            h.append(h_n)
        h = torch.stack(h)
        return h

    def select_key_variables(self, tensor: torch.Tensor):
        if self.key_variables:
            filtered_tensors = []
            if len(tensor.shape) == 3:
                for feature in self.key_variables:
                    index = feature['index']
                    filtered_tensors.append(tensor[:, :, index])
                filtered_tensor = torch.stack(filtered_tensors, dim=2)
            else:
                for feature in self.key_variables:
                    index = feature['index']
                    filtered_tensors.append(tensor[:, index])
                filtered_tensor = torch.stack(filtered_tensors, dim=1)
        else:
            filtered_tensor = tensor
        return filtered_tensor

    def segment_data(self, data: torch.Tensor):
        segments = []
        for i in range(data.shape[0] - (self.forecast_length + 1)):
            segment = data[i + 1:i + self.forecast_length + 1]
            segments.append(segment)
        x = data[:-(self.forecast_length + 1)]
        y = torch.stack(segments)
        return x, y



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
    for j in range(prediction.shape[1]):
        out = (prediction[:, j] - target[:, j]).pow(2)
        error_rate += out
    error_rate /= prediction.shape[0]
    error_rate = error_rate.detach().numpy()
    return error_rate


def convert_to_torch_tensor(data: np.ndarray):
    return Variable(from_numpy(data)).float()

def generate_state(shape: tuple):
    tensor = torch.zeros(shape)
    return tensor