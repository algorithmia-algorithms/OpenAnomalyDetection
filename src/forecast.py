
import math
import torch
from src.data_processing import select_key_variables
from torch.autograd import Variable
from torch import from_numpy
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
        anorm_subset = anom_data[0:max_bound, :]
        x, y = self.segment_data(anorm_subset)
        h = self.forecast_every_step(x, working_residual, working_memory)
        y_f = select_key_variables(self.key_variables, y)
        h_f = select_key_variables(self.key_variables, h)
        deviations = criterion(y_f, h_f)
        results = process_output_advanced(deviations)
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

    def segment_data(self, data: torch.Tensor):
        segments = []
        for i in range(data.shape[0] - (self.forecast_length + 1)):
            segment = data[i + 1:i + self.forecast_length + 1]
            segments.append(segment)
        x = data[:-(self.forecast_length + 1)]
        y = torch.stack(segments)
        return x, y



def process_output_advanced(measurements: np.ndarray):
    metadata = {}
    for i in range(measurements.shape[1]):
        metadata[i] = {}
        metadata[i]['mean'] = 0
        metadata[i]['std'] = None
        metadata[i]['max'] = {'value': 0, 'id': None}
        metadata[i]['min'] = {'value': -1, 'id': None}
        metadata[i]['errors'] = []

    for j in range(measurements.shape[1]):
        for_dim = metadata[j]
        for i in range(measurements.shape[0]):
            error = measurements[i, j]
            for_dim['errors'].append(error)
            for_dim['mean'] += error
            if error > for_dim['max']['value']:
                for_dim['max']['value'] = error
                for_dim['max']['id'] = i
            if error < for_dim['min']['value'] or for_dim['min']['value'] == -1:
                for_dim['min']['value'] = error
                for_dim['min']['id'] = i
        for_dim['mean'] /= len(measurements[:, j])
        for_dim['std'] = np.std(np.asarray(measurements[:, j]))
    output = {
        'info': measurements,
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