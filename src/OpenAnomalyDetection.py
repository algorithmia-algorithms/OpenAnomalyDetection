from src import network_processing, graph, forecast
import numpy as np

class InputFormat:
    data_path = None
    graph_save_path = None
    max_sigma = 2
    model_path = ""
    calibration_percentage = 0.1
    def __init__(self, data):
        if 'data_path' in data:
            self.data_path = type_check(data, 'data_path', str)
        else:
            raise network_processing.AlgorithmError("'data_path' was either not found, or not a string")
        if 'model_input_path' in data:
            self.model_path = type_check(data, 'model_input_path', str)
        if 'max_sigma' in data:
            self.max_sigma = type_check(data, 'max_sigma', float)
        if 'calibration_percentage' in data:
            self.calibration_percentage = type_check(data, 'calibration_percentage', float)
        if 'graph_save_path' in data:
            self.graph_save_path = type_check(data, 'graph_save_path', str)

def type_check(dic, id, type):
    if isinstance(dic[id], type):
        return dic[id]
    else:
        raise network_processing.AlgorithmError("'{}' must be of {}".format(str(id), str(type)))



def find_anomalies(result, max_sigma):
    point_anomalies = []
    error_mean = result['summary']['error']['mean']
    error_std = result['summary']['error']['std']
    error_max = error_mean + error_std*max_sigma
    for obs in result['info']:
        error = obs['error']
        if error >= error_max:
            sigma = (error - error_mean) / error_std
            anomaly = {'sigma': sigma, 'index': obs['index']}
            point_anomalies.append(anomaly)
    return point_anomalies

def convert_to_anomalous_regions(point_anomalies, anomaly_radius, threshold):
    anom_gaussians = []
    for point in point_anomalies:
        index = point['index']
        sigma = point['sigma']
        anom_gaussian = {'lower': index - anomaly_radius, 'upper': index + anomaly_radius, 'sigma': sigma}
        anom_gaussians.append(anom_gaussian)

    processed_anomalies = []
    already_inspected = []
    for anomaly in anom_gaussians:
        if anomaly not in already_inspected:
            interfering_anomalies = find_interfering_anomalies(anomaly, anom_gaussians, list(), threshold)
            already_inspected += interfering_anomalies
            if len(interfering_anomalies) == 0:
                processed_anomalies.append(anomaly)
            else:
                conglomerate_anomaly = correct_interference(interfering_anomalies)
                processed_anomalies.append(conglomerate_anomaly)

    return processed_anomalies

def find_interfering_anomalies(specimen, input_anomalies, interfering_anomalies, threshold):
    for anomaly_y in input_anomalies:
        if anomaly_y != specimen:
            if detect_interference(specimen, anomaly_y, threshold):
                if specimen not in interfering_anomalies:
                    interfering_anomalies.append(specimen)
                if anomaly_y not in interfering_anomalies:
                    interfering_anomalies.append(anomaly_y)
                removed_duplicates = [elm for elm in input_anomalies if elm not in interfering_anomalies]
                find_interfering_anomalies(anomaly_y, removed_duplicates, interfering_anomalies, threshold)
    return interfering_anomalies

def correct_interference(interfering_anomalies):
    corrected = {'sigma': 0}
    for inter_anom in interfering_anomalies:
        # corrected['confidence'] += inter_anom['confidence']
        corrected['sigma'] += inter_anom['sigma']
        if 'upper' not in corrected or corrected['upper'] < inter_anom['upper']:
            corrected['upper'] = inter_anom['upper']
        if 'lower' not in corrected or corrected['lower'] > inter_anom['lower']:
            corrected['lower'] = inter_anom['lower']
    # corrected['confidence'] /= len(interfering_anomalies)
    corrected['sigma'] /= len(interfering_anomalies)
    return corrected


def detect_interference(elm_x, elm_y, threshold):
    if elm_x['upper'] in range(elm_y['lower'], elm_y['upper']) or elm_x['lower'] in range(
            elm_y['lower'], elm_y['upper']):
        if within_threshold(elm_x, elm_y, threshold):
            return True
    else:
        return False

def within_threshold(elm_x, elm_y, threshold):
    if elm_x['sigma'] <= elm_y['sigma']+threshold and elm_x['sigma'] >= elm_y['sigma']-threshold:
        return True
    else:
        return False

def calc_num_evals(dataframe, coverage_percentage):
    seq_length = dataframe.shape[0]
    steps = seq_length
    num_evals = int(steps * coverage_percentage)
    return num_evals



def apply(input):
    guard = InputFormat(input)
    threshold = 1
    data_path = network_processing.get_data(guard.data_path)
    data = network_processing.load_json(data_path)
    data = np.asarray(data['tensor'])
    model, meta = network_processing.get_model_package(guard.model_path)
    forecaster = forecast.Model(meta, model)
    result = forecaster.execute(data, guard.calibration_percentage)
    anomalies = find_anomalies(result, guard.max_sigma)
    anomalous_regions = convert_to_anomalous_regions(anomalies, meta['forecast_length'] * 2, threshold)
    if guard.graph_save_path:
        local_graph_path = graph.graph_anomalies(anomalous_regions, data)
        remote_file_path = network_processing.put_file(local_graph_path, guard.graph_save_path)
        output = {'graph_save_path': remote_file_path, 'anomalous_regions': anomalous_regions}
    else:
        output = {'anomalous_regions': anomalous_regions}
    return output
