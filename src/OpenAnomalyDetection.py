import scipy.stats as st
from src import algo_processing, data_processing, graphing, forecast


class InputFormat:
    data_path = None
    max_sigma = 2
    model_path = ""
    calibration_percentage = 0.0
    view_size = 15
    resolution_step_size = 5
    zero = False
    def __init__(self, data):
        if 'data_path' in data and isinstance(data['data_path'], str):
            self.data_path = data['data_path']
        else:
            raise algo_processing.AlgorithmError("'data_path' was either not found, or not a string")
        if 'model_path' in data and isinstance(data['model_path'], str):
            self.model_path = data['model_path']
        if 'max_sigma' in data and isinstance(data['max_sigma'], float):
            self.max_sigma = data['max_sigma']
        if 'calibration_percentage' in data and isinstance(data['calibration_percentage'], float):
            self.calibration_percentage = data['calibration_percentage']
        if 'view_size' in data and isinstance(data['view_size'], int):
            self.view_size = data['view_size']
        if 'resolution_step_size' in data and isinstance(data['resolution_step_size'], int):
            self.resolution_step_size = data['resolution_step_size']
        if 'zero_state' in data and isinstance(data['zero_state'], str):
            if data['zero_state'] == "True" or data['zero_state'] == "true":
                zero = True
            else:
                zero = False

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
        confidence = st.norm.cdf(sigma)
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
    sideeffect_prefix = 'data://.algo/timeseries/anomalydetection/temp'
    threshold = 1
    dataframe = data_processing.get_sequence(guard.data_path)
    result = forecast.execute(dataframe, guard.model_path, guard.view_size, guard.resolution_step_size, guard.calibration_percentage, guard.zero)
    anomalies = find_anomalies(result, guard.max_sigma)
    anomalous_regions = convert_to_anomalous_regions(anomalies, guard.view_size * 2, threshold)
    local_graph_path = graphing.graph_anomalies(anomalous_regions, dataframe)
    remote_file_path = algo_processing.upload_image(local_graph_path, sideeffect_prefix)
    output = {'graph_path': remote_file_path, 'anomalous_regions': anomalous_regions}
    return output
