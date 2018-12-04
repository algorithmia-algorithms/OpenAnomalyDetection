from src import network_processing, data_processing, graph, forecast

class Parameters:
    def __init__(self):
        self.data_path = None
        self.graph_save_path = None
        self.sigma_threshold = 2
        self.model_path = ""
        self.variable_index = 1
        self.calibration_percentage = 0.1


def process_input(input):
    parameters = Parameters()
    if 'data_path' in input:
        parameters.data_path = type_check(input, 'data_path', str)
    else:
        raise network_processing.AlgorithmError("'data_path' was either not found, or not a string")
    if 'model_input_path' in input:
        parameters.model_path = type_check(input, 'model_input_path', str)
    if 'sigma_threshold' in input:
        parameters.sigma_threshold = type_check(input, 'sigma_threshold', [float, int])
    if 'calibration_percentage' in input:
        parameters.calibration_percentage = type_check(input, 'calibration_percentage', [float, int])
    if 'graph_save_path' in input:
        parameters.graph_save_path = type_check(input, 'graph_save_path', str)
    if 'variable_index' in input:
        parameters.variable_index = type_check(input, 'variable_index', int)
    return parameters

def type_check(dic, id, typedef):
    if isinstance(typedef, type):
        if isinstance(dic[id], typedef):
            return dic[id]
        else:
            raise network_processing.AlgorithmError("'{}' must be of {}".format(str(id), str(typedef)))
    else:
        for i in range(len(typedef)):
            if isinstance(dic[id], typedef[i]):
                return dic[id]
        raise network_processing.AlgorithmError("'{}' must be of {}".format(str(id), str(typedef)))


def find_point_anomalies(statistics, data, forecast_step, sigma_threshold):
    point_anomalies = []
    error_mean = statistics['mean']
    error_std = statistics['std']
    error_max = error_mean + error_std * sigma_threshold
    for i in range(data.shape[0]):
        error = data[i]
        if error >= error_max:
            sigma = (error - error_mean) / error_std
            anomaly = {'sigma': sigma, 'index': i+forecast_step}
            point_anomalies.append(anomaly)
    return point_anomalies

def convert_to_anomalous_regions(point_anomalies, anomaly_radius, threshold):
    anom_gaussians = []
    for point in point_anomalies:
        index = point['index']
        sigma = point['sigma']
        anom_gaussian = {'lower': index - anomaly_radius, 'upper': index + anomaly_radius, 'avg_sigma': sigma, 'max_sigma': sigma}
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

def find_interfering_anomalies(specimen, input_anomalies, interfering_anomalies, similar_sigma_threshold):
    for anomaly_y in input_anomalies:
        if anomaly_y != specimen:
            if detect_interference(specimen, anomaly_y, similar_sigma_threshold):
                if specimen not in interfering_anomalies:
                    interfering_anomalies.append(specimen)
                if anomaly_y not in interfering_anomalies:
                    interfering_anomalies.append(anomaly_y)
                removed_duplicates = [elm for elm in input_anomalies if elm not in interfering_anomalies]
                find_interfering_anomalies(anomaly_y, removed_duplicates, interfering_anomalies, similar_sigma_threshold)
    return interfering_anomalies

def correct_interference(interfering_anomalies):
    corrected = {'avg_sigma': 0}
    for inter_anom in interfering_anomalies:
        # corrected['confidence'] += inter_anom['confidence']
        corrected['avg_sigma'] += inter_anom['avg_sigma']
        if 'upper' not in corrected or corrected['upper'] < inter_anom['upper']:
            corrected['upper'] = inter_anom['upper']
        if 'lower' not in corrected or corrected['lower'] > inter_anom['lower']:
            corrected['lower'] = inter_anom['lower']
    # corrected['confidence'] /= len(interfering_anomalies)
    corrected['avg_sigma'] /= len(interfering_anomalies)
    corrected['max_sigma'] = max([anom['max_sigma'] for anom in interfering_anomalies])
    return corrected


def detect_interference(elm_x, elm_y, similar_sigma_threshold):
    if elm_x['upper'] in range(elm_y['lower'], elm_y['upper']) or elm_x['lower'] in range(
            elm_y['lower'], elm_y['upper']):
        if within_threshold(elm_x, elm_y, similar_sigma_threshold):
            return True
    else:
        return False

def within_threshold(elm_x, elm_y, similar_sigma_threshold):
    if elm_x['avg_sigma'] <= elm_y['avg_sigma']+similar_sigma_threshold or elm_x['avg_sigma'] >= elm_y['avg_sigma']-similar_sigma_threshold:
        return True
    else:
        return False

def calc_num_evals(dataframe, coverage_percentage):
    seq_length = dataframe.shape[0]
    steps = seq_length
    num_evals = int(steps * coverage_percentage)
    return num_evals



def apply(input):
    guard = process_input(input)
    similar_sigma_threshold = 3
    data_path = network_processing.get_data(guard.data_path)
    data = network_processing.load_json(data_path)
    model, meta = network_processing.get_model_package(guard.model_path)
    anomaly_radius = meta['forecast_length'] * 2

    normalized_data = data_processing.process_input(data, meta)
    forecaster = forecast.ForecastModel(meta, model)
    statistics, data = forecaster.execute(normalized_data, guard.calibration_percentage, guard.variable_index)
    anomalies = find_point_anomalies(statistics, data, meta['forecast_length'], guard.sigma_threshold)
    anomalous_regions = convert_to_anomalous_regions(anomalies, anomaly_radius, similar_sigma_threshold)
    if guard.graph_save_path:
        key_data = data_processing.select_key_variables(meta['key_variables'], normalized_data)
        selected_index = key_data[:, guard.variable_index:guard.variable_index+1]
        local_graph_path = graph.graph_anomalies(anomalous_regions, selected_index)
        remote_file_path = network_processing.put_file(local_graph_path, guard.graph_save_path)
        output = {'graph_save_path': remote_file_path, 'anomalous_regions': anomalous_regions}
    else:
        output = {'anomalous_regions': anomalous_regions}
    return output
