#!/usr/bin/env python3
from src.OpenAnomalyDetection import *
import os

def test_detection_algorithmia():
    input = dict()
    input['data_path'] = 'data://TimeSeries/GenerativeForecasting/m4_daily.json'
    input['model_input_path'] = 'data://TimeSeries/GenerativeForecasting/m4_daily_0.1.0.zip'
    input['graph_save_path'] = 'data://.my/replaceme/graph_file1.png'
    input['sigma_threshold'] = 3
    input['variable_index'] = 3
    input['calibration_percentage'] = 0.1
    result = apply(input)
    first_anomaly = result['anomalous_regions'][0]
    print(result)
    assert first_anomaly['max_sigma'] >= first_anomaly['avg_sigma']
    assert first_anomaly['avg_sigma'] >= input['sigma_threshold']
    assert os.path.isfile(result['graph_save_path'])


def test_detection_local():
    input = dict()
    input['data_path'] = 'file://tmp/m4_daily.json'
    input['model_input_path'] = 'file://tmp/m4_daily_0.1.0.zip'
    input['graph_save_path'] = 'file://tmp/graph_file1.png'
    input['sigma_threshold'] = 3
    input['variable_index'] = 3
    input['calibration_percentage'] = 0.1
    result = apply(input)
    first_anomaly = result['anomalous_regions'][0]
    print(result)
    assert first_anomaly['max_sigma'] >= first_anomaly['avg_sigma']
    assert first_anomaly['avg_sigma'] >= input['sigma_threshold']
    assert os.path.isfile(result['graph_save_path'])



if __name__ == "__main__":
    test_detection_local()