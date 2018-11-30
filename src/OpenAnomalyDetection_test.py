#!/usr/bin/env python3
from src.OpenAnomalyDetection import *
import os

def test_detect():
    input = dict()
    input['data_path'] = 'data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json'
    input['model_input_path'] = 'file://tmp/rossman_0.1.0.zip'
    input['graph_save_path'] = 'file://tmp/graph_file.png'
    input['max_sigma'] = 1.5
    input['calibration_percentage'] = 0.1
    result = apply(input)
    assert os.path.isfile(result['graph_save_path'])

    result = apply(input)
    assert result


if __name__ == "__main__":
    test_detect()