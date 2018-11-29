import numpy as np
from src.algo_processing import get_file



def get_sequence(data_path):
    local_path = get_file(data_path)
    dataframe = process_sequence(local_path)
    return dataframe


def is_header(row):
    try:
        _ = float(row[0])
        return False
    except:
        return True

def process_sequence(data_path):
    data = get_frame(data_path)
    if is_header(data[0]):
       data.pop(0)
    floated = []
    for elm in data:
        new_dim = []
        for dim in elm:
            new_dim.append(float(dim))
        floated.append(new_dim)
    npdata = np.asarray(floated).astype(np.float)
    return npdata


def get_frame(local_path):
    with open(local_path) as f:
        lines = f.read().split('\n')
        csv = [x.split(',') for x in lines]
    csv.pop(-1)
    return csv