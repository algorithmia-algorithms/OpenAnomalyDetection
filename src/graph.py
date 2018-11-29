import numpy as np
import matplotlib
from uuid import uuid4
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def graph_anomalies(anomalies, dataframe):
    local_filename = '/tmp/{}.png'.format(str(uuid4()))
    full_range = np.arange(dataframe.shape[0])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range(dataframe.shape[1]):
                dim_color = np.random.rand(3,)
                plt.plot(full_range, dataframe[:, i],
                         c=dim_color,
                         label='data for dimension {}'.format(str(i)))
    for j in range(dataframe.shape[1]):
        for anomaly in anomalies:
            anomaly_color = np.random.rand(3, )
            if anomaly['upper'] > dataframe.shape[0]:
                anom_max = dataframe.shape[0]
            else:
                anom_max = anomaly['upper']
            if anomaly['lower'] <= 0:
                anom_min = 0
            else:
                anom_min = anomaly['lower']
            anom_range = np.arange(anom_min, anom_max)
            anom_data = dataframe[anom_min:anom_max, j]
            plt.plot(anom_range, anom_data,
                     c=anomaly_color,
                     label='anomaly detected, sigma:{}'.format(str(anomaly['sigma'])))
        plt.title("we detected {} anomalies.".format(str(len(anomalies))))
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend(loc=2, fontsize='xx-small')
    plt.savefig(local_filename, dpi=300)
    plt.close()
    return local_filename