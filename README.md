Open Anomaly Detection is an open source **multivariate**, **portable** and **customizable** Prediction based Anomaly Detection algorithm, powered by our [OpenForecast][forecast] model. You can see it in action [here][algolink].
<img src="https://i.imgur.com/wcFCKL5.png"></img>

## Introduction

That's a lot of stuff, lets break down each of those terms:
* Multivariate - Being able to predict anomalies in univariate problems (situations with only 1 variable) are not that common. More often we
need the capability to support multiple independent variables, this algorithm can do that.
* Portable - This algorithm is designed to run best on the [Algorithmia][algo] platform, however it's capable of running on any linux based system.
* Open Source - As you can see, this algorithm is fully open source and available for public consumption! If you have any ideas on how to make it even better, please feel free to file a PR.
* Customizable - This model is designed to be trained and specialized to your data. All variables are exposed and available for tinkering.
* Prediction based - This algorithm uses a [Prediction based][pred] replication failure metric for measuring anomalies, this means that we go deeper than just looking at periods or peaks - this algorithm learns what normal data looks like, and what constitutes an anomaly within your data.

To do all of this, we heavily the [Pytorch][pytorch] machine learning framework, along with the [OpenForecast][forecast] Algorithm.


## Getting Started Guide




### Anomaly Detection

##### Example IO
Input: 
```json
```

Output:

```json
```

## IO Schema

<a id="commonTable"></a>

#### Input

| Parameter | Type | Description | Default if applicable |
| --------- | ----------- | ----------- | ----------- |
| data_path | String | The data path | N/A |
| model_input_path | String | The data API path to the trained model you've previously built. |N/A|
| graph_save_path | String | The output path for a visual graph describing the found Anomalies. | N/A |
| sigma_threshold | Float | The minimum sigma deviation from the mean to consider an event anomalous. | 2.0 |
| variable_index | Int | The specific dimension to detect anomalies on for a given dataset. | 1 |
| calibration_percentage | Float | The percentage of the start of the dataset used to calibrate the model. This data won't be scored for anomalies. | 0.1 |

#### Output

| Parameter | Type | Description |
| --------- | ----------- | ----------- |
| graph_save_path | String | If you set a graph_save_path, then we successfully saved a graph at this data API . |
| anomalous_regions | List[Anomaly] | A json List object containing all detected anomalies. |



#### Anomaly

| Parameter | Type | Description |
| --------- | ----------- | ----------- |
| avg_sigma | Float | The anomaly's average sigma deviation from the norm. |
| max_sigma | Float | The anomaly's maximum measured sigma deviation from the norm. |
| upper | Int | The anomaly's upper limit (along the x axis), as an index value |
| lower | Int | The anomaly's lower limit (along the x axis), as an index value |


#### Example
**Input**
 ```json
{  
   "data_path":"'data://timeseries/example_collection/m4-hourly-data.json",
   "model_input_path":"data://timeseries/example_collection/m4_daily_0.1.0.zip",
   "graph_save_path":"data://.algo/temp/graph_file1.png",
   "sigma_threshold":3,
   "variable_index":3,
   "calibration_percentage":0.1
}
```

**Output**
```json
{  
   "graph_save_path": "data://.algo/temp/graph_file1.png",
   "anomalous_regions":[  
      {  
         "avg_sigma":3.1524142798011714,
         "upper":247,
         "lower":210,
         "max_sigma":3.251589215327179
      },
      {  
         "avg_sigma":4.508339281832453,
         "upper":345,
         "lower":298,
         "max_sigma":5.23575088149042
      },
      {  
         "avg_sigma":3.7968170661690834,
         "upper":407,
         "lower":369,
         "max_sigma":4.309617776250398
      },
      {  
         "avg_sigma":3.6096409752636327,
         "upper":498,
         "lower":461,
         "max_sigma":4.221452424551777
      }
   ]
}
```
[algo]: https://www.algorithmia.com
[pred]: https://www.dynatrace.com/support/help/monitor/problems/problem-detection/prediction-based-anomaly-detection/
[forecast]: 
[algolink]: https://algorithmia.com/algorithms/TimeSeries/OpenAnomalyDetection
[pytorch]: https://algorithmia.com/algorithms/TimeSeries/OpenForecast
[gitreadme]: GITREADME.d