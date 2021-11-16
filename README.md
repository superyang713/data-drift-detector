## About The Package
The package is a wrapper of tensorflow data validation for our specific needs.
It can analyze training data and serving data to compute desscriptive
statistics, infer a schema, and detect anomalies.

## Dependencies

* [tensorflow-data-validation](https://www.tensorflow.org/tfx/data_validation/get_started)


## Installation
```sh
pip install data-drift-detector
```
<!-- USAGE EXAMPLES -->
## Usage

Initialize a Harvest client:
```python
# The Dataset, TrainDataset, ServeDataset can be initialized with different methods.

train = TrainDataset.from_GCS()
train = TrainDataset.from_bigquery()
train = TrainDataset.from_dataframe()
train = TrainDataset.from_stats_file()
```

Populate the class variables and submit.
```python
# Get training dataset schema
schema = train.schema_dict()
```
