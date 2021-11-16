"""
Description:

    Validator class contains the main logics for data drift detection and
    schema validation

"""
from typing import Optional

import pandas as pd

import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
from google.protobuf.json_format import MessageToDict

from tensorflow_metadata.proto.v0 import statistics_pb2

from .bigquery import BigQuery


class Dataset:
    """A base Dataset class that contains statistical information of a dataset.

    :param stats: statistics_pb2.DatasetFeatureStatisticsList, a proto
    that contains dataset statistical information.

    Initialize the validator from different source connectors:
      >>> import Dataset
      >>> val = Dataset.from_GCS()
      >>> val = Dataset.from_bigquery()
      >>> val = Dataset.from_dataframe()
      >>> val = Dataset.from_stats_file()
    """
    def __init__(self, stats: statistics_pb2.DatasetFeatureStatisticsList):
        self.stats = stats

    @property
    def stats_dict(self):
        return MessageToDict(self.stats)

    @classmethod
    def from_gcs(cls, url: str):
        """A factory method to create an instance of Dataset.

        :param url -> str: gsutil URI for the dataset in the form of
        gs://{bucket}/{blobname}

        """
        stats = tfdv.generate_statistics_from_csv(data_location=url)
        return cls(stats)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame):
        """A factory method to create an instance of Dataset.

        :param dataframe -> pd.dataframe: A pandas dataframe that contains the
        dataset.

        """
        stats = tfdv.generate_statistics_from_dataframe(
            dataframe=dataframe)
        return cls(stats)

    @classmethod
    def from_bigquery(cls, query: str, service_path: str = None):
        """A factory method to create an instance of Dataset.

        :param query -> str: A bigquery query used for pulling the dataset.

        """
        bigquery = BigQuery(service_path)
        dataframe = bigquery.run(query=query, threshold=10)
        return cls.from_dataframe(dataframe)

    @classmethod
    def from_stats_file(cls, input_path: str):
        """A factory method to create an instance of Dataset.

        :param query -> str: A bigquery query used for pulling the dataset.

        """
        stats = tfdv.load_stats_text(input_path)
        return cls(stats)

    def stats_html(self) -> str:
        """Generate a html containing the statistics information for outside of
            jupyter notebook use.

        :return -> str: a html string

        """
        html = get_statistics_html(lhs_statistics=self.stats)
        return html

    def show_stats(self) -> None:
        """This method is used to display stats inside a jupyter notebook.

        """
        tfdv.visualize_statistics(lhs_statistics=self.stats)

    def save_stats(self, output_path: str):
        """Writes statistics of the dataset to a file in text format.

        """
        tfdv.write_stats_text(self.stats, output_path)


class ServeDataset(Dataset):
    """This class is exactly the same as Dataset class.

    """
    pass


class TrainDataset(Dataset):
    """The biggest difference of TrainDataset and ServeDataset is that
    TrainDataset needs to deal with schema. So the additional methods are all
    created for this purpose.

    :param stats: statistics_pb2.DatasetFeatureStatisticsList, a proto that
    contains dataset statistical information.

    :param schema_input_path: if specified, the schema will be loaded from a
    saved schema text file, otherwise, the schema will be inferred using
    training dataset.

    """
    def __init__(
            self,
            stats: statistics_pb2.DatasetFeatureStatisticsList,
            schema_input_path
    ):
        super().__init__(stats)
        if schema_input_path is None:
            self.schema = tfdv.infer_schema(statistics=stats)
        else:
            self.schema = tfdv.load_schema_text(schema_input_path)

    @property
    def schema_dict(self):
        return MessageToDict(self.schema)

    @classmethod
    def from_gcs(
            cls,
            url: str,
            schema_input_path: Optional[str] = None,
    ):
        """A factory method to create an instance of Dataset.

        :param url -> str: gsutil URI for the dataset in the form of
        gs://{bucket}/{blobname}

        """
        stats = tfdv.generate_statistics_from_csv(data_location=url)
        return cls(stats, schema_input_path)

    @classmethod
    def from_dataframe(
            cls,
            dataframe: pd.DataFrame,
            schema_input_path: Optional[str] = None,
    ):
        """A factory method to create an instance of Dataset.

        :param dataframe -> pd.dataframe: A pandas dataframe that contains the
        dataset.

        """
        stats = tfdv.generate_statistics_from_dataframe(
            dataframe=dataframe)
        return cls(stats, schema_input_path)

    @classmethod
    def from_bigquery(
            cls,
            query: str,
            service_path: str = None,
            schema_input_path: Optional[str] = None,
    ):
        """A factory method to create an instance of Dataset.

        :param query -> str: A bigquery query used for pulling the dataset.

        """
        bigquery = BigQuery(service_path)
        dataframe = bigquery.run(query=query, threshold=10)
        return cls.from_dataframe(dataframe, schema_input_path)

    @classmethod
    def from_stats_file(
            cls,
            input_path: str,
            schema_input_path: Optional[str] = None,
    ):
        """A factory method to create an instance of Dataset.

        :param query -> str: A bigquery query used for pulling the dataset.

        """
        stats = tfdv.load_stats_text(input_path)
        return cls(stats, schema_input_path)

    def show_schema(self) -> None:
        """This method is used to display schema inside a jupyter notebook.

        """
        tfdv.display_schema(schema=self.schema)

    def save_schema(self, output_path: str):
        """Writes input schema to a file in text format.

        """
        tfdv.write_schema_text(self.schema, output_path)

    def add_schema_constraint(
            self,
            feature: str,
            kind: str,
            value) -> None:
        """This method should be called before "self.detect_anormaly()" method.
        It modifies or adds a constraint to the existing schema.

        :param feature -> str: the column (feature) name in the dataset. The
            constraint will be associated with the specified column.

        :param kind -> str: the type of the constraint. Allowed types of
            constraint are max, min, mean, etc.

        :param constraint -> float: the threshold, which, when exceeded, will
            be detected when detect_anormaly method is called.
        """
        feature = tfdv.get_feature(self.schema, feature)
        action_selector = {
            "max":                         self._change_max,
            "min":                         self._change_min,
            "datatype":                    self._change_datatype,
            "disallow_nan":                self._disallow_nan,
            "is_categorical":              self._change_to_categorical,
            "min_domain_mass":             self._change_min_domain_mass,
            "presence.min_count":          self._change_presence_min_count,
            "presence.min_fraction":       self._change_presence_min_fraction,
            "string_domain.value":         self._add_value_to_string_domain,
            "categorical_drift_threshold": self._change_categorical_drift_threshold, # noqa
            "numerical_drift_threshold":   self._change_numerical_drift_threshold, # noqa
        }
        action = action_selector.get(kind)
        if not action:
            raise ValueError(f"The constraint type {kind} is not supported")
        action(feature, value)

    @staticmethod
    def _change_datatype(feature, datatype: str):
        """Change the datatype of the feature.
        """
        mapping = {
            "TYPE_UNKNOWN": 0,
            "BYTES": 1,
            "INT": 2,
            "FLOAT": 3,
            "STRUCT": 4
        }
        value = mapping.get(datatype)
        if value is None:
            message = (
                f"The datatype {datatype} is not allowed. The accepted values "
                "are TYPE_UNKNOWN BYTES INT FLOAT STRUCT"
            )
            raise ValueError(message)
        feature.type = value

    @staticmethod
    def _change_presence_min_fraction(feature, value: float):
        """Change minimum fraction of examples that have this feature
        """
        feature.presence.min_fraction = value

    @staticmethod
    def _change_presence_min_count(feature, value: int):
        """Change minimum count of examples that have this feature
        """
        feature.presence.min_count = value

    @staticmethod
    def _change_min_domain_mass(feature, value: float):
        """The minimum fraction (in [0,1]) of values across all examples that
        should come from the feature's domain, e.g.: 1.0 => All values must
        come from the domain. .9 => At least 90% of the values must come from
        the domain. Only supported for StringDomains.

        """
        feature.distribution_constraints.min_domain_mass = value

    @staticmethod
    def _add_value_to_string_domain(feature, value: str):
        """Add the value appearing in the domain of a feature.
        """
        feature.string_domain.value.append(value)

    @staticmethod
    def _change_max(feature, value: str):
        """Change the max value of the domain
        """
        feature_info = MessageToDict(feature)
        datatype = feature_info["type"]
        if datatype == "INT":
            feature.int_domain.max = int(value)
        elif datatype == "FLOAT":
            value = float(value)
            feature.float_domain.max = float(value)
        else:
            raise TypeError(f"Datatype {feature} does not have maximum")

    @staticmethod
    def _change_min(feature, value: str):
        """Change the max value of the domain
        """
        feature_info = MessageToDict(feature)
        datatype = feature_info["type"]
        if datatype == "INT":
            feature.int_domain.min = int(value)
        elif datatype == "FLOAT":
            feature.float_domain.min = float(value)
        else:
            raise TypeError(f"Datatype {feature} does not have minimum")

    @staticmethod
    def _disallow_nan(feature, value: bool):
        """If true, feature should not contain NaNs
        """
        feature.float_domain.disallow_nan = value

    @staticmethod
    def _change_to_categorical(feature, value: bool):
        """If true then the domain encodes categorical values (i.e., ids)
        rather than ordinal values.
        """
        feature_info = MessageToDict(feature)
        datatype = feature_info["type"]
        if datatype == "INT":
            feature.int_domain.is_categorical = value
        elif datatype == "FLOAT":
            feature.float_domain.is_categorical = value
        else:
            raise TypeError("Numerical feature is needed")

    @staticmethod
    def _change_categorical_drift_threshold(feature: str, threshold: float):
        """This method changes the threshold used for drift detection.

        There are two types of metrics used for drift detection, Jensen Shannon
        divergence and infinity norm, which are used for numerical and
        categorical features, respectively.

        :param feature -> str: the column (feature) name in the dataset. The
            threshold will be associated with the specified column.

        :param threshold -> float: the threshold, which, when exceeded, will be
            detected when detect_anormaly method is called. The threshold is in
            the interval [0.0, 1.0]

        """
        threshold = float(threshold)
        feature.skew_comparator.infinity_norm.threshold = threshold

    @staticmethod
    def _change_numerical_drift_threshold(feature: str, threshold: float):
        """This method changes the threshold used for drift detection.

        There are two types of metrics used for drift detection, Jensen Shannon
        divergence and infinity norm, which are used for numerical and
        categorical features, respectively.

        :param feature -> str: the column (feature) name in the dataset. The
            threshold will be associated with the specified column.

        :param threshold -> float: the threshold, which, when exceeded, will be
            detected when detect_anormaly method is called. The threshold is in
            the interval [0.0, 1.0]

        """
        threshold = float(threshold)
        feature.skew_comparator.jensen_shannon_divergence.threshold = threshold


class Validator:
    """A validator is used for schema validation, and drift detection on
    serving dataset.

    :param train -> TrainDataset: it contains the information of training
    dataset statistical information and schema.

    :param serve -> ServeDataset: it contains the information of serving
    dataset statistical information.

    Usage:
      >>> import Validator
      >>> val = Validator(train, serve)
      >>> val.detect_drift()
      >>> val.validate_schema()

    """
    def __init__(self, train: TrainDataset, serve: ServeDataset):
        self.train = train
        self.serve = serve

    def detect_drift(self, visual=False) -> dict:
        """detects data drift anomalies.

        :param visual -> bool: whether to display anomalies result inside a
            jupyter notebook.

        :return: A dictionary containing the information of the anamalies.
        """
        anomaly = tfdv.validate_statistics(
            statistics=self.train.stats,
            schema=self.train.schema,
            serving_statistics=self.serve.stats,
        )
        if visual is True:
            tfdv.display_anomalies(anomaly)
        return MessageToDict(anomaly)

    def validate_schema(self, visual=False) -> dict:
        """detects schema violation or statistical anomalies on serving data.

        :param visual -> bool: whether to display stats inside a jupyter
            notebook.

        :return: A dictionary containing the information of the anamalies.
        """
        anomaly = tfdv.validate_statistics(self.serve.stats, self.train.schema)
        if visual is True:
            tfdv.display_anomalies(anomaly)
        return MessageToDict(anomaly)

    def stats_html(self) -> str:
        """
        :return: a html containing both training and serving statistics
        information for outside of jupyter notebook use.

        """
        html = get_statistics_html(
            lhs_statistics=self.serve.stats,
            rhs_statistics=self.train.stats,
            lhs_name="SERVE",
            rhs_name='TRAIN'
        )
        return html

    def show_stats(self) -> None:
        """This method is used to display both serving and training stats
        inside a jupyter notebook.

        """
        tfdv.visualize_statistics(
            lhs_statistics=self.serve.stats,
            rhs_statistics=self.train.stats,
            lhs_name="SERVE",
            rhs_name='TRAIN'
        )
