"""
Description:
    Userful functions related to GCP BigQuery
"""
import json
import os
from io import BytesIO
import hashlib

import pandas as pd
from google.cloud import bigquery, storage, exceptions

from .common import setup_log, get_credentials


logger = setup_log(__name__)


class BigQuery:
    def __init__(self, service_path=None):
        """Client requires bigquery credential scope
        """
        if service_path:
            credentials = get_credentials(
                service_path=service_path,
                scopes=[
                    "https://www.googleapis.com/auth/bigquery",
                    "https://www.googleapis.com/auth/devstorage.full_control"
                ]
            )
            self.bq_client = bigquery.Client(credentials=credentials)
            self.storage_client = storage.Client(credentials=credentials)
        else:
            self.bq_client = bigquery.Client()
            self.storage_client = storage.Client()

        self.project_id = self._get_project_id(service_path)
        self.bucket_name = f"{self.project_id}-temp-ga-life-cycle-one-day"
        self.hash_name = None
        self.csv_path = None

    def run(self, query, threshold=5) -> pd.DataFrame:
        """Run a query in bigquery and return a pandas dataframe. Result from
        the bigquery will always be cached in GCS tmp/google-analytics bucket.
        The life cycle for this folder is set to 1 day.
        The cache file name is the hash of the query.
        Parameters
        ----------
        query : str
            A query that can be executed by GCP BigQuery.
        threshold: float
            Default is 5 (GB). The estimated cost should be lower than this
            value in order to execute the query. Otherwise, ValueError will be
            raised. If cache is found, this threshold will be ignored.
        Returns
        -------
        query_result : DataFrame
            Result from BigQuery loaded in Pandas DataFrame
        """
        # Check if the caching bucket exist. If not, create one
        self._create_cache_bucket_if_not_exist()
        self.hash_name = f"{self.convert_to_hash(query)}.csv"
        self.csv_path = f"gs://{self.bucket_name}/{self.hash_name}"

        try:
            logger.debug("Looking for cache in GCS")
            byte_stream = self._get_byte_fileobj(self.hash_name)
            df = pd.read_csv(byte_stream)
        except exceptions.NotFound:
            logger.debug("Cache not found. Running query...")
            df = self._run(query, threshold)

            logger.debug("Exporting result to GCS tmp folder.")
            self._cache_to_gcs(df, self.hash_name)

        return df

    def _estimate_query_cost(self, query: str):
        """
        Returns
        -------
        cost_in_GB: float
            An estimate of the cost of the query
        """
        job_config = bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False
        )

        query_job = self.bq_client.query(query, job_config=job_config)
        cost_in_GB = query_job.total_bytes_processed / 1024 ** 3

        return cost_in_GB

    def _run(self, query: str, threshold: float):
        estimated_cost = self._estimate_query_cost(query)
        logger.info(f"Estimated cost is {estimated_cost:.4f} G")

        if estimated_cost > threshold:
            logger.debug("Query failed to execute.")
            raise ValueError
        query_result = self.bq_client.query(query).result().to_dataframe()
        return query_result

    def _create_cache_bucket_if_not_exist(self):
        bucket = self.storage_client.bucket(self.bucket_name)
        if bucket.exists():
            return
        else:
            bucket = self.storage_client.create_bucket(self.bucket_name)
            bucket.add_lifecycle_delete_rule(age=1)
            bucket.patch()

    def _get_byte_fileobj(self, blob_name) -> BytesIO:
        blob = self.storage_client.bucket(self.bucket_name).blob(blob_name)
        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        return byte_stream

    def _cache_to_gcs(self, df: pd.DataFrame, blob_name: str):
        # convert dataframe to csv file
        # filepath = Path(gettempdir(), blob_name)
        # df.to_csv(filepath, index=False)

        # upload csv to gcs
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(df.to_csv(index=False), "text/csv")

    @staticmethod
    def _get_project_id(service_path):
        service_path = (
            service_path
            if service_path
            else os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if service_path:
            with open(service_path) as fin:
                project_id = json.load(fin).get("project_id")
                return project_id
        else:
            project_id = "mightyhive-data-science-poc"
            return project_id

    @staticmethod
    def convert_to_hash(value: str):
        byte = hashlib.md5(value.encode())
        result = byte.hexdigest()
        return result
