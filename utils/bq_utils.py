"""
Helper class to simplify common read-only BigQuery tasks.
"""
import logging
import sys
import time

import daiquiri
import pandas as pd
from google.cloud import bigquery

from utils import bq_client_helper as bq_helper, cloud_constants as cc

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


class BigQueryHelper(object):
    """
    Helper class to simplify common BigQuery tasks like executing queries,
    showing table schemas, etc without worrying about table or dataset pointers.

    See the BigQuery docs for details of the steps this class lets you skip:
    https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html
    """

    def __init__(self, active_project, dataset_name, max_wait_seconds=180):
        self.project_name = active_project
        self.dataset_name = dataset_name
        self.max_wait_seconds = max_wait_seconds
        self.client = bigquery.Client()
        self.__dataset_ref = self.client.dataset(self.dataset_name, project=self.project_name)
        self.dataset = None
        self.tables = dict()  # {table name (str): table object}
        self.__table_refs = dict()  # {table name (str): table reference}
        self.total_gb_used_net_cache = 0
        self.BYTES_PER_GB = 2 ** 30

    def __fetch_dataset(self):
        """
        Lazy loading of dataset. For example,
        if the user only calls `self.query_to_pandas` then the
        dataset never has to be fetched.
        """
        if self.dataset is None:
            self.dataset = self.client.get_dataset(self.__dataset_ref)

    def __fetch_table(self, table_name):
        """
        Lazy loading of table
        """
        self.__fetch_dataset()
        if table_name not in self.__table_refs:
            self.__table_refs[table_name] = self.dataset.table(table_name)
        if table_name not in self.tables:
            self.tables[table_name] = self.client.get_table(self.__table_refs[table_name])

    def __handle_record_field(self, row, schema_details, top_level_name=""):
        """
        Unpack a single row, including any nested fields.
        """
        name = row["name"]
        if top_level_name != "":
            name = top_level_name + "." + name
        schema_details.append(
            [
                {
                    "name": name,
                    "type": row["type"],
                    "mode": row["mode"],
                    "fields": pd.np.nan,
                    "description": row["description"],
                }
            ]
        )
        # float check is to dodge row['fields'] == np.nan
        if type(row.get("fields", 0.0)) == float:
            return None
        for entry in row["fields"]:
            self.__handle_record_field(entry, schema_details, name)

    def __unpack_all_schema_fields(self, schema):
        """
        Unrolls nested schemas. Returns dataframe with one row per field,
        and the field names in the format accepted by the API.
        Results will look similar to the website schema, such as:
            https://bigquery.cloud.google.com/table/bigquery-public-data:github_repos.commits?pli=1

        Args:
            schema: DataFrame derived from api repr of raw table.schema
        Returns:
            Dataframe of the unrolled schema.
        """
        schema_details = []
        schema.apply(lambda row: self.__handle_record_field(row, schema_details), axis=1)
        result = pd.concat([pd.DataFrame.from_dict(x) for x in schema_details])
        result.reset_index(drop=True, inplace=True)
        del result["fields"]
        return result

    def table_schema(self, table_name):
        """
        Get the schema for a specific table from a dataset.
        Unrolls nested field names into the format that can be copied
        directly into queries. For example, for the `github.commits` table,
        the this will return `committer.name`.

        This is a very different return signature than BigQuery's table.schema.
        """
        self.__fetch_table(table_name)
        raw_schema = self.tables[table_name].schema
        schema = pd.DataFrame.from_dict([x.to_api_repr() for x in raw_schema])
        # the api_repr only has the fields column for tables with nested data
        if "fields" in schema.columns:
            schema = self.__unpack_all_schema_fields(schema)
        # Set the column order
        schema = schema[["name", "type", "mode", "description"]]
        return schema

    def list_tables(self):
        """
        List the names of the tables in a dataset
        """
        self.__fetch_dataset()
        return [x.table_id for x in self.client.list_tables(self.dataset)]

    def estimate_query_size(self, query):
        """
        Estimate gigabytes scanned by query.
        Does not consider if there is a cached query table.
        See https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
        """
        my_job_config = bigquery.job.QueryJobConfig()
        my_job_config.dry_run = True
        my_job = self.client.query(query, job_config=my_job_config)
        return my_job.total_bytes_processed / self.BYTES_PER_GB

    def query_to_pandas(self, query):
        """
        Execute a SQL query & return a pandas dataframe
        """
        my_job = self.client.query(query)
        start_time = time.time()
        while not my_job.done():
            if (time.time() - start_time) > self.max_wait_seconds:
                print("Max wait time elapsed, query cancelled.")
                self.client.cancel_job(my_job.job_id)
                return None
            time.sleep(0.1)
        # Queries that hit errors will return an exception type.
        # Those exceptions don't get raised until we call my_job.to_dataframe()
        # In that case, my_job.total_bytes_billed can be called but is None
        if my_job.total_bytes_billed:
            self.total_gb_used_net_cache += my_job.total_bytes_billed / self.BYTES_PER_GB
        return my_job.to_dataframe()

    def query_to_pandas_safe(self, query, max_gb_scanned=1):
        """
        Execute a query, but only if the query would scan less than `max_gb_scanned` of data.
        """
        query_size = self.estimate_query_size(query)
        if query_size <= max_gb_scanned:
            return self.query_to_pandas(query)
        msg = "Query cancelled; estimated size of {0} exceeds limit of {1} GB"
        print(msg.format(query_size, max_gb_scanned))

    def head(self, table_name, num_rows=5, start_index=None, selected_columns=None):
        """
        Get the first n rows of a table as a DataFrame.
        Does not perform a full table scan; should use a trivial amount of data as long as n is small.
        """
        self.__fetch_table(table_name)
        active_table = self.tables[table_name]
        schema_subset = None
        if selected_columns:
            schema_subset = [col for col in active_table.schema if col.name in selected_columns]
        results = self.client.list_rows(
            active_table,
            selected_fields=schema_subset,
            max_results=num_rows,
            start_index=start_index,
        )
        results = [x for x in results]
        return pd.DataFrame(
            data=[list(x.values()) for x in results], columns=list(results[0].keys())
        )


def get_bq_data_for_inference(ecosystem, day_count, date_range) -> pd.DataFrame:
    """Query bigquery to retrieve data that is required for running the inference."""
    # ======= BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA -----")

    GH_BQ_CLIENT = bq_helper.create_github_bq_client()
    if ecosystem == "openshift":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.GOKUBE_REPO_LIST)
    elif ecosystem == "knative":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KNATIVE_REPO_LIST)
    elif ecosystem == "kubevirt":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KUBEVIRT_REPO_LIST)
    else:
        _logger.error(
            "Unsupported ecosystem, please re-run the inference with a valid ecosystem parameter."
        )
        # Returning 0 because this isn't a "HARD" error.
        sys.exit(0)

    # ======= BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA -----")

    # Don't change this
    YEAR_PREFIX = "20*"
    DAY_LIST = [item[2:] for item in date_range]
    QUERY_PARAMS = {
        "{year_prefix_wildcard}": YEAR_PREFIX,
        "{year_suffix_month_day}": "(" + ", ".join(["'" + d + "'" for d in DAY_LIST]) + ")",
        "{repo_names}": "(" + ", ".join(["'" + r + "'" for r in REPO_NAMES]) + ")",
    }

    _logger.info("\n")

    # ======= BQ GET DATASET SIZE ESTIMATE ========
    _logger.info("----- BQ Dataset Size Estimate -----")

    query = """
    SELECT  type as EventType, count(*) as Freq
            FROM `githubarchive.day.{year_prefix_wildcard}`
            WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
            AND repo.name in {repo_names}
            AND type in ('PullRequestEvent', 'IssuesEvent')
            GROUP BY type
    """
    query = bq_helper.bq_add_query_params(query, QUERY_PARAMS)
    df = GH_BQ_CLIENT.query_to_pandas(query)
    _logger.info("Dataset Size for Last N={n} days:-".format(n=day_count))
    _logger.info("\n{data}".format(data=df))

    _logger.info("\n")

    # ======= BQ GITHUB DATASET RETRIEVAL & PROCESSING ========
    _logger.info("----- BQ GITHUB DATASET RETRIEVAL & PROCESSING -----")

    ISSUE_QUERY = """
    SELECT
        repo.name as repo_name,
        type as event_type,
        'golang' as ecosystem,
        JSON_EXTRACT_SCALAR(payload, '$.action') as status,
        JSON_EXTRACT_SCALAR(payload, '$.issue.id') as id,
        JSON_EXTRACT_SCALAR(payload, '$.issue.number') as number,
        JSON_EXTRACT_SCALAR(payload, '$.issue.url') as api_url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') as creator_name,
        JSON_EXTRACT_SCALAR(payload, '$.issue.user.html_url') as creator_url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.created_at') as created_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.updated_at') as updated_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.closed_at') as closed_at,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.issue.title'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as title,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.issue.body'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as body

    FROM `githubarchive.day.{year_prefix_wildcard}`
        WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
        AND repo.name in {repo_names}
        AND type = 'IssuesEvent'
        """

    ISSUE_QUERY = bq_helper.bq_add_query_params(ISSUE_QUERY, QUERY_PARAMS)
    qsize = GH_BQ_CLIENT.estimate_query_size(ISSUE_QUERY)
    _logger.info("Retrieving GH Issues. Query cost in GB={qc}".format(qc=qsize))

    issues_df = GH_BQ_CLIENT.query_to_pandas(ISSUE_QUERY)
    if issues_df.empty:
        _logger.warn("No issues present for given time duration.")
    else:
        _logger.info("Total issues retrieved: {n}".format(n=len(issues_df)))

        issues_df.created_at = pd.to_datetime(issues_df.created_at)
        issues_df.updated_at = pd.to_datetime(issues_df.updated_at)
        issues_df.closed_at = pd.to_datetime(issues_df.closed_at)
        issues_df = issues_df.loc[
            issues_df.groupby("url").updated_at.idxmax(skipna=False)
        ].reset_index(drop=True)
        _logger.info("Total issues after deduplication: {n}".format(n=len(issues_df)))

    PR_QUERY = """
    SELECT
        repo.name as repo_name,
        type as event_type,
        'golang' as ecosystem,
        JSON_EXTRACT_SCALAR(payload, '$.action') as status,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as id,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') as number,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.url') as api_url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as creator_name,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.html_url') as creator_url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') as created_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.updated_at') as updated_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.pull_request.title'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as title,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.pull_request.body'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as body

    FROM `githubarchive.day.{year_prefix_wildcard}`
        WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
        AND repo.name in {repo_names}
        AND type = 'PullRequestEvent'
    """

    PR_QUERY = bq_helper.bq_add_query_params(PR_QUERY, QUERY_PARAMS)
    qsize = GH_BQ_CLIENT.estimate_query_size(PR_QUERY)
    _logger.info("Retrieving GH Pull Requests. Query cost in GB={qc}".format(qc=qsize))

    prs_df = GH_BQ_CLIENT.query_to_pandas(PR_QUERY)
    if prs_df.empty:
        _logger.warn("No pull requests present for given time duration.")
    else:
        _logger.info("Total pull requests retrieved: {n}".format(n=len(prs_df)))

        prs_df.created_at = pd.to_datetime(prs_df.created_at)
        prs_df.updated_at = pd.to_datetime(prs_df.updated_at)
        prs_df.closed_at = pd.to_datetime(prs_df.closed_at)
        prs_df = prs_df.loc[prs_df.groupby("url").updated_at.idxmax(skipna=False)].reset_index(
            drop=True
        )
        _logger.info("Total pull requests after deduplication: {n}".format(n=len(prs_df)))

    _logger.info("\n")

    _logger.info("Merging issues and pull requests datasets")
    cols = issues_df.columns
    df = pd.concat([issues_df, prs_df], axis=0, sort=False, ignore_index=True).reset_index(
        drop=True
    )
    df = df[cols]

    if df.empty:
        _logger.warn("Nothing to predict today :)")
        sys.exit(0)

    _logger.info("Creating description column for NLP")
    df["description"] = df["title"].fillna(value="").map(str) + " " + df["body"].fillna(value="")
    columns = ["title", "body"]
    df.drop(columns, inplace=True, axis=1)
    return df
