"""Test class that covers unit test for api utils."""
from unittest import TestCase
from unittest.mock import patch

import arrow
import pandas as pd
from aiohttp import ClientSession

from utils.api_util import save_data_to_db, report_failures, read_probable_cve_data


class MockResponse:
    """Mock response for request output."""

    def __init__(self, json_data, status):
        """Init method for MockResponse."""
        self.json_data = json_data
        self.status = status

    def json(self):
        """Json method to return json which was set while creating Mock object."""
        return self.json_data

    async def __aenter__(self, *args):  # noqa
        """Aenter method require for asyncio call."""
        return self

    async def __aexit__(self, *args):  # noqa
        """Aexit method require for asyncio call."""
        return self


def _get_mocked_error_response():
    return MockResponse({"message": "Internal Server Error"}, 500)


def _get_mocked_success_response():
    return MockResponse({"status": "success"}, 200)


class APIUtilTestCase(TestCase):
    """Test the API util functions."""

    def setUp(self):
        """Do required Setup for unit tests."""
        self.start = arrow.now()
        self.end = arrow.now().shift(days=-7)

    @patch("aiohttp.ClientSession.post", return_value=_get_mocked_success_response())
    def test_save_data_to_db(self, mocked_success_response):
        """Test save_data_to_db method without any  error."""
        # assert post call
        assert ClientSession.post is mocked_success_response

        probable_cve_data = pd.read_csv('tests/test_data/sample_probable_cve_data.csv')
        updated_df, failed_to_insert = save_data_to_db(probable_cve_data, "openshift")

        # assert failed record count
        assert 0 == len(failed_to_insert)
        assert 5 == len(updated_df)

    @patch("aiohttp.ClientSession.post", return_value=_get_mocked_error_response())
    def test_save_data_to_db_with_error(self, mocked_error_response):
        """Test save_data_to_db method with error."""
        # assert post call
        assert ClientSession.post is mocked_error_response

        probable_cve_data = pd.read_csv('tests/test_data/sample_probable_cve_data.csv')
        updated_df, failed_to_insert = save_data_to_db(probable_cve_data, "openshift")

        # assert failed data count
        assert 5 == len(failed_to_insert)
        assert 5 == len(updated_df)

    @patch("pandas.DataFrame.to_csv")
    def test_report_failures_no_data(self, save_data_to_csv_call):
        """Test save_data_to_db method without any  error."""
        df = pd.read_csv('tests/test_data/sample_probable_cve_data.csv')
        failed_records = []

        report_failures(df, failed_records, self.start, self.end, True, "openshift")

        # Check to_csv method should not be called as we dont have any failed records
        save_data_to_csv_call.assert_not_called()

    @patch("s3fs.S3FileSystem.open")
    def test_report_failures_with_data(self, save_data_to_csv_call):
        """Test save_data_to_db method without any  error."""
        df = pd.read_csv('tests/test_data/sample_probable_cve_data.csv')
        failed_records = ['https://api.github.com/repos/google/cadvisor/issues/2584']

        report_failures(df, failed_records, self.start, self.end, True, "openshift")

        # Check to_csv method should be called to save failed record as csv file
        save_data_to_csv_call.assert_called()

    @patch("s3fs.S3FileSystem.open", return_value=open('tests/test_data/sample_probable_cve_data.csv'))
    def test_read_probable_cve_data(self, mock_data):
        """Test read_probable_cve_data method."""

        df = read_probable_cve_data(self.start, self.end, "bert", True, "openshift")

        # assert data count
        assert 5 == len(df)
