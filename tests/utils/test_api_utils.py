"""Test class that covers unit test for api utils."""
from unittest import TestCase
from unittest.mock import patch, MagicMock

import arrow
import pandas as pd

from utils.api_util import save_data_to_db, failed_to_insert


class MockResponse:
    """Mock response for request output."""

    def __init__(self, json_data, status_code):
        """Init method for MockResponse."""
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        """Json method to return json which was set while creating Mock object."""
        return self.json_data


def _get_mocked_error_response():
    return MockResponse({"message": "Internal Server Error"}, 500)


def _get_mocked_success_response():
    return MockResponse({"status": "success"}, 200)


class APIUtilTestCase(TestCase):
    """Test the API util functions."""

    @patch("pandas.read_csv", return_value=pd.read_csv('tests/test_data/sample_probable_cve_data.csv'))
    @patch("requests.post", return_value=_get_mocked_success_response())
    @patch("pandas.DataFrame.to_csv", return_value=MagicMock())
    def test_save_data_to_db(self, save_data_to_csv_call, mock_data, mocked_success_response):
        """Test save_data_to_db method without any  error."""
        start = arrow.now()
        end = arrow.now().shift(days=-7)

        save_data_to_db(start, end, "bert", True, "openshift")

        # assert failed record count
        assert 0 == len(failed_to_insert)

        # check method should not be called as there is no failed record to save.
        save_data_to_csv_call.assert_not_called()

    @patch("pandas.read_csv", return_value=pd.read_csv('tests/test_data/sample_probable_cve_data.csv'))
    @patch("requests.post", return_value=_get_mocked_error_response())
    @patch("pandas.DataFrame.to_csv")
    def test_save_data_to_db_with_error(self, save_data_to_csv_call, mock_data, mocked_error_response):
        """Test save_data_to_db method with error."""
        start = arrow.now()
        end = arrow.now().shift(days=-7)

        save_data_to_db(start, end, "bert", True, "openshift")

        # assert failed data count
        assert 5 == len(failed_to_insert)

        # Check to_csv method should be called to save failed record as csv file
        save_data_to_csv_call.assert_called()
