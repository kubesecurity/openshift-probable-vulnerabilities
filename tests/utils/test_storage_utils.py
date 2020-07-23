"""Test the write utilities for both localfilesystem and S3."""
import os
from unittest import TestCase
from unittest import mock

import arrow
import pandas as pd

from utils import other_constants as oc
from utils.storage_utils import write_output_csv


def to_mock_csv(*args, **kwargs):
    """Mock the .to_csv() call of a dataframe."""
    return args, kwargs


def do_nothing(*args, **kwargs):
    """No action."""
    pass


@mock.patch("utils.other_constants.BASE_TRIAGE_DIR", 'test')
class TestStorageUtils(TestCase):
    """Test the storage utils functions."""

    @mock.patch("pandas.DataFrame.to_csv")
    @mock.patch("os.makedirs")
    def test_write_output_csv(self, do_nothing, to_mock_csv):
        """Test to check output csv."""
        df = pd.read_csv("tests/test_data/test_triage_results_with_cves.csv")

        # In the code we are adding fillna & map function before passing to write_output_csv function
        # so doing the same with the df created above
        df["body"] = df["body"].fillna(value="")
        df["title"] = df["title"].fillna(value="")

        # Check if mocks are setup properly.
        assert pd.DataFrame.to_csv is to_mock_csv
        assert os.makedirs is do_nothing

        start = arrow.now()
        end = arrow.now().shift(days=-7)

        df = write_output_csv(start, end, "bert_torch", "knative", df, False)
        # Check if CSV local creation happens normally.
        to_mock_csv.assert_called_with(
            "test/{}-{}/bert_model_inference_probable_cves_{}-{}_knative.csv".format(
                start.format("YYYYMMDD"),
                end.format("YYYYMMDD"),
                start.format("YYYYMMDD"),
                end.format("YYYYMMDD"),
            ),
            index=False,
        )

        # Check if the list of columns of CSV is properly filtered.
        self.assertSetEqual(
            set(df.columns.to_list()),
            {
                "repo_name",
                "event_type",
                "status",
                "url",
                "security_model_flag",
                "cve_model_flag",
                "triage_feedback_comments",
                "id",
                "number",
                "api_url",
                "created_at",
                "updated_at",
                "closed_at",
                "creator_name",
                "creator_url",
                "title",
                "body",
                "ecosystem",
                "cves"
            },
        )

        # Check if ecosystem is properly set.
        self.assertListEqual(df["ecosystem"].unique().tolist(), ["knative"])

        # Check body text character length
        # As string is around 2400 characters based on constant it should trim to 2000
        df_with_trimed_body = df[df.url == 'https://github.com/Azure/azure-sdk-for-go/issues/4408']
        assert oc.MAX_STRING_LEN_FOR_CSV_EXPORT == len(df_with_trimed_body['body'][0])

        # Test body value with Empty string
        df_with_blank_body = df[df.url == 'https://github.com/Azure/azure-sdk-for-go/issues/5222']
        assert df_with_blank_body['body'].iloc[0] is None
