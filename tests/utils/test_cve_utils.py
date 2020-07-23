"""Test class for the CVE utils."""

import pandas as pd

from utils.cve_utils import get_cves_from_text, update_cve_info


def test_update_cve_info():
    """Test dataframe cve update logic."""
    df = pd.read_csv("tests/test_data/test_triage_results.csv")
    df = update_cve_info(df)

    # assert cves detail
    df_without_cve = df[df.url == 'https://github.com/Azure/azure-sdk-for-go/issues/4408']
    assert len(df_without_cve['cves'].iloc[0]) == 0

    df_with_cve = df[df.url == 'https://github.com/Azure/azure-sdk-for-go/issues/5222']
    cves = df_with_cve['cves'].iloc[0]
    assert "CVE-2019-3546" in cves
    assert "CVE-2020-3546" in cves


def test_get_cves_from_text():
    """Test the retriving CVE logic."""
    text = """
            -----Valid CVEs-----
            CVE-2014-0001
            CVE-2014-0999
            CVE-2014-10000
            CVE-2014-100000
            CVE-2014-1000000
            CVE-2014-100000000
            CVE-2019-111111111
            CVE-2019-456132
            CVE-2019-54321
            CVE-2020-65537
            CVE-2020-7654321
            cve-1234-1234 - This is a valid CVE as we are converting text to uppercase for retriving CVEs
            -----Invalid CVEs Text-----
            CVE-0a34-9823
            CVE-2019-998a
            CVE-2020
            CVE-123-1234
            """
    cves = get_cves_from_text(text)
    assert len(cves) == 12


def test_get_cves_with_duplicate_data():
    """Test the duplicate logic."""
    text = "CVE-2019-0001 and CVE-2019-0001"
    cves = get_cves_from_text(text)
    assert len(cves) == 1
    assert 'CVE-2019-0001' in cves


def test_get_cves_with_extra_text():
    """Test logic with extra text as prefix and suffix."""
    text = """ Test CVE aCVE-2019-2341andCVE-2019-3546b CVE-2019- VE-2019-1234"""
    cves = get_cves_from_text(text)
    assert len(cves) == 2
    assert 'CVE-2019-2341' in cves
    assert 'CVE-2019-3546' in cves
