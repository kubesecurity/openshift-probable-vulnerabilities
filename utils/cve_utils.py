"""Util class to handle cve realted operation."""

import re
import pandas as pd

CVE_REGULAR_EXPRESSION = r"CVE-\d{4}-\d{4,}"


def get_cves_from_text(data: str) -> set:
    """Get the CVEs from the given text."""
    cves = re.findall(CVE_REGULAR_EXPRESSION, data.upper())
    return set(cves)


def update_cve_info(df: pd.DataFrame) -> pd.DataFrame:
    """Update CVE details from the description."""
    df.loc[:, "cves"] = df.apply(lambda x: ",".join(get_cves_from_text(x['description'])), axis=1)
    return df
