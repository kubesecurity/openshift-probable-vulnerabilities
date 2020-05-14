import argparse
from pathlib import Path
from models.run_torch_model import run_torch_cve_model_bert
import pandas as pd


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a Pytorch .bin model on historical data."
    )
    parser.add_argument(
        "--data-path",
        metavar="dpath",
        type=str,
        help="the path to the entire historical data.",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        metavar="mpath",
        type=str,
        help="the path to the pytorch model files.",
        required=True,
    )
    parser.add_argument(
        "--excel-path",
        metavar="xlpath",
        type=str,
        help="the path to the directory containing the triage excel files with the final result.",
        required=True,
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    ground_truth = get_triaged_csv_data(args)
    print("Ground truth size: {}".format(ground_truth["url"].unique().shape[0]))
    security_issues = get_all_security_cves(args)
    triaged_csv = do_triage(security_issues, args)
    print("Number of issues marked as csv: {}".format(triaged_csv["url"].unique().shape[0]))
    found = triaged_csv[triaged_csv["url"].isin(ground_truth["url"].unique())]
    print("Number of true positives detected by model: {}".format(found["url"].unique().shape[0]))


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"Feeedback": "Feedback",})


def get_triaged_csv_data(args) -> pd.DataFrame:
    path = Path(args.excel_path)
    triaged_openshift = pd.read_excel(
        path.joinpath("OpenShift Probable vulnerabilities.xlsx"), sheet_name=None
    )
    triaged_openshift_combined = pd.concat(
        (sanitize_column_names(v) for k, v in triaged_openshift.items()), sort=False
    ).reset_index(drop=True)
    triaged_kubevirt = pd.read_excel(
        path.joinpath("Kubevirt Probable vulnerabilities.xlsx"), sheet_name=None
    )
    triaged_kubevirt_combined = pd.concat(
        (sanitize_column_names(v) for k, v in triaged_kubevirt.items()), sort=False
    ).reset_index(drop=True)
    triaged_knative = pd.read_excel(
        path.joinpath("Knative Probable vulnerabilities.xlsx"), sheet_name=None
    )
    triaged_knative_combined = pd.concat(
        (sanitize_column_names(v) for k, v in triaged_knative.items()), sort=False
    ).reset_index(drop=True)
    triaged_openshift_combined["ecosystem"] = "openshift"
    triaged_knative_combined["ecosystem"] = "knative"
    triaged_kubevirt_combined["ecosystem"] = "kubevirt"
    all_csv = pd.concat(
        [triaged_openshift_combined, triaged_knative_combined, triaged_kubevirt_combined], sort=True
    ).reset_index(drop=True)
    all_csv = all_csv[
        ~all_csv["triage_feedback_comments"].isna()
        | ~all_csv["Feedback"].isna()
        | ~all_csv["triage_is_cve"].isna()
    ]
    return get_cve_rows(all_csv)


def get_cve_rows(all_csv: pd.DataFrame) -> pd.DataFrame:
    cve_rows = all_csv[
        ~all_csv["triage_is_cve"].isna()
        | all_csv["Feedback"].str.strip().str.lower().str.contains("yes")
        | all_csv["triage_feedback_comments"].str.strip().str.lower().str.contains("yes")
    ]
    # Inspect this for debugging:
    # non_cve_rows = all_csv[~all_csv.index.isin(cve_rows.index)]
    return cve_rows


def get_all_security_cves(args) -> pd.DataFrame:
    data_path = Path(args.data_path)
    security_csv_list = list(data_path.glob("**/*probable_security_and_cves*.csv"))
    security_issue_df = pd.concat(
        (pd.read_csv(s) for s in security_csv_list), sort=True
    ).reset_index(drop=True)
    security_issue_df["cve_model_flag"] = 0
    return security_issue_df


def do_triage(data_frame, args):
    return run_torch_cve_model_bert(data_frame, args.model_path, batch_size_prediction=8)


if __name__ == "__main__":
    main()
