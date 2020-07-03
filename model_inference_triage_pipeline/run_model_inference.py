"""Invokes inference pipeline and uploads result to Object store."""

import argparse
import logging
import sys
import textwrap
import warnings

import arrow
import daiquiri
import pandas as pd

from utils import aws_utils as aws
from utils import cloud_constants as cc
from utils.bq_utils import get_bq_data_for_inference
from utils.storage_utils import write_output_csv, save_data_to_csv
from utils.api_util import save_data_to_db, report_failures, read_probable_cve_data

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)


def main():
    """Entry point for inference pipeline."""
    parser = get_argument_parser()
    args = parser.parse_args()

    ECOSYSTEM = args.eco_system.lower()
    COMPUTE_DEVICE = args.compute_device.lower()

    if COMPUTE_DEVICE != "cpu":
        import torch

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    SEC_MODEL_TYPE = args.security_filter_model.lower()
    if SEC_MODEL_TYPE != "" and SEC_MODEL_TYPE != "gru":
        _logger.error("GRU is the only supported security model type.")
        sys.exit(0)
    CVE_MODEL_TYPE = args.probable_cve_model.lower()
    S3_UPLOAD = args.s3_upload_cves

    day_count, start_time, end_time, date_range = setup_dates_for_triage(
        args.days_since_yday, args.start_date, args.end_date
    )

    _logger.info("Encoding: {}".format(sys.stdout.encoding))
    raw_data = "### atenci\u00f3n\u2013 \xbb as \xf3 控制器添加参 General information * OS: Windows * Hypervisor: Hyper-V * Did you run `crc setup` before starting it (Yes/No)? Yes * Running CRC on: Laptop / Baremetal-Server / VM: Laptop ## CRC version ```bash # Put `crc version` output here CodeReady Containers version: 1.12.0+6710aff OpenShift version: 4.4.8 (embedded in binary) ``` ## CRC status ```bash # Put `crc status --log-level debug` output here DEBU CodeReady Containers version: 1.12.0+6710aff DEBU OpenShift version: 4.4.8 (embedded in binary) Machine 'crc' does not exist. Use 'crc start' to create it. ``` ## CRC config ```bash # Put `crc config view` output here ``` ## Host Operating System ```bash # Put the output of `cat /etc/os-release` in case of Linux # put the output of `sw_vers` in case of Mac # Put the output of `systeminfo` in case of Windows Nom de l\u2019h\xf4te: DESKTOP-U2EVGBM Nom du syst\xe8me d\u2019exploitation: Microsoft Windows 10 Professionnel Version du syst\xe8me: 10.0.18363 N/A version 18363 Fabricant du syst\xe8me d\u2019exploitation: Microsoft Corporation Configuration du syst\xe8me d\u2019exploitation: Station de travail autonome Type de version du syst\xe8me d\u2019exploitation: Multiprocessor Free Propri\xe9taire enregistr\xe9: Jeff MAURY Organisation enregistr\xe9e: Identificateur de produit: 00330-80819-37372-AA630 Date d\u2019installation originale: 18/11/2019, 15:37:22 Heure de d\xe9marrage du syst\xe8me: 11/06/2020, 09:01:18 Fabricant du syst\xe8me: LENOVO Mod\xe8le du syst\xe8me: 20LAS3NJ0H Type du syst\xe8me: x64-based PC Processeur(s): 1 processeur(s) install\xe9(s). [01]\xa0: Intel64 Family 6 Model 142 Stepping 10 GenuineIntel ~1910 MHz Version du BIOS: LENOVO N27ET36W (1.22 ), 04/07/2019 R\xe9pertoire Windows: C:\WINDOWS R\xe9pertoire syst\xe8me: C:\WINDOWS\system32 P\xe9riph\xe9rique d\u2019amor\xe7age: \Device\HarddiskVolume1 Option r\xe9gionale du syst\xe8me: fr;Fran\xe7ais (France) Param\xe8tres r\xe9gionaux d\u2019entr\xe9e: fr;Fran\xe7ais (France) Fuseau horaire: (UTC+01:00) Bruxelles, Copenhague, Madrid, Paris M\xe9moire physique totale: 32\xa0379 Mo M\xe9moire physique disponible: 6\xa0731 Mo M\xe9moire virtuelle\xa0: taille maximale: 59\xa0243 Mo M\xe9moire virtuelle\xa0: disponible: 13\xa0451 Mo M\xe9moire virtuelle\xa0: en cours d\u2019utilisation: 45\xa0792 Mo Emplacements des fichiers d\u2019\xe9change: C:\pagefile.sys Domaine: WORKGROUP Serveur d\u2019ouverture de session: \\DESKTOP-U2EVGBM Correctif(s): 13 Corrections install\xe9es. [01]: KB4552931 [02]: KB4497165 [03]: KB4516115 [04]: KB4517245 [05]: KB4524569 [06]: KB4528759 [07]: KB4537759 [08]: KB4538674 [09]: KB4541338 [10]: KB4552152 [11]: KB4560959 [12]: KB4561600 [13]: KB4560960 Carte(s) r\xe9seau: 5 carte(s) r\xe9seau install\xe9e(s). [01]: Intel(R) Ethernet Connection (4) I219-LM Nom de la connexion\xa0: Ethernet \xc9tat\xa0: Support d\xe9connect\xe9 [02]: TAP-Windows Adapter V9 Nom de la connexion\xa0: Ethernet 2 DHCP activ\xe9\xa0: Oui Serveur DHCP\xa0: 10.36.127.254 Adresse(s) IP [01]: 10.36.113.103 [02]: fe80::d527:58a7:4965:c2ad [03]: Hyper-V Virtual Ethernet Adapter Nom de la connexion\xa0: vEthernet (Default Switch) DHCP activ\xe9\xa0: Non Adresse(s) IP [01]: 172.17.134.145 [02]: fe80::1dbc:68b3:1e6d:b893 [04]: Intel(R) Dual Band Wireless-AC 8265 Nom de la connexion\xa0: Wi-Fi DHCP activ\xe9\xa0: Oui Serveur DHCP\xa0: 192.168.0.254 Adresse(s) IP [01]: 192.168.0.42 [02]: fe80::8548:85c0:593:b8e9 [03]: 2a01:e35:2ffe:3020:a02b:4519:8489:a51b [04]: 2a01:e35:2ffe:3020:8548:85c0:593:b8e9 [05]: Bluetooth Device (Personal Area Network) Nom de la connexion\xa0: Connexion r\xe9seau Bluetooth \xc9tat\xa0: Support d\xe9connect\xe9 Configuration requise pour Hyper-V: Un hyperviseur a \xe9t\xe9 d\xe9tect\xe9. Les fonctionnalit\xe9s n\xe9cessaires \xe0 Hyper-V ne seront pas affich\xe9es. ``` ### Steps to reproduce 1. `crc setup --enable-experimental-features` 2. 3. 4. ### Expected OK status ### Actual ``` INFO Checking if oc binary is cached INFO Checking if podman remote binary is cached INFO Checking if goodhosts binary is cached INFO Checking if CRC bundle is cached in '$HOME/.crc' INFO Checking if running as normal user INFO Checking Windows 10 release INFO Checking if Hyper-V is installed and operational INFO Checking if user is a member of the Hyper-V Administrators group INFO Checking if Hyper-V service is enabled INFO Checking if the Hyper-V virtual switch exist INFO Found Virtual Switch to use: Default Switch INFO Checking if user is allowed to log on as a service INFO Will run as admin: Running secedit export command INFO Enabling service log on for user FATA Unable to get sid for username: Jeff MAURY: Get-LocalUser : Impossible de trouver un param\xe8tre positionnel acceptant l'argument \xab\xa0MAURY\xa0\xbb. Au caract\xe8re Ligne:1 : 2 + (Get-LocalUser -Name Jeff MAURY).Sid.value + ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ + CategoryInfo : InvalidArgument : (:) [Get-LocalUser], ParameterBindingException + FullyQualifiedErrorId : PositionalParameterNotFound,Microsoft.PowerShell.Commands.GetLocalUserCommand : exit status 1 ``` ### Logs Before gather the logs try following if that fix your issue ```bash $ crc delete -f $ crc cleanup $ crc setup $ crc start --log-level debug ``` Please consider posting the output of `crc start --log-level debug` on http://gist.github.com/  and post the link in the issue. atenci\u00f3n\u2013"
    test_list = [[raw_data]]
    temp_df = pd.DataFrame(test_list, columns=[u'col_A'])
    save_data_to_csv(temp_df, True, "bq_data", "test", ECOSYSTEM, "dummy")
    _logger.info("Saved dummy data to s3 completed")

    df = get_bq_data_for_inference(ECOSYSTEM, day_count, date_range)
    df = run_inference(df, CVE_MODEL_TYPE)
    write_output_csv(
        start_time,
        end_time,
        cve_model_type=CVE_MODEL_TYPE,
        ecosystem=ECOSYSTEM,
        df=df,
        s3_upload=S3_UPLOAD,
    )

    # Save data to database using api server
    if not cc.SKIP_INSERT_API_CALL:
        probable_cve_data = read_probable_cve_data(start_time, end_time, CVE_MODEL_TYPE, S3_UPLOAD, ECOSYSTEM)
        updated_df, failed_to_insert = save_data_to_db(probable_cve_data, ECOSYSTEM)
        # Save data to csv file those are failed to ingest
        if len(failed_to_insert) > 0:
            report_failures(updated_df, failed_to_insert, start_time, end_time, S3_UPLOAD, ECOSYSTEM)


# noinspection PyTypeChecker
def get_argument_parser():
    """Define all the command line arguments for the program."""
    description: str = textwrap.dedent(
        """
        This script can be used to run our AI models for probable vulnerability predictions.
        Check usage patterns below.
        """
    )
    epilog: str = textwrap.dedent(
        """
         Usage patterns for inference script
         -----------------------------------
         The -days flag should be used with number of prev days data you want to pull

         1. GRU Models (Baseline): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=gru
         2. BERT Model (CVE): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=bert
         3. CPU inference:
                python run_model_inference.py -days=7 -device=cpu -sec-model=gru -cve-model=gru
         """
    )
    parser = argparse.ArgumentParser(
        prog="python",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-days",
        "--days-since-yday",
        type=int,
        default=7,
        help="The number of days worth of data to retrieve from GitHub including yesterday",
    )

    parser.add_argument(
        "-eco",
        "--eco-system",
        type=str,
        default="openshift",
        choices=["openshift", "knative", "kubevirt"],
        help="The eco-system to monitor",
    )

    parser.add_argument(
        "-device",
        "--compute-device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "GPU", "CPU"],
        help="[Not implemented should work automatically] The backend device to run the AI models on - GPU or CPU",
    )

    parser.add_argument(
        "-sec-model",
        "--security-filter-model",
        type=str,
        default="gru",
        choices=["gru", "GRU"],
        help="The AI Model to use for security filtering - Model 1",
    )

    parser.add_argument(
        "-cve-model",
        "--probable-cve-model",
        type=str,
        default="gru",
        choices=["gru", "GRU", "bert", "BERT", "bert_torch", "BERT_TORCH"],
        help="The AI Model to use for probable CVE predictions - Model 2",
    )

    parser.add_argument(
        "-s3-upload",
        "--s3-upload-cves",
        type=bool,
        default=False,
        choices=[False, True],
        help=(
            "Uploads inference CSVs to S3 bucket - should have write access to the appropriate S3 bucket."
        ),
    )

    parser.add_argument(
        "-sd",
        "--start-date",
        default="",
        help="If running for a custom interval, set this and the end-date in yyyy-mm-dd format.",
    )

    parser.add_argument(
        "-ed",
        "--end-date",
        default="",
        help="If running for a custom interval, set this and the start-date in yyyy-mm-dd format.",
    )

    return parser


def setup_dates_for_triage(days_since_yday, start_date_user, end_date_user):
    """Prepare date range for triage."""
    # ======= DATES SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- DATES SETUP FOR GETTING GITHUB BQ DATA -----")
    if start_date_user != "" and end_date_user != "":
        start_time = arrow.get(start_date_user, "YYYY-MM-DD")
        end_time = arrow.get(end_date_user, "YYYY-MM-DD")
        day_count = (end_time - start_time).days
        date_range = [
            dt.format("YYYYMMDD") for dt in arrow.Arrow.range("day", start_time, end_time)
        ]
        return day_count, start_time, end_time, date_range

    # Don't change this
    present_time = arrow.now()

    # CHANGE NEEDED
    # to get data for N days back starting from YESTERDAY
    # e.g if today is 20190528 and DURATION DAYS = 2 -> BQ will get data for 20190527, 20190526
    # We don't get data for PRESENT DAY since github data will be incomplete on the same day
    # But you can get it if you want but better not to for completeness :)

    # You can set this directly from command line using the -d or --days-since-yday argument
    day_count = days_since_yday or 3  # Gets 3 days of previous data including YESTERDAY

    # Don't change this
    # Start time for getting data
    start_time = present_time.shift(days=-day_count)

    # Don't change this
    # End time for getting data (present_time - 1) i.e yesterday
    # you can remove -1 to get present day data
    # but it is not advised as data will be incomplete
    end_time = present_time.shift(days=-1)

    date_range = [dt.format("YYYYMMDD") for dt in arrow.Arrow.range("day", start_time, end_time)]
    _logger.info(
        "Data will be retrieved for Last N={n} days: {days}\n".format(n=day_count, days=date_range)
    )
    return day_count, start_time, end_time, date_range


def run_inference(df, CVE_MODEL_TYPE="bert") -> pd.DataFrame:
    """Run inference pipeline."""
    if cc.S3_MODEL_REFRESH.lower() == "true":
        aws.s3_download_folder(aws.S3_OBJ.Bucket(cc.S3_BUCKET_NAME_MODEL), "model_assets", "/")

    if "torch" not in CVE_MODEL_TYPE.lower():
        from models.run_tf_models import (
            run_bert_tensorflow_model,
            run_gru_cve_model,
            run_tensorflow_security_classifier,
        )

        df = run_tensorflow_security_classifier(df)
        if CVE_MODEL_TYPE == "gru":
            df = run_gru_cve_model(df)

        if CVE_MODEL_TYPE == "bert":
            df = run_bert_tensorflow_model(df)
    else:
        from models.run_torch_model import run_torch_cve_model_bert
        from models.run_tf_models import run_tensorflow_security_classifier

        # Re-use the GRU based security/non-security classifier then pipe its output to the new BERT model.
        df = run_torch_cve_model_bert(run_tensorflow_security_classifier(df))
    return df


if __name__ == "__main__":
    main()
    _logger.info("All done!")
