#!/usr/bin/env bash
set +ex

ecosystem_list=("openshift" "knative" "kubevirt")

# Run the inference for each ecosystem in the inference list.
for ecosystem in ${ecosystem_list[*]};do
  python3 /model_inference_triage_pipeline/run_model_inference.py -days "${DAYS}" -eco "${ecosystem}" -cve-model "${CVE_MODEL}" -s3-upload True -device=cpu;
done
