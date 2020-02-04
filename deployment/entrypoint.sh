#!/usr/bin/env bash
set +ex

ecosystem_list=("openshift" "knative" "kubevirt")

# Run the inference for each ecosystem in the inference list.
for ecosystem in ${ecosystem_list[*]};do
  python3 /model_inference_triage_pipeline/run_model_inference.py -days 14 -eco "${ecosystem}" -cve-model bert;
done
