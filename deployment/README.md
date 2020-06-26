# Deployment of BERT inference pipeline to Openshift

The following README explains the various components in this folder and elaborates on the various strategies
that can be employed to run an inference job on Openshift.

# Table of Contents
- [Initial Setup of Secrets](#initial-setup-of-secrets)
- [Component Descriptions](#component-descriptions)
    - [entrypoint.sh](#entrypointsh)
    - [Dockerfile](#dockerfile)
    - [Multi-phase build Dockerfiles](#multi-phase-build-dockerfiles)
    - [Openshift Template- DeploymentConfig/Service/Route (template-service.yaml)](#openshift-template--deploymentconfigserviceroute)
    - [Openshift Template- Openshift Job (template-job.yaml)](#openshift-template--openshift-job)
    - [Openshift Template- Openshift CronJob (template-cronjob.yaml)](#openshift-template--openshift-cronjob)
    - [Configurable variables in template](#configurable-variables-in-template)
- [Deployment on a cluster with only CPU nodes](#deployment-on-a-cluster-with-only-cpu-nodes)
- [Setup of a GPU based cluster](#setup-of-a-gpu-based-cluster)
- [Deployment on a cluster with GPU nodes](#deployment-on-a-cluster-with-gpu-nodes)
- [Inference Result Storage](#inference-result-storage)

## Initial Setup of secrets

Using any of the templates in this folder requires two secrets that need to be present on the cluster, the first
of the two contains the aws keys where the data will be pushed and models will be read from. Here is the
definition of the secret:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aws
type: Opaque
data:
  aws_access_key_id: <Your base 64 encoded access key here>
  aws_secret_access_key: <Your base 64 encoded secret key here>
```
To base64 encode a piece of text, use the command:
```bash
echo -n "<Your text here>" | base64
```
**N.B.** on a Mac install and use the command `gbase64` instead of the default `base64`.

Save the full yaml definition to a file, say `aws-secret.yaml` and use this command
to apply the secret to your namespace:

```bash
oc apply -f aws-secret.yaml
```

Similarly, base64 encode the complete contents of the bigquery service account `key.json` file as is and create
a new secret definition, say `google-secret.yaml`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: google-services-secret
type: Opaque
data:
  google-services.json: |-
    <Full multi-line secret content goes here.>
```
If everything went according to plan till this step, you now have the following two secrets on the cluster:
```bash
$ oc get secrets
NAME                       TYPE                                  DATA   AGE
aws                        Opaque                                2      22h
google-services-secret     Opaque                                1      22h
```

You can now proceed with the deployment.

## Component Descriptions
- entrypoint.sh
- Dockerfile
- Multi-phase build Dockerfiles (Dockerfile-base and Dockerfile-from-base)
- Openshift Template- DeploymentConfig/Service/Route (template-service.yaml)
- Openshift Template- Openshift Job (template-job.yaml)
- Openshift Template- Openshift CronJob (template-cronjob.yaml)
- Configurable variables in template

### entrypoint.sh

The Entrypoint script, contains the details of the actual processing that will happen inside the Pod. In order
to add new ecosytems this needs to be modified.

### Dockerfile

The Dockerfile uses the CUDA 10.0 (only version of CUDA comaptible with Tensorflow 1.15) UBI-7 base image from
Nvidia. We are using Tensorflow 1.15 as it has both a CPU and GPU based backends in the same package so the same
container image will work for both the deployment strategies. If the Pod is scheduled with a GPU resource
(more on that in the GPU deployment section) then the CUDA libraries will "discover" the GPU resource and
processing will move to the GPU, otherwise everything proceeds as normal on the CPU.

To change the Dockerfile, ensure that the CUDA version and the Tensorflow version that are used are compatible.

### Multi-phase build Dockerfiles

The multi-phase build Dockerfiles (Dockerfile-base and Dockerfile-from-base) can save a lot of time and headache
during the development phase as if you are not changing the requirements a simple build of Dockerfile-base
once and changing the `FROM` directive of the `Dockerfile-from-base` configuration or even keeping it as is if
the requirements are never changed after the first inception of this document and workflow.

### Openshift Template- DeploymentConfig/Service/Route

This Template defines a Deployment Config object and creates an associated service and route by exposing a port
of the Pod. This will come in handy if there is a requirement for a readily-available online exection of the
inference workflow at any point. In order to make this template usable an API wrapper needs to be defined that
will run at the defined port. Without that, the job will continue to restart until it finally starts to CrashLoop.


### Openshift Template- Openshift Job

This is the template that defines and Openshift Job (and not CronJob) and the one that should be used in the
case where the inference is not run periodically rather on-demand when required.

### Openshift Template- Openshift CronJob

This is the template that defines an Openshift CronJob and should be used when periodic inference execution is
required instead of ad-hoc. Adjust the schedule accordingly, the one hardcoded currently is to run
the job at 12 A.M. (00:00) daily.

### Configurable variables in template

- S3_MODEL_BUCKET - This is the bucket from which the models will be read.
- S3_INFERENCE_BUCKET - This is the bucket to which the inference results will be written
- DAYS - The number of days for which the job has to be run

## Deployment on a cluster with only CPU nodes

Our default templates only request the CPU resource and are as such configured to work as-is on a CPU-only
cluster since the image is common for both CPU and GPU based environments. In order to run inference on
a CPU node in an ad-hoc fashion simply use the Job template with appopriate adjustments to the environment
if required.

```bash
oc process -f template-job.yaml | oc create -f -
```

## Setup of a GPU based cluster

### Adding GPU nodes to a cluster

If you're doing so at the time of cluster installation, install the cluster by first creating an ignition
config:

```openshift-install create install-config --dir=<Your cluster dir>```

Now edit the worker pool the `install-config.yaml` that has been created in your cluster directory.
```yaml
compute:
- hyperthreading: Enabled
  name: worker
  platform:
      aws:
        type: p3.2xlarge
  replicas: 2
```
`p3.2xlarge` instances contain a single Tesla V100 GPU which is currently state of the art AWS GPU resource.
Now go forward with an install of the cluster as with:
```openshift-install create cluster --dir=<Your cluster dir that contains the install-config>```

If you want to add GPU nodes to an already running cluster, follow [this tutorial](https://blog.openshift.com/creating-a-gpu-enabled-node-with-openshift-4-2-in-amazon-ec2/).


### Installing the operators

In order to use GPU as a resource we require the [special-resource-operator](https://github.com/openshift-psap/special-resource-operator)
to be installed on the cluster, which in turn depends on the [cluster-nfd-operator](https://github.com/openshift/cluster-nfd-operator) operator.
This should be a one click install through operatorhub but if that fails clone these repositories, make sure you have
a valid `KUBECONFIG` environment set and run `make deploy` and do a manual deployment of the master branch. I
suggest using a Makefile based deployment from the repositories as the one click install requires a manual 
creation of the `special resource` and the `node-feature-discovery` objects.

- Cluster nfd operator installation(Tested with Openshift 4.3, nfd operator version: c4015b8)
 - First you need to build an image of the operator's source code in the master branch after making the appropriate
   changes to the Makefile. ex:
   ```Makefile
   -REGISTRY       ?= quay.io
   -ORG            ?= zvonkok
   -TAG            ?= $(shell git branch | grep \* | cut -d ' ' -f2)
   +REGISTRY       ?= docker.io
   +ORG            ?= avgupta
   +TAG            ?= master
   IMAGE          ?= ${REGISTRY}/${ORG}/cluster-nfd-operator:${TAG}
   NAMESPACE      ?= openshift-nfd
   PULLPOLICY     ?= IfNotPresent
   ```
 - The build the image that you have now specified, ex:
 ```bash
    docker build -t avgupta/cluster-nfd-operator:master -f Dockerfile .
 ```
 - Followed by deployment with `make deploy`.
 
If all goes according to plan, you will have an `openshift-sro` and `openshift-nfd` namespace
with a bunch of running(and not crashlooping) pods. The `openshift-sro` namespace takes a while to get
populated, basically don't try anything before the `exporter` pods come up.

Once both the operators have been successfully installed, your control plane is now ready to schedule workloads
on the GPU.

## Deploying on a cluster with GPU based nodes.

Once the setup of a GPU node cluster is complete this step is quite trivial, as now `nvidia.com/gpu` will be
available as a resource. To use it in a template simply change the `resources` part of your spec:
```yaml
resources:
            requests:
              nvidia.com/gpu: 1
              cpu: ${CPU_REQUEST}
              memory: ${MEMORY_REQUEST}
            limits:
              nvidia.com/gpu: 1
              cpu: ${CPU_LIMIT}
              memory: ${MEMORY_LIMIT}
```
Partial GPU allocation is currently not possible, the request and limit have to be a whole number.

If everything before this was done properly, once you apply the relevant template as with:
```bash
oc process [templatename.yaml] | oc create -f -
```
The Pod should be in `ContainerCreating` state for a while and then move on to the `running` state. If the GPU
resource has not been properly registered the pod will remain un-schedulable and `oc get events` should
get you the appropriate error message from the namespace.

## Inference Result Storage

The inference results will be stored in the appropriately configured AWS bucket if the keys have appropriate write permission.
See: [Configurable variables](#configurable-variables-in-template).
