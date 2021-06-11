# Introduction

In this reference example we demonstrate MLOps on Google Cloud Platform with Vertex. This represents what we at Fuzzy Labs consider to be _MLOps done right_.

At the beginning of this project, we set out to address the following questions:

* How do we version data?
* How would two data scientists work collaboratively on a model?
* How do we track experiments?
* How do we set up a training pipeline in the cloud?
* How do we test the model?
* How do we serve the model?
* How do other software components interact with the model?
* How do we monitor the model the model on an ongoing basis?

## Table of contents

The README has two parts. First, we explain the concepts that underlie the reference example. Second, we explain step-by-step how to setup and run the example in your GCP environment.

* **[Concepts](#concepts)**
* **[How to run the example - step-by-step](#running)**

<a name="concepts"></a>
# Concepts

<!-- perhaps move this paragraph further down -->
Any productionised machine learning project will consist not only of models but other software components that are necessary in order to make those models useful. We will typically be building models along-side other pieces of software. Both of these need to be tracked, deployed, and monitored, but the approach taken for models differs somewhat from other kinds of software.
<!-- -->

A machine learning model passes through a few stages of maturity:

## Experimental phase

We imagine a team of data scientists starting from scratch on a particular problem. Every problem is different but we can still introduce some tools in this phase that will make life easier.

### The data

<!-- what about VC on unstructured data? -->
The data may not be well-understood, and it may be incomplete. It's important to have data version control from the very start, because:

* It's easier for a team to share data while ensuring that everybody is working with the same version of that data.
* It allows us to track changes over time.
* It allows us to link every experiment to a specific data version.

We use DVC to do data versioning.

### The code

As we're going to be training a model, we're going to need to write some code as well. Code versioning is just as important as data versioning, for exactly the same reasons as stated above.

We're using Git to track code versions. Additionally, it's worth noting that DVC interoperates with Git, so this single code repository is enough to get somebody up-and-running with everything they will need in order to train the model.

Training a model involves a few steps. At the very least, we must prepare data and then run a training script. We use DVC to specify a training pipeline. Something to keep in mind: we're going to be talking about two different kinds of pipeline: as well as the model training pipeline, there will be a deployment pipeline, which we'll come to soon.

### The experiments

Every run of the model pipeline gets logged to a central location. Specifically, we record:

* When it ran, who ran it, and where it ran.
* The Git commit associated with the experiment.
* The data version associated with the experiment.
* The hyperparameters in use.
* The performance of the model.

This way, anybody on the team is able to review past experiments and reproduce them consistently.

We use Sacred for experiment tracking.

## Adding cloud training infrastructure (Vertex AI)

While at the start of a project we're usually doing everything locally, on our own computers, we ultimately want the ability to train a model on cloud-based resources. This gives us more computational power, but it also centralises training and prepares us for cloud-based deployment, which will come later.

By this point we've already got a model training pipeline in DVC, but we add an option to run the training itself on Google Vertex. Running it locally is still possible, of course.

<!-- need to explain a little bit more of what the pipeline entails and where the handoff is to GCP. Also, how data is accessed differently in GCP vs local -->

## Training plus deployment: CI/CD

Finally, we want to deploy a model. We introduce CI/CD, using Circle CI, for this. A Circle pipeline itself invokes the model pipeline. The model pipeline in turn starts a training job on Vertex. It also pushes an experiment to experiment tracking, and a trained model to the Vertex model registry.

The model is deployed along with an endpoint, which exposes the model for online inference.

## Monitoring

<!-- TODDO -->

## Project layout
<!--
data/{...}
models/model1

models/pipelines/{p1, p2....}   ->
   every time a pipeline runs, whether locally or on Vertex, an experiment must be logged centrally.
   What is logged? the tasks themselves (input + output), and the lineage
   What goes into a lineage? versioned inputs, outputs, versioned data
   The thing that runs the pipeline builds the lineage

services/...

.circle/pipelines
-->

<a name="running"></a>
# How to run the example - step-by-step
<!--TODO-->

## Preparation
```
python -m venv env/
source env/bin/activate
pip install -r requirements.txt
dvc pull
```
## Step 0 -- locally train a model
## Step 1 -- locally train with DVC pipeline
* Copy `dvc-step1.yaml` to `dvc.yaml`
* `dvc repro` to run training

### Experiment tracking

This step uses [Sacred](https://github.com/IDSIA/sacred) for experiment tracking. The results are saved to MongoDB.
To run MongoDB locally with Docker:

```
docker run --name mongo -p 27017:27017 -d mongo:latest
```

To view the experiments, you can use [Omniboard](https://vivekratnavel.github.io/omniboard/#/README). To start Omniboard locally with the local MongoDB:
```
docker run -it --rm -p 9000:9000 --network host --name omniboard vivekratnavel/omniboard -m localhost:27017:sacred
```

and open dashboard in the browser [http://localhost:9000](http://localhost:9000)
## Step 2 -- train on Vertex AI
* Copy `dvc-step2.yaml` to `dvc.yaml`
* `dvc repro` to run training
## Step 3 -- train and deploy on Vertex AI
* Copy `dvc-step3.yaml` to `dvc.yaml`
* `dvc repro` to run training
* `cd step3 && python3 deploy-vertex.py` to deploy
