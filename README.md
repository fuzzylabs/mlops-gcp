# Introduction

This repository is our end-to-end MLOps example for [Google Cloud AI Platform](https://cloud.google.com/ai-platform). It contains everything you would need to get up-and-running with an MLOps stack, and covers a number of different use-cases and machine learning frameworks. Before diving into the details, here's an overview of the three topics we've set out to address in this project:

## Experimenting and training

Enabling teams to collaboratively develop a model with confidence. This includes:

* Data versioning.
* Data labelling.
* Provenance tracking.
* Experiment tracking.
* Collaborative development.

## Production deployment

This not only covers how to deploy a model to production, but also how to re-train a model on demand in the cloud.

## On-going monitoring

Once a model has been deployed we need to monitor it. Why?

* Data can drift over time. When it does, the model might not be producing expected results anymore, and could need re-training.
* Models need to tolerate unexpected inputs, such as values outside of the expected range for a particular feature.
* A served model may go wrong, so we need to monitor error rates, as we would with any software deployed to production.

# Contents

* [MLOps concepts explained](#concepts).
* Examples
** [Wine quality with sklearn](examples/sklearn-wine/README.md).

<a name="#concepts">
# MLOps concepts explained

...
