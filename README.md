[![Build Status](https://travis-ci.com/elvoai/elvo-analysis.svg?branch=master)](https://travis-ci.com/elvoai/elvo-analysis)
[![codecov](https://codecov.io/gh/elvoai/elvo-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/elvoai/elvo-analysis)
[![Documentation Status](https://readthedocs.org/projects/elvo-analysis/badge/?version=latest)](https://elvo-analysis.readthedocs.io/en/latest/?badge=latest)

# Blueno

A platform to quickly build effective machine learning infrastructures for any project.

### Why Blueno?

Reasons are twofold:

- Creating a machine learning pipeline for a given project is laborious and time-consuming. Managing raw data, and all its preprocessed forms, are difficult to keep track of and easy to lose throughout the course of a project. Keeping track and evaluating model results are also difficult and easy to be less rigorous than it should be. Blueno provides an end-to-end workflow for machine learning, with methods for rigorous preprocessing, training, and model evaluating. Blueno also provides plug-and-play style methods for using external data stores (such as Google Cloud Storage) and result analysis (such as Kibana), which makes large-scale data science projects more approachable.

- Blueno also provides a means for mass production of machine learning models. Modern machine learning research is plagued with "Graduate Student Descent," having multiple graduate students twiddling around machine learning parameters and model architectures (waiting for model to train, see results, tweak values, ad infinitum). This is inefficient in both time and talent. Blueno's training methods focus on mass-producing large amounts of models based on user's specifications of different hyperparameters, architectures, and preprocessing methods. This makes training a much more of a long-term, hands-off background process, allowing people to focus on other things in the meantime.

### Getting started
Create a virtual environment, run `pip install -e .`
and then run `pytest`. If the test cases pass you should be good to
start developing.

### Organization

#### Folder Organization

The project contains the following core folders:

- `credentials/` # The other secrets folder
- `blueno/` # Shared code for our ML platform
- `data/` # Contains all downloaded data. Data used in the code should be stored in the cloud.
- `docs/` # Documentation for the project.
- `etl/` # All data pipeline scripts, code which should be scheduled on Airflow
- `logs/` # For storing application logs
- `ml/` # All ML specific scripts, contains the blueno ML tookit as well
- `models/` # For storing trained ML models (HDF5, etc.)

`dashboard`, `etl`, and `ml` should be seen as top-level python
projects.
This means each folder should contain top level scripts with
packages as sub-folders.
As our codebase is small, we are keeping them in a single repo but there
are plans to separate these in the future.


### Contributing
To contribute, create a pull request. Every PR should
 be reviewed by at least one other person. See
[this gist](https://gist.github.com/kashifrazzaqui/44b868a59e99c2da7b14)
for a guide on how to review a PR.

### Other

For developing on GPUs. See the `docs` folder for more info.
