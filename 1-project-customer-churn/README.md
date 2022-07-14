# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Project for implementing, validation and testing model for checking if client is likely to churn. Project is following best practices.

The linkting and tests results can be found at `expermentation.ipynb`.

## Files and data description

```bash
📂 /path/to/project
┣━━ 📂 models                       # models
┃   ┣━━ 🔢 rfc_model.pkl            # random forest classifier model
┃   ┗━━ 🔢 lrc_model.pkl            # logistic regression classifier model
┣━━ 📂 logs                         # logs
┃   ┣━━ 📄 churn_model_tests.logs   # logs for testing
┃   ┗━━ 📄 churn_library.logs       # logs for churn library module
┣━━ 📂 images                       # images and reports
┃   ┣━━ 📂 results                  # results of models performance
┃   ┃   ┣━━ 📊 random_forest_scores.png          # random forest classifier model scores 
┃   ┃   ┣━━ 📊 logistic_regression_scores.png    # logistic regression classifier model scores
┃   ┃   ┣━━ 📊 feature_importance_rfc.png        # feature importance of RFC model
┃   ┃   ┗━━ 📊 feature_importance_lrc.png        # feature importance of LGC model
┃   ┗━━ 📂 eda                      # plots of explaratory data analysis
┃       ┣━━ 📊 total_transactoins_plot.png       # total transactions KDE plot
┃       ┣━━ 📊 quantitative_variables_plot.png   # quantitative variables histogram plot
┃       ┣━━ 📊 correlation_map.png               # correlation map of all variables
┃       ┗━━ 📊 categorical_variables.png         # categorical variables bar plot
┣━━ 📂 data                         # data folder
┃   ┗━━ 📄 bank_data.csv            # raw source data
┣━━ 📄 README.md                    # readme file
┣━━ 🐍 churn_library.py             # main module that implements model pipeline
┣━━ 🐍 churn_script_logging_and_tests.py         # tests
┣━━ 🐍 constants.py                 # module for constants variables
┣━━ 📄 Guide.ipynb                  # guide notebook
┣━━ 📄 churn_notebook.ipynb         # main notebook used for interactive implementation
┣━━ 📄 experimentation.ipynb        # notebook for check-ups and validations
┣━━ 🐋 Dockerfile                   # dockerfile
┣━━ 📄 requirements.txt             # full requirements file
┣━━ 📄 requirements_py3.6.txt       # requirements file for running on Udacity servers
┗━━ 📄 requirements_py3.8.txt       # requirements file for running locally (not complete)
```


## Running Files - locally in environment

Make sure to install requirements at `requirements.txt` using `virtualenv` or `conda` (`miniconda`) environment with Python version 3.6 (or 3.8).

Also one can use `Dockerfile` for isolated environment.



### Running tests

- Using `PyTest`:
```bash
pytest --disable-warnings churn_script_logging_and_tests.py
```

- Using `ipython` shell command:
```bash
ipython churn_script_logging_and_tests.py
```

### Running whole pipeline

Run the following command to run the whole pipeline (from importing data to saving models and reports):
```bash
ipython churn_library.py
```

## Running Files - using dockerfile

Run the following commands:
```bash
docker run --rm -it --entrypoint /bin/ash .
```

### Notes
One can have a look at `experimentation.ipynb` to see what the output of each command gives, or use the commands directly on jupyter notebook if it's tied to your bash shell.