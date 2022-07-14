'''
Module for testing churn_library module.

Author: Dauren Baitursyn
Date: 10.07.22
'''

# import libraries
import os
import logging

from pathlib import Path
import pytest

import constants
import churn_library as cl

# configuration for logging
logging.basicConfig(
    handlers=[
        logging.FileHandler(constants.TEST_LOGS_FILE_PATH),
        logging.StreamHandler()],
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture
def get_data():
    '''
    Read raw data.
    '''
    return cl.import_data(constants.DATA_FILE_PATH)


@pytest.fixture
def get_target_transformed_data(get_data):
    '''
    Get data with transformed target variable.
    '''
    df_data = get_data
    df_data[constants.TRANSFORMED_TARGET_VARIABLE] = df_data[constants.RAW_TARGET_VARIABLE].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df_data


@pytest.fixture
def get_transformed_data(get_data):
    '''
    Get data with feature engineering.
    '''
    df_data = get_data
    return cl.perform_feature_engineering(
        df_data,
        constants.TRANSFORMED_TARGET_VARIABLE
    )


def test_import():
    '''
    Test data import.
    '''
    try:
        df_data = cl.import_data(constants.DATA_FILE_PATH)
        logging.info('SUCCESS: Testing import_data.')
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda - the file wasn't found.")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data - the file doesn't appear to have rows and columns.")
        raise err


def test_eda(get_data):
    '''
    Test perform eda function.
    '''
    df_data = get_data

    try:
        cl.perform_eda(df_data)
    except (KeyError, IndexError) as err:
        logging.error(
            ('ERROR: Testing perform_eda - corresponding categorical and/or '
             'quantitative columns are not present in DataFrame.'))
        raise err
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing perform_eda - folder for EDA reports doesn't exit.")
        raise err

    try:
        assert Path(
            os.path.join(
                constants.EDA_PLOTS_FOLDER_PATH,
                'categorical_variables_plot.png')).exists()
        assert Path(
            os.path.join(
                constants.EDA_PLOTS_FOLDER_PATH,
                'quantitative_variables_plot.png')).exists()
        assert Path(
            os.path.join(
                constants.EDA_PLOTS_FOLDER_PATH,
                'total_transactions_plot.png')).exists()
        assert Path(
            os.path.join(
                constants.EDA_PLOTS_FOLDER_PATH,
                'correlation_map.png')).exists()
        assert Path(
            os.path.join(
                constants.EDA_PLOTS_FOLDER_PATH,
                'total_transactions_plot.png')).exists()
    except AssertionError as err:
        logging.error('ERROR: Testing perform_eda - report(s) not saved.')
        raise err


def test_encoder_helper(get_target_transformed_data):
    '''
    Test encoder helper.
    '''
    df_data = get_target_transformed_data

    try:
        df_cat_encoded = cl.encoder_helper(
            df_data, constants.to_encode_variables,
            constants.TRANSFORMED_TARGET_VARIABLE)
    except KeyError as err:
        logging.error(
            ('ERROR: Testing encoder_helper - corresponding categorical '
             'variables not found in dataframe.'))
        raise err

    try:
        assert df_cat_encoded.shape[0] > 0
        assert df_cat_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            ("ERROR: Testing encoder_helper - the encoded data doesn't "
             "appear to have rows and columns."))
        raise err


def test_perform_feature_engineering(get_data):
    '''
    Test perform_feature_engineering.
    '''
    df_data = get_data

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            df_data,
            constants.TRANSFORMED_TARGET_VARIABLE)
    except KeyError as err:
        logging.error(
            ('ERROR: Testing perform_feature_engineering - '
             'categorical, quantitative and/or target variables '
             ' are/is not found in the dataframe.'))
        raise err

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        assert x_train.shape[1] == x_test.shape[1]
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
    except AssertionError as err:
        logging.error(
            ("ERROR: Testing perform_feature_engineering - the transformed "
             "data doesn't appear to have rows and columns."))
        raise err


def test_train_models(get_transformed_data):
    '''
    Test train_models.
    '''
    x_train, x_test, y_train, y_test = get_transformed_data

    try:
        cl.train_models(x_train, x_test, y_train, y_test)
    except FileNotFoundError as err:
        logging.error(
            ("ERROR: Testing train_models - model and/or "
             "report folder doesn't exist."))
        raise err
    except TypeError as err:
        logging.error(
            ('ERROR: Testing train_models - models are not instance of '
             'sklearn.linear_model.LogisticRegression or '
             'sklearn.ensemble.RandomForestClassifier.'))
        raise err
    except (AttributeError, IndexError) as err:
        logging.error(
            ("ERROR: Testing train_models - passed data doesn't "
             "correpond to trained model."))
        raise err

    try:
        assert Path(
            os.path.join(
                constants.MODEL_RESULT_FOLDER_PATH,
                'random_forest_scores.png')).exists()
        assert Path(
            os.path.join(
                constants.MODEL_RESULT_FOLDER_PATH,
                'logistic_regression_scores.png')).exists()
        assert Path(
            os.path.join(
                constants.MODEL_RESULT_FOLDER_PATH,
                'feature_importance_rfc.png')).exists()
        assert Path(
            os.path.join(
                constants.MODEL_RESULT_FOLDER_PATH,
                'feature_importance_lrc.png')).exists()
        assert Path(
            os.path.join(
                constants.MODEL_SAVE_FOLDER_PATH,
                'rfc_model.pkl')).exists()
        assert Path(
            os.path.join(
                constants.MODEL_SAVE_FOLDER_PATH,
                'lrc_model.pkl')).exists()
    except AssertionError as err:
        logging.error(
            ('ERROR: Testing train_models - failed to save '
             'models, repots and/or images.'))
        raise err


if __name__ == '__main__':

    test_import()

    df_data_imported = cl.import_data(constants.DATA_FILE_PATH)
    test_eda(df_data_imported)

    df_data_target_transformed = df_data_imported.copy()
    df_data_target_transformed[constants.TRANSFORMED_TARGET_VARIABLE] = df_data_imported[
        constants.RAW_TARGET_VARIABLE].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_encoder_helper(df_data_target_transformed)

    test_perform_feature_engineering(df_data_imported)

    transformed_data = cl.perform_feature_engineering(
        df_data_imported,
        constants.TRANSFORMED_TARGET_VARIABLE)
    test_train_models(transformed_data)
