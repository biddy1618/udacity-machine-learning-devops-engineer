'''
Module for finding customer who are likely to churn.

Author: Dauren Baitursyn
Date: 08.07.22
'''

# import libraries
import logging
import os

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import constants

# configuration for logging
logging.basicConfig(
    handlers=[
        logging.FileHandler(constants.MODEL_LOGS_FILE_PATH),
        logging.StreamHandler()],
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df_data: pandas dataframe
    '''

    # import file as DataFrame object
    df_data = pd.DataFrame()
    try:
        df_data = pd.read_csv(pth, index_col=0)
    except FileNotFoundError as err:
        logging.error('ERROR - File not found at specified path - "%s".', pth)
        raise err
    except pd.errors.ParserError as err:
        logging.error(
            'ERROR - Make sure that file at path - ""%s" - is in CSV format.',
            pth)
        raise err

    logging.info('SUCCESS - reading CSV file at path - "%s".', pth)

    return df_data


def perform_eda(df_data):
    '''
    perform eda on df and save figures to images folder
    input:
        df_data: pandas dataframe

    output:
        None
    '''

    # save barplots for categorical variables
    fig, axes = plt.subplots(2, 3, figsize=(30, 10))
    for i, col in enumerate(constants.cat_columns):
        row_i = i // 3
        col_i = i % 3
        try:
            df_data[col].value_counts('normalize').plot.bar(
                figure=fig, ax=axes[row_i][col_i])
        except (KeyError, IndexError) as err:
            logging.error(
                ('ERROR - make sure that corresponding '
                 'categorical columns are present in DataFrame.')
            )
            raise err
        axes[row_i][col_i].set_title(col, fontdict={'fontsize': 'x-large'})
        axes[row_i][col_i].tick_params(axis='x', rotation=0)

    fig.suptitle('Categorical variables plot', fontsize='xx-large')
    save_pth = os.path.join(
        constants.EDA_PLOTS_FOLDER_PATH,
        'categorical_variables_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.EDA_PLOTS_FOLDER_PATH)
        raise err
    logging.info(
        'SUCCESS - saved categorical variables plot at - "%s".',
        save_pth)

    # save histograms for quantitative variables
    fig, axes = plt.subplots(5, 3, figsize=(30, 25))
    for i, col in enumerate(constants.quant_columns):
        row_i = i // 3
        col_i = i % 3
        try:
            df_data[col].hist(figure=fig, bins=40, ax=axes[row_i][col_i])
        except (KeyError, IndexError) as err:
            logging.error(
                ('ERROR - make sure that corresponding quantitative'
                 'columns are present in DataFrame.'))
            raise err
        axes[row_i][col_i].set_title(col, fontdict={'fontsize': 'x-large'})

    fig.suptitle('Quantitative variables plot', fontsize='xx-large')
    save_pth = os.path.join(
        constants.EDA_PLOTS_FOLDER_PATH,
        'quantitative_variables_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.EDA_PLOTS_FOLDER_PATH)
        raise err
    logging.info(
        'SUCCESS - saved quantitative variables plot at path - "%s".',
        save_pth)

    # save distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    plt.figure(figsize=(20, 10))
    plt.title('KDE plot of total transactions')
    try:
        sns.histplot(df_data[constants.KDE_VARIABLE], stat='density', kde=True)
    except KeyError as err:
        logging.error(
            'ERROR - make sure that "%s" quantitative column is present in DataFrame.',
            constants.KDE_VARIABLE)
        raise err

    save_pth = os.path.join(
        constants.EDA_PLOTS_FOLDER_PATH,
        'total_transactions_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.EDA_PLOTS_FOLDER_PATH)
        raise err
    logging.info(
        'SUCCESS - saved KDE plot for total transactions at path - "%s".',
        save_pth)

    # save correlation heatmap for all variables
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_data.corr(), annot=False, cmap='Blues', linewidths=2)
    plt.title('Correlation map')
    save_pth = os.path.join(
        constants.EDA_PLOTS_FOLDER_PATH,
        'correlation_map.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.EDA_PLOTS_FOLDER_PATH)
        raise err
    logging.info('SUCCESS - saved correlation map at path - "%s".', save_pth)


def encoder_helper(df_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df_data: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
        df_cat_encoded: pandas dataframe with new columns for
    '''

    # encoding categorical variables using mean target variables
    df_cat_encoded = pd.DataFrame()
    for col in category_lst:
        try:
            col_groups_map = df_data.groupby(col)[response].mean().to_dict()
            col_name = col + '_' + response
            df_cat_encoded[col_name] = df_data[col].map(col_groups_map)
        except KeyError as err:
            logging.error(
                ('ERROR - make sure that corresponding categorical columns for encoding and '
                 'transoformed target variable - "%s" - are present in DataFrame.'), response)
            raise err

    logging.info(
        'SUCCESS - finished encoding categorical variables by mean target variables.')

    return df_cat_encoded


def perform_feature_engineering(df_data, response):
    '''
    input:
        df_data: pandas dataframe
        response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
        x_train: x training data
        x_test: x testing data
        y_train: y training data
        y_test: y testing data
    '''

    # encoding target variable
    try:
        df_data[response] = df_data[constants.RAW_TARGET_VARIABLE].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    except KeyError as err:
        logging.error(
            'ERROR - make sure that target variable - "%s" - is present in DataFrame.',
            constants.RAW_TARGET_VARIABLE)
        raise err

    # encoding categorical variables using mean target variables
    df_cat_encoded = encoder_helper(
        df_data, constants.to_encode_variables, response)

    # selecting quantitative variables
    try:
        df_quant = df_data[constants.quant_columns]
    except KeyError as err:
        logging.error(
            'ERROR - make sure that correspoding quantitative variables are present in DataFrame.')
        raise err

    # concatenating encoded categorical and quantitative variables
    x_data = pd.concat([df_quant, df_cat_encoded], axis=1)
    y_data = df_data[response]

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)
    logging.info(
        'SUCCESS - successfully transformed and train-test splitted the data.')

    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf
):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''

    # plot random forest classifier report
#     plt.rc('figure', figsize = (5, 5))
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_pth = os.path.join(
        constants.MODEL_RESULT_FOLDER_PATH,
        'random_forest_scores.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.MODEL_RESULT_FOLDER_PATH)
        raise err
    logging.info(
        'SUCCESS - saved random forest classifier report at path - "%s".',
        save_pth)

    # plot logistic regression classifier report
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_pth = os.path.join(
        constants.MODEL_RESULT_FOLDER_PATH,
        'logistic_regression_scores.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that folder exists at path - "%s".',
            constants.MODEL_RESULT_FOLDER_PATH)
        raise err
    logging.info(
        'SUCCESS - saved logistic regression classifier report at path - "%s".',
        save_pth)


def feature_importance_plot(
    model,
    x_data,
    output_pth,
    plot_title='Feature Importance'
):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        output_pth: path to store the figure
        plot_title [optional]: title for the plot, defaults to "Feature Importance"

    output:
        None
    '''

    # Calculate feature importances and sort feature importances in descending
    # order
    try:
        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
        elif isinstance(model, LogisticRegression):
            importances = model.coef_[0]
        else:
            raise TypeError(
                ('"model" object should be trained instance of either '
                 'sklearn.linear_model.LogisticRegression or '
                 'sklearn.ensemble.RandomForestClassifier.'))
        indices = np.argsort(importances)[::-1]
    except AttributeError as err:
        logging.error(
            'ERROR - make sure to path model with "feature_impotances_" attribute.')
        raise err
    # Rearrange feature names so they match the sorted feature importances
    try:
        names = [x_data.columns[i] for i in indices]
    except (AttributeError, IndexError) as err:
        logging.error(
            'ERROR - make sure to pass data corresponding to the trained model.')
        raise err

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title(plot_title)
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    try:
        plt.savefig(output_pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that parent directory for file at path - "%s" - exists.',
            output_pth)
        raise err
    logging.info(
        'SUCCESS - saved feature importance report at path - "%s".',
        output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # random forest classifier
    rfc = RandomForestClassifier(
        random_state=constants.RANDOM_STATE,
        criterion=constants.CRITERION,
        max_depth=constants.MAX_DEPTH,
        max_features=constants.MAX_FEATURES,
        n_estimators=constants.N_ESTIMATORS,
        verbose=0)
    # logistic regression classifier
    lrc = LogisticRegression(
        solver=constants.SOLVER,
        max_iter=constants.MAX_ITER,
        verbose=0
    )

    # train models
    rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # get predictions for random forest classifier
    y_train_preds_rf = rfc.predict(x_train)
    y_test_preds_rf = rfc.predict(x_test)

    # get predictions for logistic regression classifier
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save score reports for both reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    # save feature importances for random forest classifier
    save_rfc_importance_path = os.path.join(
        constants.MODEL_RESULT_FOLDER_PATH,
        'feature_importance_rfc.png')
    feature_importance_plot(
        rfc,
        x_train,
        save_rfc_importance_path,
        plot_title='Feature Importance - Random Forest Classifier')

    save_lrc_importance_path = os.path.join(
        constants.MODEL_RESULT_FOLDER_PATH,
        'feature_importance_lrc.png')
    feature_importance_plot(
        lrc,
        x_train,
        save_lrc_importance_path,
        plot_title='Feature Importance - Logistic Regression Classifier')

    # save best model
    save_rfc_model_path = os.path.join(
        constants.MODEL_SAVE_FOLDER_PATH, 'rfc_model.pkl')
    save_lrc_model_path = os.path.join(
        constants.MODEL_SAVE_FOLDER_PATH, 'lrc_model.pkl')
    try:
        joblib.dump(rfc, save_rfc_model_path)
        joblib.dump(lrc, save_lrc_model_path)
    except FileNotFoundError as err:
        logging.error(
            'ERROR - make sure that parent directory for file at path - "%s" - exists.',
            constants.MODEL_SAVE_FOLDER_PATH)
        raise err

    logging.info(
        'SUCCESS - saved models in folder - %s',
        constants.MODEL_SAVE_FOLDER_PATH)


if __name__ == '__main__':

    df_data_imported = import_data(constants.DATA_FILE_PATH)
    perform_eda(df_data_imported)
    x_train_data, x_test_data, y_train_data, y_test_data = perform_feature_engineering(
        df_data_imported, constants.TRANSFORMED_TARGET_VARIABLE)
    train_models(x_train_data, x_test_data, y_train_data, y_test_data)
