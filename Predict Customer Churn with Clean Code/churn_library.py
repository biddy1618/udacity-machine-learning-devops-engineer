'''
Module for finding customer who are likely to churn.

Author: Dauren Baitursyn
Date: 08.07.22
'''

# import libraries
import constants

import logging
import sys
import os

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'

# configuration for logging
logging.basicConfig(
    stream   = sys.stdout,
    level    = logging.INFO,
    filemode ='w',
    format   = '%(name)s - %(levelname)s - %(message)s'
)

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    
    '''
    High-level pseudocode:
    - Check if it's been read by Pandas
    - handle FileNotFoundError
    - handle ParserError
    - return DataFrame
    '''
    
    # import file as DataFrame object
    df = pd.DataFrame()
    try:
        df = pd.read_csv(pth, index_col = 0)
    except FileNotFoundError as e:
        logging.error('ERROR - File not found at specified path - "%s".', pth)
        raise e
    except pd.errors.ParserError as e:
        logging.error('ERROR - Make sure that file at path - ""%s" - is in CSV format.', pth)
        raise e
    
    logging.info('SUCCESS - reading CSV file at path - "%s".', pth)
    
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    
    '''
    High-level pseudocode:
    - save categorical variables bar plot
    - save quantitative variables histogram plot
    - save KDE plot of total transactions
    - save correlation heatmap plot
    - handle KeyError
    '''
    
    # save barplots for categorical variables
    fig, axes = plt.subplots(2, 3, figsize = (30, 10))
    for i, col in enumerate(constants.cat_columns):
        r = i//3
        c = i%3
        try:
            df[col].value_counts('normalize').plot.bar(figure = fig, ax = axes[r][c])
        except (KeyError, IndexError) as e:
            logging.error('ERROR - make sure that corresponding categorical columns are present in DataFrame.')
            raise e
        axes[r][c].set_title(col, fontdict = {'fontsize': 'x-large'})
        axes[r][c].tick_params(axis='x', rotation=0)

    fig.suptitle('Categorical variables plot', fontsize = 'xx-large')
    save_pth = os.path.join(constants.EDA_PLOTS_FOLDER_PATH, 'categorical_variables_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.EDA_PLOTS_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved categorical variables plot at - "%s".', save_pth)

    
    # save histograms for quantitative variables
    fig, axes = plt.subplots(5, 3, figsize = (30, 25))
    for i, col in enumerate(constants.quant_columns):
        r = i//3
        c = i%3
        try:
            df[col].hist(figure = fig, bins = 40, ax = axes[r][c])
        except (KeyError, IndexError) as e:
            logging.error('ERROR - make sure that corresponding quantitative columns are present in DataFrame.')
            raise e
        axes[r][c].set_title(col, fontdict = {'fontsize': 'x-large'})

    fig.suptitle('Quantitative variables plot', fontsize = 'xx-large')
    save_pth = os.path.join(constants.EDA_PLOTS_FOLDER_PATH, 'quantitative_variables_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.EDA_PLOTS_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved quantitative variables plot at path - "%s".', save_pth)
                
    
    # save distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    plt.figure(figsize = (20, 10)) 
    plt.title('KDE plot of total transactions')
    try:
        sns.histplot(df[constants.kde_variable], stat = 'density', kde = True)
    except KeyError as e:
        logging.error('ERROR - make sure that "%s" quantitative column is present in DataFrame.', constants.kde_variable)
    
    save_pth = os.path.join(constants.EDA_PLOTS_FOLDER_PATH, 'total_transactions_plot.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.EDA_PLOTS_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved KDE plot for total transactions at path - "%s".', save_pth)
    
    
    # save correlation heatmap for all variables 
    plt.figure(figsize = (20, 10)) 
    sns.heatmap(df.corr(), annot = False, cmap = 'Blues', linewidths = 2)
    plt.title('Correlation map')
    save_pth = os.path.join(constants.EDA_PLOTS_FOLDER_PATH, 'correlation_map.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.EDA_PLOTS_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved correlation map at path - "%s".', save_pth)

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    '''
    High-level pseudocode:
    - for given categorical variable to encode
        - create new columns based on the mean of the target variable
    - handle KeyError
    '''
    
    # encoding categorical variables using mean target variables
    df_cat_encoded = pd.DataFrame()
    for col in category_lst:
        col_lst = []
        try:
            col_groups_map = df.groupby(col)[response].mean().to_dict()
            col_name = col + '_' + response
            df_cat_encoded[col_name] = df[col].map(col_groups_map)
        except KeyError as e:
            logging.error('ERROR - make sure that corresponding categorical columns for encoding and transoformed target variable - "%s" - are present in DataFrame.', response)
            raise e
    
    logging.info('SUCCESS - finished encoding categorical variables by mean target variables.')
    
    return df_cat_encoded


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    '''
    High-level pseudocode:
    - encode target variable
    - encode categorical variables
    - concatenate quantitative variables and categorical variables
    - train-test split independent and target variables
    - handle KeyError(s)
    '''
    
    # encoding target variable
    try:
        df[response] = df[constants.raw_target_variable].apply(lambda val: 0 if val == "Existing Customer" else 1)
    except KeyError as e:
        logging.error('ERROR - make sure that target variable - "%s" - is present in DataFrame.', constants.raw_target_variable)
        raise e
    
    
    # encoding categorical variables using mean target variables
    df_cat_encoded = encoder_helper(df, constants.to_encode_variables, response)
    
    
    # selecting quantitative variables
    try:
        df_quant = df[constants.quant_columns]
    except KeyError as e:
        logging.error('ERROR - make sure that correspoding quantitative variables are present in DataFrame.')
        raise e
    
    
    # concatenating encoded categorical and quantitative variables
    X = pd.concat([df_quant, df_cat_encoded], axis = 1)
    y = df[response]
    
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = constants.test_size, random_state = constants.random_state)
    logging.info('SUCCESS - successfully transformed and train-test splitted the data.')
    
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
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
    
    '''
    High-level pseudocode:
    - plot classification report for random forest classifier scores
    - plot classification report for logistic regression classifier scores
    - handle FileNotFoundError(s)
    '''
    
    # plot random forest classifier report
#     plt.rc('figure', figsize = (5, 5))
    plt.figure(figsize = (5, 5)) 
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    save_pth = os.path.join(constants.MODEL_RESULT_FOLDER_PATH, 'random_forest_scores.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.MODEL_RESULT_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved random forest classifier report at path - "%s".', save_pth)
    
    
    # plot logistic regression classifier report
#     plt.rc('figure', figsize = (5, 5))
    plt.figure(figsize = (5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    save_pth = os.path.join(constants.MODEL_RESULT_FOLDER_PATH, 'logistic_regression_scores.png')
    try:
        plt.savefig(save_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that folder exists at path - "%s".', constants.MODEL_RESULT_FOLDER_PATH)
        raise e
    logging.info('SUCCESS - saved logistic regression classifier report at path - "%s".', save_pth)


def feature_importance_plot(model, X_data, output_pth, plot_title = 'Feature Importance'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    
    '''
    High-level pseudocode:
    - map data variable names to feature importances of model
    - save the plot
    - handle AttributeError, IndexError, and FileNotFoundError
    '''
    
    # Calculate feature importances and sort feature importances in descending order
    try:
        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
        elif isinstance(model, LogisticRegression):
            importances = model.coef_[0]
        else:
            raise TypeError('"model" object should be trained instance of either sklearn.linear_model.LogisticRegression or sklearn.ensemble.RandomForestClassifier.')
        indices = np.argsort(importances)[::-1]
    except AttributeError as e:
        logging.error('ERROR - make sure to path model with "feature_impotances_" attribute.')
        raise e
    # Rearrange feature names so they match the sorted feature importances
    try:
        names = [X_data.columns[i] for i in indices]
    except (AttributeError, IndexError) as e:
        logging.error('ERROR - make sure to pass data corresponding to the trained model.')
        raise e

    
    # Create plot
    plt.figure(figsize = (20, 5))
    plt.title(plot_title)
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    try:
        plt.savefig(output_pth)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that parent directory for file at path - "%s" - exists.', output_pth)
        raise e
    logging.info('SUCCESS - saved feature importance report at path - "%s".', output_pth)
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    
    '''
    High-level pseudocode:
    - define models with parameters
    - train models
    - save model score reports
    - save model feature importances
    - save models
    '''
    # random forest classifier
    rfc = RandomForestClassifier(
        random_state = constants.random_state,
        criterion = constants.criterion,
        max_depth = constants.max_depth,
        max_features = constants.max_features,
        n_estimators = constants.n_estimators
    )
    # logistic regression classifier 
    lrc = LogisticRegression(
        solver = constants.solver,
        max_iter = constants.max_iter
    )
    
    # train models
    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    
    # get predictions for random forest classifier
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)
    
    # get predictions for logistic regression classifier
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
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
    save_rfc_importance_path = os.path.join(constants.MODEL_RESULT_FOLDER_PATH, 'feature_importance_rfc.png')
    feature_importance_plot(rfc, X_train, save_rfc_importance_path, plot_title = 'Feature Importance - Random Forest Classifier')

    save_lrc_importance_path = os.path.join(constants.MODEL_RESULT_FOLDER_PATH, 'feature_importance_lrc.png')
    feature_importance_plot(lrc, X_train, save_lrc_importance_path, plot_title = 'Feature Importance - Logistic Regression Classifier')
    
    # save best model
    save_rfc_model_path = os.path.join(constants.MODEL_SAVE_FOLDER_PATH, 'rfc_model.pkl')
    save_lrc_model_path = os.path.join(constants.MODEL_SAVE_FOLDER_PATH, 'lrc_model.pkl')
    try:
        joblib.dump(rfc, save_rfc_model_path)
        joblib.dump(lrc, save_lrc_model_path)
    except FileNotFoundError as e:
        logging.error('ERROR - make sure that parent directory for file at path - "%s" - exists.', output_pth)
        raise e
    
    logging.info('SUCCESS - saved models in folder - %s', constants.MODEL_SAVE_FOLDER_PATH)