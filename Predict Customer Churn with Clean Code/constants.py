'''
Module for storing constants.

Author: Dauren Baitursyn
Date: 09.07.22
'''
import os

# file paths
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data/bank_data.csv'   )
LOGS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs/churn_model.logs')

# folder paths
EDA_PLOTS_FOLDER_PATH    = os.path.join(os.path.dirname(__file__), 'images/eda/'    )
MODEL_RESULT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'images/results/')
MODEL_SAVE_FOLDER_PATH   = os.path.join(os.path.dirname(__file__), 'models/'        )

# categorical variables
cat_columns = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
]

# quantitative variables
quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

# variable for KDE plot
kde_variable = 'Total_Trans_Ct'

# variables for encoding
to_encode_variables = [
    'Gender', 
    'Education_Level', 
    'Marital_Status', 
    'Income_Category', 
    'Card_Category'
]

# raw target variable name
raw_target_variable = 'Attrition_Flag'

# transformed target variable name
transformed_target_variable = 'Churn'

# random state variable
random_state = 42

# data split parameters
test_size = 0.3

# logistic regression parameters
solver = 'lbfgs'
max_iter = 3000

# grid search best parameters for random forest classifier
criterion = 'entropy'
max_depth = 100
max_features = 'auto'
n_estimators = 200