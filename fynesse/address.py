# This file contains code for suporting addressing questions in the data
import random

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import math
from .utils import aws_utils, pandas_utils, plot_utils
from . import access

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""


def train_model(x, y):  # note: y is 1 must be 1 dimensional
    # create model
    m_linear = sm.GLM(y, x, family=sm.families.Gaussian())
    m_gaussian_results = m_linear.fit()

    return m_gaussian_results


def predict(model, x_pred):
    y_pred = model.get_prediction(x_pred).summary_frame(alpha=0.05)
    return y_pred


def plot_against_data(model, x_pred, y_actual, title):
    y_pred = predict(model, x_pred)

    plt.scatter(x_pred,y_actual,color='blue')

    plt.plot(x_pred,y_pred['mean'],color='red',zorder=2)
    plt.plot(x_pred,y_pred['mean_ci_lower'], color='red',linestyle='dotted',zorder=2)
    plt.plot(x_pred,y_pred['mean_ci_upper'], color='red',linestyle='dotted',zorder=2)

    plt.title(title)

    plt.tight_layout()

    print(model.summary())


def plot_correlation(model, x_pred, y_actual):
    y_pred = predict(model, x_pred)
    y_pred_mean = y_pred['mean']
    correlation = np.corrcoef(y_actual, y_pred_mean)[0, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred_mean, alpha=0.7, label=f'Correlation: {correlation:.2f}', color='blue')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='red', linestyle='--', label='y = y_pred')

    plt.xlabel('Actual Values (y_actual)')
    plt.ylabel('Predicted Values y_pred)')
    plt.title('Correlation Between y_actual and y_pred')

    plt.legend()
    plt.show()


def get_correlation_regularised(x_df, y_df, alpha, L1_weight):
    # extract data
    x = x_df.values
    y = y_df.values.flatten()

    # create model
    m_linear = sm.OLS(y, x)
    model = m_linear.fit_regularized(alpha=alpha, L1_wt=L1_weight)

    # prediction
    x_pred = x
    y_actual = y

    y_pred = model.predict(x_pred)
    correlation = np.corrcoef(y_actual, y_pred)[0, 1]

    return correlation, model


def train_regularised_model(input_df, output_df, start_alpha=0.001, start_l1_weight=0.001, max_steps=5):
    # note: 0.001 seemed to work geat for both values, if want to save time, should set max_steps to 1

    alpha = start_alpha
    L1_weight = start_l1_weight

    correlations = {}
    correlations[(alpha, L1_weight)], _ = get_correlation_regularised(input_df, output_df, alpha, L1_weight)

    neighbours = [0.5, 1, 2]

    for _ in range(max_steps):
        best = -math.inf
        next_alpha, next_L1 = alpha, L1_weight
        for mult_alpha in neighbours:
            for mult_L1 in neighbours:
                if (alpha*mult_alpha, L1_weight*mult_L1) not in correlations:
                    correlations[(alpha*mult_alpha, L1_weight*mult_L1)], _ = get_correlation_regularised(input_df, output_df, alpha*mult_alpha, L1_weight*mult_L1)
                  
                if correlations[(alpha*mult_alpha, L1_weight*mult_L1)] > best: 
                    next_alpha, next_L1 = alpha*mult_alpha, L1_weight*mult_L1
                    best = correlations[(alpha*mult_alpha, L1_weight*mult_L1)]

        if (next_alpha, next_L1) == (alpha, L1_weight):
            break
        
        alpha, L1_weight = next_alpha, next_L1
        
    model = get_correlation_regularised(input_df, output_df, alpha, L1_weight)

    return model


def get_model(conn, input_table_name, input_columns, output_table_name, output_column, max_steps=5):
    input_df = aws_utils.query_AWS_load_table(conn, input_table_name, ["OA"] + input_columns)
    output_df = aws_utils.query_AWS_load_table(conn, output_table_name, ["OA"] + output_column)

    joined_df = input_df.merge(output_df, how="inner", on=["OA"])

    input_df = joined_df[input_columns]
    output_df = joined_df[output_column]

    return train_regularised_model(input_df, output_df, max_steps=max_steps)

def transport_model_1(conn, max_steps=5):
    tables = access.get_census_data_column_names()
    columns = ["density"] + tables["TS007"] + tables["TS038"] + tables["TS058"] + tables["TS061"]

    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    feature_df = census_df[list(set(census_df[["density"] + tables["TS007"] + tables["TS038"] + tables["TS058"]]).difference({"TS058_home_office"}))]

    return {transport_method: train_regularised_model(feature_df, census_df[[transport_method]], max_steps=max_steps) for transport_method in tables["TS061"]}


def transport_model_2(conn, extra_census_features, max_steps=5):
    tables = access.get_census_data_column_names()
    extra_features = []
    for _, column_names in extra_census_features.items():
        extra_features += column_names
    extra_features = list(set(extra_features))

    columns = list(set(["density"] + tables["TS007"] + tables["TS038"] + tables["TS058"] + tables["TS061"] + extra_features))

    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    feature_dfs = {transport_method: census_df[list(set(["density"] + tables["TS007"] + tables["TS038"] + tables["TS058"] + extra_census_features[transport_method]).difference({"TS058_home_office"}))] for transport_method in tables["TS061"]}

    return {transport_method: train_regularised_model(feature_dfs[transport_method], census_df[[transport_method]], max_steps=max_steps) for transport_method in tables["TS061"]}


def transport_model_3(conn, extra_census_features, extra_osm_features, max_steps=5):
    tables = access.get_census_data_column_names()
    extra_features = []
    for _, column_names in extra_census_features.items():
        extra_features += column_names
    extra_features = list(set(extra_features))

    columns = list(set(["OA", "density"] + tables["TS007"] + tables["TS038"] + tables["TS058"] + tables["TS061"] + extra_features))
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    osm_features = []
    for _, column_names in extra_osm_features.items():
        osm_features += column_names
    osm_features = list(set(osm_features))

    columns = list(set(["OA"] + osm_features))
    osm_df = aws_utils.query_AWS_load_table(conn, "nearby_amenity_non_transport", columns)

    joined_df = census_df.merge(osm_df, how="inner", on=["OA"])

    feature_dfs = {transport_method: joined_df[list(set(["density"] + tables["TS007"] + tables["TS038"] + tables["TS058"] + extra_census_features[transport_method] + extra_osm_features[transport_method]).difference({"TS058_home_office"}))] for transport_method in tables["TS061"]}

    return {transport_method: train_regularised_model(feature_dfs[transport_method], joined_df[[transport_method]], max_steps=max_steps) for transport_method in tables["TS061"]}


def get_difference_in_transport(models, feature_df, transport_df):
    difference_df = pd.DataFrame
    for transport_method in models:
        difference_df[transport_method] = transport_df.values - predict(models[transport_method], feature_df)

    return difference_df