# This file contains code for suporting addressing questions in the data
import random

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

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


def train_model(x, y):
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

