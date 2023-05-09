# Telecom Customer Churn Prediction - README

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [Model Comparison and Analysis](#model-comparison-and-analysis)
5. [Data Summary and Implications](#data-summary-and-implications)
6. [Recommendations](#recommendations)
7. [Demonstration](#demonstration)

## Introduction

This project aims to predict customer churn for a telecom company using logistic regression models. We clean and transform the data, create an initial logistic regression model, perform step-forward feature selection, and finally create a reduced logistic regression model for better interpretability.

## Dataset

The dataset contains information about customer demographics, services, and churn rates. It has 7,043 rows and 21 columns, including the target variable 'Churn'.

## Data Preparation

The data preparation process involves encoding binary nominal variables using label encoder from Sklearn, ordinal variables with their respective rank, and creating additional columns to describe the type of internet service customers were subscribed to. After data preparation, the clean dataset is saved as 'churn_logistic_regression.csv'.

## Model Comparison and Analysis

We create an initial logistic regression model and then implement step-forward feature selection to reduce the number of variables in the model. We test the selected features for multicollinearity using the variance inflation factor formula from Sklearn and remove variables with high multicollinearity. The final reduced logistic regression model uses six independent variables and has a comparable F1 and accuracy score to the initial model.

## Data Summary and Implications

Our reduced logistic regression model shows that as tenure, options, and timely replacements increase, a customer's likelihood to churn decreases. Inversely, as a customer's monthly charges, age, and the number of times they contact technical support increase, so does their likelihood of churn. A telecom company can use these insights to make informed decisions on marketing strategies and customer incentives.

## Recommendations

We recommend that the telecom provider focus on providing more service options, ensuring efficient replacements, and reviewing their pricing strategies. They should also focus on customer retention, especially considering the impact of customer age and tenure on churn. The model can be used to predict churn for other customers with similar characteristics, informing future marketing, customer service training, and pricing strategies.
