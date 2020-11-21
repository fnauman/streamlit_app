"""Streamlit app for the Helsingborg project"""

import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_data
from load_model import load_algorithm
from pipe import model_pipeline

from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2
SEED = 123


# Title and description of the app
st.title('Boston housing: Data exploration and model training')

df = load_data()

chosen_cols = st.multiselect("Select columns: ", df.columns)
if chosen_cols:
    st.dataframe(df[chosen_cols])
else:
    st.dataframe(df)

algorithm = st.sidebar.selectbox(
    'Select model',
    ('Linear Regression', 'Lasso', 'Random Forest')
)

X, y = df.drop(columns=['target']), df['target']

st.write(f'Features: {X.shape[1]} with samples: {X.shape[0]}')
st.write(f'Unique target values: {y.nunique()}')


# =========================================================
def sidebar_params(model_name):
    params = dict()

    if model_name == 'Lasso':
        alpha = st.sidebar.slider('alpha', 0.001, 1.0)
        params['alpha'] = alpha
    elif model_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 8, 4, 2, key="initial")
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 10, 100, 20, 10)
        params['n_estimators'] = n_estimators
    else:
        st.sidebar.write("No parameters to tune")
    return params

params = sidebar_params(algorithm)

model = load_algorithm(algorithm, params, SEED)


# =========================================================
# Standard sklearn pipeline
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

numerical_features = X_train.columns.tolist()

pipe = model_pipeline(model, numerical_features)

# Model is not training the pipe directly so transforming the data
X_interp = pipe['preprocessor'].fit_transform(X_train)
X_interp_test = pipe['preprocessor'].fit_transform(X_test)
model.fit(X_interp, y_train)
y_pred = model.predict(X_interp_test)

chart_data = pd.DataFrame({
    'prediction': y_pred, 
    'truth': y_test
})
st.line_chart(chart_data)


st.write(f'Model = {algorithm}')
st.write(f'Score = {model.score(X_interp_test, y_test)}')
