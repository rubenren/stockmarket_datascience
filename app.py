# the streamlit application for a web user interface
import streamlit as st
import pandas as pd
import numpy as np
import logging
import logging.config
import os
import json

from pathlib import Path
from src.data.data_loader import DataLoader
from src.models.model_manager import ModelManager
from src.features.feature_builder import FeatureBuilder


def setup_logging(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)  # load config from file
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()  # First we set up the logging of the system

st.title('5-day Stock Market Prediction')  # Set the title for our web app

# Creating some base classes to manipulate various pieces of our logic
downloader = DataLoader()
transformer = FeatureBuilder()
model_man = ModelManager()

with st.form(key='update_form'):
    # This is the form for adding an arbitrary stock to our list
    ticker = st.text_input('Stock ticker')

    cols = st.columns([1, 2, 9, 1])
    add_ticker = cols[0].form_submit_button("Add")
    remove_ticker = cols[1].form_submit_button("Remove")

    cols[2].write('Will immediately delete any ticker that does not exist!')

    if add_ticker:
        downloader.add_ticker(ticker)
    elif remove_ticker:
        downloader.remove_ticker(ticker)

# The selector for the stock we will be focusing on for our analysis
ticker = st.selectbox("Stock to focus on", downloader.tickers)

downloader.update(ticker)  # Update our database for this specific ticker

cols = st.columns([.7, .825, .9, 5])

btn_train = cols[0].button('Train')
btn_predict = cols[1].button('Predict')
btn_evaluate = cols[2].button('Evaluate')

cols = st.columns(2)

# conditional for running bayesian hyper-parameter optimization with gaussian process
hyper_opt = cols[0].checkbox("Use hyper-parameter optimization (will take a few minutes)")

# grabbing the maximum number of epochs the user would like to run
nb_epochs = cols[1].number_input('Number of epochs to train (uses EarlyStopping)',
                                 min_value=1,
                                 max_value=1000,
                                 value=250)


if btn_train:
    transformer.prep_data(ticker)  # transform and load our data
    train, val, test = transformer.train, transformer.val, transformer.test

    # set the ticker so the model can be named and saved
    model_man.create_or_load_model(ticker, train, val, nb_epochs)

    result = None
    if hyper_opt:
        result = model_man.train_with_opt(train, val, nb_epochs)
    else:
        model_man.train_model(train, val, nb_epochs)
    model_man.save_model()  # save the model we just created for reuse later

    st.success('Created and saved the model!')

if btn_evaluate:
    transformer.prep_data(ticker)  # transform and load our data
    train, val, test = transformer.train, transformer.val, transformer.test

    # loads model for a particular ticker, or trains one and serves it on the fly with default parameters
    model_man.create_or_load_model(ticker, train, val)

    result = model_man.evaluate(val, test)  # get the evaluation metrics to display to the user
    for key in result.keys():
        st.write(key, ': ', result[key])

if btn_predict:
    transformer.prep_data(ticker)  # transform and load our data
    train, val, test = transformer.train, transformer.val, transformer.test

    # loads model for a particular ticker, or trains one and serves it on the fly with default parameters
    model_man.create_or_load_model(ticker, train, val)
    pred = model_man.predict(train, val, test)

    if pred[0] > .5:  # if the prediction is above .5 we predicted positive
        st.write('prediction: %s will go up in 5 days! %s' % (ticker, pred))
    elif pred[0] <= .5:  # if it is below .5 we predicted negative
        st.write('prediction: %s may go down in 5 days! %s' % (ticker, pred))
    st.write('reference date: ', transformer.data.index[-1].date())  # ensure our users know from what date
