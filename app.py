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
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()

st.title('5-day Stock Market Prediction')

downloader = DataLoader()
transformer = FeatureBuilder()
model_man = ModelManager()

with st.form(key='update_form'):
    ticker = st.text_input('Stock ticker')

    cols = st.columns([1,1,10])
    add_ticker = cols[0].form_submit_button("Add")
    remove_ticker = cols[1].form_submit_button("Remove")

    if add_ticker:
        downloader.add_ticker(ticker)
    elif remove_ticker:
        downloader.remove_ticker(ticker)

ticker = st.selectbox("Stock to focus on", downloader.tickers)

downloader.update(ticker)
transformer.prep_data(ticker)

cols = st.columns([.7, .825, .9, 5])

btn_train = cols[0].button('Train')
btn_predict = cols[1].button('Predict')
btn_evaluate = cols[2].button('Evaluate')

cols = st.columns(2)

hyper_opt = cols[0].checkbox("Use hyper-parameter optimization (will take a few minutes)")
nb_epochs = cols[1].number_input('Number of epochs to train', min_value=1, max_value=250)

if btn_train:
    train, val, test = transformer.train, transformer.val, transformer.test
    model_man.create_or_load_model(ticker, train, val, nb_epochs)

    result = None
    if hyper_opt:
        result = model_man.train_with_opt(train, val, nb_epochs)
    else:
        model_man.train_model(train, val, nb_epochs)
    model_man.save_model()

    if result:
        st.write(result)


if btn_evaluate:
    train, val, test = transformer.train, transformer.val, transformer.test

    model_man.create_or_load_model(ticker, train, val)

    result = model_man.evaluate(val, test)
    st.write(result)

if btn_predict:
    train, val, test = transformer.train, transformer.val, transformer.test

    model_man.create_or_load_model(ticker, train, val)
    pred = model_man.predict(train, val, test)

    st.write('prediction: %s' % pred)
    st.write(transformer.data.index[-1].date())
