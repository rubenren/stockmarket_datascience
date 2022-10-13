# Data processing class
import os

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _class_separation(x):
    if x < -.05:
        return -2
    elif x < -.005:
        return -1
    elif x > .05:
        return 2
    elif x > .005:
        return 1
    else:
        return 0


def _bin_separation(x):
    if x < 0:
        return 0
    else:
        return 1


def _split_data(data, val_pct=.2, test_pct=.2):
    """
    Splits the data by percentage amount
    Returns: train, val, test
    """
    test_split = int(len(data) * (1 - test_pct))
    val_split = test_split - int(len(data) * val_pct)
    train, val, test = data[:val_split], data[val_split:test_split], data[test_split:]
    return train, val, test


def _scale(in_train, in_val, in_test):
    """Rescales the train, val and test sets
    Args:
        in_train -> numpy array of training data
        in_val -> numpy array of validation set
        in_test -> numpy array of test data
    Returns:
        scaler -> scaler object for transformation
        train_scaled -> rescaled version of training data
        test_scaled -> a rescaled version of the test data
    """
    train = in_train
    val = in_val
    test = in_test
    # scale train and test to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform validation
    val = val.reshape(val.shape[0], val.shape[1])
    val_scaled = scaler.transform(val)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, val_scaled, test_scaled


class FeatureBuilder(object):
    """Object that can be used to transform our data"""
    def __init__(self, logger=None):
        # Set up Logger for the class
        self.logger = logger or logging.getLogger(__name__)

        self.train = None
        self.val = None
        self.test = None
        self.transformed = None
        self.data = None

    def __calc_rsi(self, periods=14, ema=True):
        close_delta = self.data.adj_close.diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        return rsi

    def read_data(self, ticker):
        # read in the data
        up_ticker = ticker.upper()
        filename = None
        filepath = os.path.dirname(__file__)
        print(filepath)
        path = os.path.join(filepath, '..', '..', 'data', 'raw')
        for file in os.listdir(path):
            if file.split('_')[0] == up_ticker:
                filename = file
        filename = os.path.join(path, filename)
        self.data = pd.read_csv(filename)

        # adjust the column names so they are lowercase w/ no spaces
        self.data = self.data.rename(str.lower, axis=1)
        self.data = self.data.rename(mapper={'adj close': 'adj_close'}, axis=1)

        # set the index to be in datetime
        self.data.date = pd.to_datetime(self.data.date)
        self.data = self.data.set_index('date')

    def add_features(self):
        if self.data is None:
            self.logger.error('FeatureBuilder has no data to transform! Skipping add')
            return

        # accumulation/distribution line calculations
        mult = ((self.data.close - self.data.low) - (self.data.high - self.data.close)) / (self.data.high - self.data.low)
        MFVolume = mult * self.data.volume
        accum_dist_indicator = MFVolume.cumsum()
        ret_df = pd.concat([self.data, accum_dist_indicator], axis=1)
        ret_df = ret_df.rename(mapper={0: 'accum_dist_indicator'}, axis=1)

        # MACD
        ema_12 = self.data.adj_close.ewm(span=12, adjust=False).mean()
        ema_26 = self.data.adj_close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        ret_df = pd.concat([ret_df, macd.rename('macd'), signal.rename('signal_macd')], axis=1)

        # RSI
        rsi = self.__calc_rsi()
        ret_df = pd.concat([ret_df, rsi.rename('rsi')], axis=1)

        self.transformed = ret_df

    def __series_to_supervised(self, n_in=1, n_out=1, preds=[], dropnan=True):
        """Convert a time series to a supervised learning dataset
        Args:
            n_in -> number of lag observations as input (X)
            n_out -> number of observations as output (y)
            col_names -> names of the columns
            indicies -> list of the indicies
            preds -> list of column indicies to determine which variables to predict
            dropnan -> flag of whether to drop the rows with NaN
        Returns:
            Pandas DataFrame of series framed for supervised learning
        """
        col_names = self.transformed.columns
        indicies = self.transformed.index
        n_vars = 1 if type(self.transformed) is list else self.transformed.shape[1]
        df = pd.DataFrame(self.transformed)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (col_names[j], i)) for j in range(n_vars)]
        # forecast sequence
        for i in range(0, n_out):
            cols.append(df[col_names[preds]].shift(-i))  # Only process the variables we are predicting
            if i == 0:
                names += [('%s(t)' % (col_names[j])) for j in preds]
            else:
                names += [('%s(t+%d)' % (col_names[j], i)) for j in preds]
        # putting it together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.index = indicies
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def make_digestible(self, look_back=14, pred_day=5, binary_classify=True):
        """transforms the data into a dataset that can be trained with, stored in train, val, and test
        Args:
            look_back -> number of days of history to look at
            pred_day -> number of days after today to predict direction
            binary_classify -> whether to use binary classification or multiclass
        Returns:
            scaler -> scaler for transformation
        """
        self.logger.info('Beginning splitting, labeling, and rescaling of data')
        temp = np.array([0, 1, 2, -1, -2]).reshape(-1, 1)
        if self.data is None:
            self.logger.error('No data has been read in. skipping transform')
            return
        if self.transformed is None:
            self.logger.warning('data has not been augmented yet! augmenting before moving on...')
            self.add_features()
        # frame as an RNN problem
        data = self.__series_to_supervised(look_back, pred_day, [4])
        # only keep the predictive columns I care about
        data = data.drop(data.columns[-pred_day:-1], axis=1)
        labels = (data['adj_close(t+4)'] - data['adj_close(t-1)']) / data['adj_close(t-1)']
        if binary_classify:
            labels = labels.apply(_bin_separation)
        else:
            labels = labels.apply(_class_separation)
        data = data.drop(data.columns[-1], axis=1)
        data = pd.concat([data, labels.rename('labels')], axis=1)
        data_values = data.values
        train, val, test = _split_data(data_values, .2, .2)
        scaler, train_scaled, val_scaled, test_scaled = _scale(train[:, :-1], val[:, :-1], test[:, :-1])
        if binary_classify:
            train_labels = train[:, -1].reshape((-1, 1))
            val_labels = val[:, -1].reshape((-1, 1))
            test_labels = test[:, -1].reshape((-1, 1))
        else:
            ohe = OneHotEncoder(sparse=False).fit(temp)
            train_labels = ohe.transform(train[:, -1].reshape((-1, 1)))
            val_labels = ohe.transform(val[:, -1].reshape((-1, 1)))
            test_labels = ohe.transform(test[:, -1].reshape((-1, 1)))
        self.train = np.append(train_scaled, train_labels, axis=1)
        self.val = np.append(val_scaled, val_labels, axis=1)
        self.test = np.append(test_scaled, test_labels, axis=1)
        self.logger.info('Finished splitting, labeling, and rescaling')
        self.logger.debug('lengths: %s, %s, %s', self.train.shape, self.val.shape, self.test.shape)
        return scaler

    def save_features(self, ticker):
        if self.transformed is None:
            self.logger.error('No data to save!')
            return
        filepath = os.path.dirname(__file__)
        filepath = os.path.join(filepath, '..', '..', 'data', 'interim', ticker + '.csv')
        self.transformed.to_csv(filepath)
