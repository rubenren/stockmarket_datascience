import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lsttm model"""

    def __init__(self, filename: str, split: float, cols: list):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.train_data = dataframe.get(cols).values[:i_split]
        self.test_data = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.train_data)
        self.len_test = len(self.test_data)

    def get_test_data(self, seq_len, normalize):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory
        to load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.test_data[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        X = data_windows[:, :-1] # grab all rows and columns except the last columns
        y = data_windows[:, -1, [0]] # grabs just the last columns of all rows and indexes the first in that list
        return X, y

    def get_train_data(self, seq_len, normalize):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure to have enough memory
        to load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalize):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i>= (self.len_train - seq_len):
                    # stop-condition for smaller batch if data doesnt divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalize)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalize):
        '''Generates the next data window from the given index location i'''
        window = self.train_data[i:i+seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalize_windows(self, window_data, single_window = False):
        '''Normalize window with a base value of zero'''
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                if window[0, col_i] == 0:
                    normalized_col = [((float(p) / float(window[1, col_i])) - 1) for p in window[:, col_i]]
                else:
                    normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalized_col)
            normalized_window = np.array(normalized_window).T # reshape and transpose array back to original format
            normalized_data.append(normalized_window)
        return np.array(normalized_data)
