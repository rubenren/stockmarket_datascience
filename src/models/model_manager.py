# Class for loading and training models
import numpy as np
import logging
import tensorflow as tf
from keras import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer


class ModelManager(object):
    """Class for managing models, can train evaluate and predict"""
    def __init__(self, days_ahead=5, logger=None):
        """Init method sets up some internal variables"""
        # Set up Logger for the class
        self.logger = logger or logging.getLogger(__name__)

        self.binary = True
        self.ticker = None
        self.days_ahead = days_ahead
        self.batch_size = 64
        self.model_name = None
        self.model = None

    def create_or_load_model(self, ticker, train, val, nb_epochs=1):
        """Method used to load an existing model, or create a new one on the fly.
        :param ticker: ticker symbol of the stock this model is for
        :param train: numpy array of training data to train on
        :param val: numpy array of validation data to train with
        :param nb_epochs: number of epochs to train with

        .. notes:: The training uses early stopping so the epochs may not match with the number that was actually ran.
        """

        self.ticker = ticker
        self.model_name = ticker + '-' + str(self.days_ahead) + '-day-model.h5'  # set the name for the model
        filepath = Path(__file__).parent / '..' / '..' / 'models'
        if (filepath / self.model_name).exists():  # if we have a model, we load it
            self.logger.info('Found a model, loading that model...')
            self.model = load_model(filepath / self.model_name)
        else:  # otherwise we train a new model
            self.train_model(train, val, nb_epochs)

    def train_model(self, train, val, nb_epochs):
        """Method used to train the model that is currently loaded.
        :param train: numpy array of training data
        :param val: numpy array of validation data
        :param nb_epochs: number of epochs to run the training for

        .. notes:: stores the best model in internal model variable.
        .. notes:: utilizes early stopping, so epoch number may not match the number of runs
        """
        if self.ticker is None:
            self.logger.error('There is no model loaded to train!')
            return

        # helper class for use during training
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.reset_states()
                if not epoch % 10:
                    logging.info('finished epoch %s',epoch)

        # Clear the keras session
        tf.keras.backend.clear_session()

        # fit the data into the batch size
        train_trimmed = train[len(train) % self.batch_size:]
        val_trimmed = val[:-(len(val) % self.batch_size)]\
            if (len(val) % self.batch_size) != 0 else val

        # split off the labels
        x_train, y_train = self.__split_labels(train_trimmed)
        x_val, y_val = self.__split_labels(val_trimmed)

        # reshape for training
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

        # compile our model with the shape of training and save it to class variable
        self.model = self.compile_model(x_train.shape)

        # train our model
        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=nb_epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=[CustomCallback(), EarlyStopping(patience=5, restore_best_weights=True)],
                                 verbose=False,
                                 shuffle=False)

        self.logger.debug(history.history.keys())
        self.logger.info('Finished training, here are the metrics:' +
                         '\n\tacc: %s\n\tprec: %s\n\trecall: %s\n\tauc: %s',
                         history.history['val_accuracy'][-1],
                         history.history['val_precision'][-1],
                         history.history['val_recall'][-1],
                         history.history['val_auc'][-1])

        self.logger.debug('Here are the losses: %s', history.history['val_loss'])
        self.logger.debug('Here are the accuracies: %s', history.history['val_accuracy'])
        self.logger.debug('Here are the precisions: %s', history.history['val_precision'])
        self.logger.debug('Here are the recalls: %s', history.history['val_recall'])
        self.logger.debug('Here are the aucs: %s', history.history['val_auc'])

    def compile_model(self, shape, learning_rate=.000024, num_layers=1, num_nodes=190, dropout_rate=.05):
        """Method to create various structures of LSTM models,

        :param shape: shape of the data to be processed (should be [num_samples, 1, num_variables])
        :param learning_rate: learning rate to be placed in Adam optimizer, default is .000024
        :param num_layers: number of lstm layers to use, lowest is 1, default is 1
        :param num_nodes: number of nodes per layer, defaults to 190
        :param dropout_rate: The dropout rate attached to all LSTM's, defaults to .05

        :return: returns the model that was compiled
        :rtype: Sequential

        .. notes:: these parameters were found with a previous run of Bayesian optimization
        """
        if self.ticker is None:  # if there is no ticker available error out
            self.logger.error('There is no focused stock loaded to compile!')
            return

        model = Sequential()
        for i in range(num_layers - 1):  # add the correct number of layers
            model.add(LSTM(num_nodes,
                           batch_input_shape=(self.batch_size, shape[1], shape[2]),
                           stateful=True,
                           return_sequences=True,  # need to return sequences for passing to next LSTM
                           dropout=dropout_rate))

        # This layer will always be in
        model.add(LSTM(num_nodes,
                       batch_input_shape=(self.batch_size, shape[1], shape[2]),
                       stateful=True,
                       dropout=dropout_rate))

        # if multiclass use 5 outputs for each class, otherwise just binary classification
        if self.binary:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(5, activation='softmax'))

        opt = Adam(learning_rate=learning_rate)

        # metrics we wish to keep track of
        prec = tf.keras.metrics.Precision()
        rec = tf.keras.metrics.Recall()

        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=[prec,
                               rec,
                               'AUC',
                               'accuracy'])

        return model

    def evaluate(self, val, test):
        """Method to evaluate a loaded model
        :param val: validation set for setting state before predictions
        :param test: test set for predicting on

        :return: returns a dictionary of the metrics taken on the evaluation
        """
        if self.model is None:
            self.logger.error('No model to evaluate! skipping evaluation...')
            return

        # fit data into batch size
        val_trimmed = val[len(val) % self.batch_size:]
        test_trimmed = test[:-(len(test) % self.batch_size)] \
            if len(test) % self.batch_size != 0 else test

        # split labels off
        x_val, y_val = self.__split_labels(val_trimmed)
        x_test, y_test = self.__split_labels(test_trimmed)

        # reshape for evaluation
        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        # first run the model on the validation data to set the state
        self.model.predict(x_val, batch_size=self.batch_size, verbose=0)
        # now we evaluate on the test data
        result = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)

        # collect the list of metrics
        out_list = {}
        for name, value in zip(self.model.metrics_names, result):
            out_list[name] = value
            self.logger.info('%s: %s', name, value)

        return out_list

    def __split_labels(self, data):
        """Helper method to split labels off of given data."""
        if self.binary:
            x, y = data[:, :-1], data[:, -1]
        else:
            x, y = data[:, :-5], data[:, -5:]
        return x, y

    def predict(self, train, val, test):
        """Method used to predict on the last day in the test set
        :param train: numpy array of training data
        :param val: numpy array of validation data
        :param test: numpy array of the test data

        :return: returns a float between 0 and 1 that represents its prediction (threshold is usually .5)
        """
        if self.ticker is None:
            self.logger.error('There is no model loaded to train!')
            return
        # first step is to combine all of the data
        combined = np.concatenate([train, val, test], axis=0)
        # then adjust for batch size
        combined = combined[len(combined) % self.batch_size:]
        # now we split up inputs from labels
        x, y = self.__split_labels(combined)
        # and a reshape for fitting into model
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        # now we run the prediction and return the last prediction
        preds = self.model.predict(x, batch_size=self.batch_size)
        return preds[-1]

    def save_model(self):
        """Method to save the currently loaded model into its own file."""
        if self.model is None:
            self.logger.error('There is no model loaded to save!')
            return
        filepath = Path(__file__).parent / '..' / '..' / 'models' / self.model_name
        self.model.save(filepath)

    def train_with_opt(self, train, val, nb_epochs):
        """Method for training with a hyper-parameter optimization process
        :param train: numpy array of the training data
        :param val: numpy array of the validation data
        :param nb_epochs: max number of epochs to use for each iteration

        :return: returns the result of the optimization in a dictionary format

        .. notes:: This method uses a Bayesian optimization from scikit-optimize that optimizes using a gaussian process
        .. notes:: This method also uses early stopping so the number of times it ran may not be the nb_epochs
        """
        if self.ticker is None:
            self.logger.error('No stock selected to train on!')
            return

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.reset_states()
                if not epoch % 10:
                    logging.info('finished epoch %s',epoch)

        best_fscore = -1

        def fitness(x):
            """This method is used by the Bayesian optimization method

            learning_rate: learning rate for optimizer (Adam)
            num_layers: number of LSTM layers to use
            num_nodes: how many nodes per layer
            dropout_rate: dropout rate for the LSTM layers
            """
            # Clear the keras session
            tf.keras.backend.clear_session()

            # Reading in the hyper parameters
            learning_rate = x[0]
            num_layers = x[1]
            num_nodes = x[2]
            dropout_rate = x[3]

            self.logger.info('learning_rate: %s', learning_rate)
            self.logger.info('num_layers: %s', num_layers)
            self.logger.info('num_nodes: %s', num_nodes)
            self.logger.info('dropout_rate: %s', dropout_rate)

            # fit data into batch size
            train_trimmed = train[len(train) % self.batch_size:]
            x_train, y = self.__split_labels(train_trimmed)
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

            # compile the model with the current hyper-parameters
            model = self.compile_model(x_train.shape, learning_rate, num_layers, num_nodes, dropout_rate)

            val_trimmed = val[:-(len(val) % self.batch_size)] if len(val) % self.batch_size != 0 else val
            x_val, y_val = self.__split_labels(val_trimmed)
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

            # use the given number of epochs
            nonlocal nb_epochs

            # train the model with early stopping
            history = model.fit(x_train, y,
                                epochs=nb_epochs,
                                batch_size=self.batch_size,
                                validation_data=(x_val, y_val),
                                callbacks=[CustomCallback(), EarlyStopping(patience=5, restore_best_weights=True)],
                                verbose=False,
                                shuffle=False)

            # get validation data after last epoch
            loss = history.history['val_loss'][-1]
            accuracy = history.history['val_accuracy'][-1]
            prec = history.history['val_precision'][-1]
            rec = history.history['val_recall'][-1]
            auc = history.history['val_auc'][-1]

            # preventing a division by zero
            if prec < .0000001:
                f_beta = 0
            else:
                f_beta = (1.015625 * prec * rec) / (.015625*prec + rec)  # beta of .125

            self.logger.info('val_loss: %.4f', loss)
            self.logger.info('val_accuracy: %.4f', accuracy)
            self.logger.info('val_precision: %.4f', prec)
            self.logger.info('val_recall: %.4f', rec)
            self.logger.info('val_f_beta: %.4f', f_beta)
            self.logger.info('val_auc: %.4f', auc)

            # compare the fscore with the best one seen so far
            nonlocal best_fscore
            if f_beta > best_fscore:
                self.model = model
                self.save_model()  # if we found a better model, save it
                best_fscore = f_beta

            del model

            # Clear the keras session
            tf.keras.backend.clear_session()

            # Because this is a minimizing function, but we wish to maximize we return the negative value
            return -f_beta

        # set up the ranges of values the optimization can use
        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_num_layers = Integer(low=1, high=8, name='num_layers')
        dim_num_nodes = Integer(low=5, high=256, name='num_nodes')
        dim_dropout_rate = Real(low=0.05, high=.8, name='dropout_rate')

        dimensions = [dim_learning_rate,
                      dim_num_layers,
                      dim_num_nodes,
                      dim_dropout_rate]

        # set some defaults, I used a previous iterations numbers for this
        defaults = [.00121, 2, 147, .42]

        # now we run the gaussian process for minimization
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',
                                    n_calls=40,
                                    x0=defaults,
                                    n_jobs=-1)

        # the optimization deletes the models as it goes, so make sure to reload the best model we had saved
        self.create_or_load_model(self.ticker, train, val)

        return search_result
