# Class for loading and training models
import numpy as np
import logging
import tensorflow as tf
from keras import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.features.feature_builder import FeatureBuilder
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer


class ModelManager(object):
    def __init__(self, days_ahead=5, logger=None):
        # Set up Logger for the class
        self.logger = logger or logging.getLogger(__name__)

        self.binary = True
        self.ticker = None
        self.days_ahead = days_ahead
        self.batch_size = 64
        self.model_name = None
        self.model = None

    def create_or_load_model(self, ticker, train, val, nb_epochs=1):
        self.ticker = ticker
        self.model_name = ticker + '-' + str(self.days_ahead) + '-day-model.h5'
        filepath = Path(__file__).parent / '..' / '..' / 'models'
        if (filepath / self.model_name).exists():  # if we have a model, we load it
            self.logger.info('Found a model, loading that model...')
            self.model = load_model(filepath / self.model_name)
        else:  # otherwise we train a new model
            self.train_model(train, val, nb_epochs)

    def train_model(self, train, val, nb_epochs):

        if self.ticker is None:
            self.logger.error('There is no model loaded to train!')
            return

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.reset_states()
                if not epoch % 10:
                    logging.info('finished epoch %s',epoch)

        # Clear the keras session
        tf.keras.backend.clear_session()

        train_trimmed = train[len(train) % self.batch_size:]
        val_trimmed = val[:-(len(val) % self.batch_size)]\
            if (len(val) % self.batch_size) != 0 else val

        x_train, y_train = self.__split_labels(train_trimmed)
        x_val, y_val = self.__split_labels(val_trimmed)

        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
        self.model = self.compile_model(x_train.shape)

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=nb_epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=[CustomCallback(), EarlyStopping(patience=5, restore_best_weights=True)],
                                 verbose=False,
                                 shuffle=False)

        self.logger.info(history.history.keys())
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

        if self.ticker is None:
            self.logger.error('There is no focused stock loaded to compile!')
            return

        model = Sequential()
        for i in range(num_layers - 1):
            model.add(LSTM(num_nodes,
                           batch_input_shape=(self.batch_size, shape[1], shape[2]),
                           stateful=True,
                           return_sequences=True,
                           dropout=dropout_rate))

        model.add(LSTM(num_nodes,
                       batch_input_shape=(self.batch_size, shape[1], shape[2]),
                       stateful=True,
                       dropout=dropout_rate))

        if self.binary:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(5, activation='softmax'))

        opt = Adam(learning_rate=learning_rate)

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
        if self.model is None:
            self.logger.error('No model to evaluate! skipping evaluation...')
            return

        val_trimmed = val[len(val) % self.batch_size:]
        test_trimmed = test[:-(len(test) % self.batch_size)] \
            if len(test) % self.batch_size != 0 else test

        x_val, y_val = self.__split_labels(val_trimmed)
        x_test, y_test = self.__split_labels(test_trimmed)

        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        # First run the model on the validation data to set the state
        self.model.predict(x_val, batch_size=self.batch_size, verbose=0)
        # now we evaluate on the test data
        result = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)

        out_list = {}
        for name, value in zip(self.model.metrics_names, result):
            out_list[name] =  value
            self.logger.info('%s: %s', name, value)

        return out_list

    def __split_labels(self, data):
        if self.binary:
            x, y = data[:, :-1], data[:, -1]
        else:
            x, y = data[:, :-5], data[:, -5:]
        return x, y

    def predict(self, train, val, test):
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
        if self.model is None:
            self.logger.error('There is no model loaded to save!')
            return
        filepath = Path(__file__).parent / '..' / '..' / 'models' / self.model_name
        self.model.save(filepath)

    def train_with_opt(self, train, val, nb_epochs):  # TODO: implement optimizing for hyperparameters

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
            """
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

            train_trimmed = train[len(train) % self.batch_size:]
            x_train, y = self.__split_labels(train_trimmed)
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            model = self.compile_model(x_train.shape, learning_rate, num_layers, num_nodes, dropout_rate)

            val_trimmed = val[:-(len(val) % self.batch_size)] if len(val) % self.batch_size != 0 else val
            x_val, y_val = self.__split_labels(val_trimmed)
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

            nonlocal nb_epochs

            history = model.fit(x_train, y,
                                epochs=nb_epochs,
                                batch_size=self.batch_size,
                                validation_data=(x_val, y_val),
                                callbacks=[CustomCallback(), EarlyStopping(patience=5, restore_best_weights=True)],
                                verbose=False,
                                shuffle=False)

            # get validation loss after last epoch
            loss = history.history['val_loss'][-1]
            accuracy = history.history['val_accuracy'][-1]
            prec = history.history['val_precision'][-1]
            rec = history.history['val_recall'][-1]
            auc = history.history['val_auc'][-1]

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

            nonlocal best_fscore
            if f_beta > best_fscore:
                self.model = model
                self.save_model()
                best_fscore = f_beta

            del model

            # Clear the keras session
            tf.keras.backend.clear_session()

            # Because this is a minimizing function, but we wish to maximize we return the negative value
            return -f_beta

        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_num_layers = Integer(low=1, high=8, name='num_layers')
        dim_num_nodes = Integer(low=5, high=256, name='num_nodes')
        dim_dropout_rate = Real(low=0.05, high=.8, name='dropout_rate')

        dimensions = [dim_learning_rate,
                      dim_num_layers,
                      dim_num_nodes,
                      dim_dropout_rate]
        defaults = [.00121, 2, 147, .42]

        # now we run the gaussian process for minimization
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',
                                    n_calls=40,
                                    x0=defaults,
                                    n_jobs=-1)

        self.create_or_load_model(self.ticker, train, val)

        return search_result
