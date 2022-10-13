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


class ModelManager(object):
    def __init__(self, ticker, days_ahead=5, logger=None):
        # Set up Logger for the class
        self.logger = logger or logging.getLogger(__name__)

        self.binary = True
        self.ticker = ticker.upper()
        self.days_ahead = days_ahead
        self.batch_size = 64
        self.model_name = ticker + '-' + str(days_ahead) + '-day-model.h5'
        self.model = None
        self.processor = FeatureBuilder(self.logger)

    def create_or_load_model(self):
        filepath = Path(__file__).parent / '..' / '..' / 'models'
        self.processor.read_data(self.ticker)
        self.processor.add_features()
        self.processor.save_features(self.ticker)
        self.processor.make_digestible(pred_day=self.days_ahead)
        if (filepath / self.model_name).exists():  # if we have a model, we load it
            self.logger.info('Found a model, loading that model...')
            self.model = load_model(filepath / self.model_name)
        else:  # otherwise we train a new model
            self.train_model()

    def train_model(self):

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.reset_states()

        if self.model is not None:
            self.logger.info('already have a model, call retrain to train the model')
            return

        train_trimmed = self.processor.train[len(self.processor.train) % self.batch_size:]
        val_trimmed = self.processor.val[:-(len(self.processor.val) % self.batch_size)]\
            if len(self.processor.val) % self.batch_size != 0 else self.processor.val
        if self.binary:
            x_train, y_train = train_trimmed[:, :-1], train_trimmed[:, -1]
            x_val, y_val = val_trimmed[:, :-1], val_trimmed[:, -1]
        else:
            x_train, y_train = train_trimmed[:, :-5], train_trimmed[:, -5:]
            x_val, y_val = val_trimmed[:, :-5], val_trimmed[:, -5:]

        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
        self.model = self.compile_model(x_train.shape)

        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=self.batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=[CustomCallback(), EarlyStopping(patience=5, restore_best_weights=True)],
                                 verbose=False,
                                 shuffle=False)

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

        model = Sequential()
        for i in range(num_layers - 1):
            model.add(LSTM(num_nodes,
                           batch_input_shape=(self.batch_size, shape[1], shape[2]),
                           stateful=True,
                           return_sequences=True,
                           dropout=.2))

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

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[prec,
                               rec,
                               'AUC',
                               'accuracy'])

        return model

    def evaluate(self):
        if self.model is None:
            self.logger.error('No model to evaluate! skipping evaluation...')
            return

        val_trimmed = self.processor.val[len(self.processor.val) % self.batch_size:]
        test_trimmed = self.processor.test[:-(len(self.processor.test) % self.batch_size)] \
            if len(self.processor.test) % self.batch_size != 0 else self.processor.test

        x_val, y_val = self.__split_labels(val_trimmed)
        x_test, y_test = self.__split_labels(test_trimmed)

        x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        # First run the model on the validation data to set the state
        self.model.predict(x_val, batch_size=self.batch_size, verbose=0)
        # now we evaluate on the test data
        result = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)

        out_list = []
        for name, value in zip(self.model.metrics_names, result):
            out_list.append((name, value))
            self.logger.info('%s: %s', name, value)

        return out_list

    def __split_labels(self, data):
        if self.binary:
            x, y = data[:, :-1], data[:, -1]
        else:
            x, y = data[:, :-5], data[:, -5:]
        return x, y

    def predict(self):
        # first step is to combine all of the data
        combined = np.concatenate([self.processor.train, self.processor.val, self.processor.test], axis=0)
        # then adjust for batch size
        combined = combined[len(combined) % self.batch_size:]
        # now we split up inputs from labels
        x, y = self.__split_labels(combined)
        # and a reshape for fitting into model
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        # now we run the prediction and return the last prediction
        preds = self.model.predict(x, batch_size=self.batch_size)
        thresh = preds.mean()
        preds = preds > thresh
        return preds[-1], thresh

    def predict_date(self, date):
        pass

    def save_model(self):
        if self.model is None:
            self.logger.error('There is no model loaded to save!')
            return
        filepath = Path(__file__).parent / '..' / '..' / 'models' / self.model_name
        self.model.save(filepath)

    def optimize_hypers(self):  # TODO: implement optimizing for hyperparameters
        pass
