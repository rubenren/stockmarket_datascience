# class for downloading the stock data

import datetime
import yfinance as yf
import logging
import json
from pathlib import Path


def _download_stock(symbol):
    # Using yahoo finance API to grab stock market data
    data = yf.download(symbol)
    if len(data) == 0:
        logging.error('Could not find the stock ticker %s, skipping download' % symbol)
        return
    filename = symbol + '_' + str(data.index[0])[:10] + '_' + str(data.index[-1])[:10] + '.csv'
    path_var = Path(__file__).parent / '..' / '..' / 'data' / 'raw'
    filepath = path_var / filename
    data.to_csv(filepath)


class DataLoader(object):
    """DataLoader object used for downloading stockmarket data from file tickers.txt"""
    def __init__(self, logger=None):
        # Set up Logger for the class
        self.logger = logger or logging.getLogger(__name__)

        # setting up the path for our list of tickers
        path_var = Path(__file__).parent / '..' / '..'
        self.tickers_filepath = path_var / 'data' / 'tickers.txt'
        if self.tickers_filepath.exists():
            # If we have an ongoing list, load that list
            self.logger.info('Found ticker list, loading tickers into memory...')
            with open(self.tickers_filepath, 'r') as f:
                self.tickers = json.load(f)
            self.logger.info('Read in %s tickers', len(self.tickers))
            self.logger.debug('Read list: %s', self.tickers)
        else:
            # otherwise load the backup list and re-save it
            self.logger.info('Did not find a list of tickers! loading backup...')
            with open(path_var / 'data' / 'backup' / 'tickers.txt', 'r') as f:
                self.tickers = json.load(f)
            with open(self.tickers_filepath, 'w') as f:
                json.dump(self.tickers, f)  # save
            self.logger.info('Successfully restored the backup')
            self.logger.info('Read in %s tickers', len(self.tickers))

    def add_ticker(self, symbol):
        # method to add a ticker to the list
        # TODO: check if the ticker is a valid ticker somehow, or log an error message
        symbol_clean = symbol.upper()
        if symbol_clean in self.tickers:
            self.logger.error('symbol already in ticker list! skipping appending...')
            return
        # Append symbol to internal list then save list
        self.tickers.append(symbol_clean)
        with open(self.tickers_filepath, 'w') as f:
            json.dump(self.tickers, f)
        self.logger.info('Added symbol %s to our list', symbol_clean)

    def update(self):
        # download the day's new data
        path_var = self.tickers_filepath.parent / 'raw'
        file_list = path_var.glob('*_*.csv')
        files_to_ignore = []
        self.logger.info('Moving old files to backup...')
        for file in file_list:
            # build list of files to ignore
            if datetime.datetime.fromtimestamp(file.stat().st_mtime).date() == datetime.datetime.today().date():
                files_to_ignore.append(file.name.split('_')[0])
            else:
                file.rename(file.parent.parent / 'backup' / 'raw' / file.name)  # moving file to backup database
                self.logger.debug('Moving %s to backup', file.name)

        self.logger.info('Finished moving old files to backup!')
        self.logger.info('Downloading files needed...')

        for ticker in self.tickers:
            if ticker in files_to_ignore:
                continue  # skip over files we don't need to update
            self.logger.debug('Starting download for: %s', ticker)
            _download_stock(ticker)
            self.logger.debug('Finished download!')

        self.logger.info('Finished all downloads!')

    def remove_ticker(self, symbol):
        # method to take a particular ticker off of the list
        symbol_clean = symbol.upper()
        if symbol_clean not in self.tickers:
            self.logger.error('symbol %s already does not exist!', symbol_clean)
            return
        # remove symbol from our list and save over file
        self.tickers.remove(symbol)
        with open(self.tickers_filepath, 'w') as f:
            json.dump(self.tickers, f)
        self.logger.info('Successfully removed %s from the database', symbol_clean)
