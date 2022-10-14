# class for downloading the stock data

from datetime import datetime
import yfinance as yf
import logging
import json
from pathlib import Path


def _download_stock(symbol):
    """Downloads recent stock data for the given ticker symbol."""
    # Using yahoo finance API to grab stock market data
    data = yf.download(symbol)
    if len(data) == 0:  # if we did not grab any data, error out and skip
        logging.error('Could not find the stock ticker %s, skipping download' % symbol)
        return 1
    # set the filename to be a combination of the ticker and the date ranges
    filename = symbol + '_' + str(data.index[0])[:10] + '_' + str(data.index[-1])[:10] + '.csv'
    path_var = Path(__file__).parent / '..' / '..' / 'data' / 'raw'
    filepath = path_var / filename
    data.to_csv(filepath)  # save the file to a csv
    return 0


class DataLoader(object):
    """DataLoader object used for downloading stockmarket data, tickers are in file tickers.txt"""
    def __init__(self, logger=None):
        """Checks for a ticker list, and restores a backup if not found."""
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
                json.dump(self.tickers, f)  # save the ticker list
            self.logger.info('Successfully restored the backup')
            self.logger.info('Read in %s tickers', len(self.tickers))

    def add_ticker(self, symbol):
        """Adds input ticker symbol to our list, removes if ticker does not exist."""
        symbol_clean = symbol.upper()  # uppercasify
        if symbol_clean in self.tickers:  # skip if the symbol is already in the list
            self.logger.error('symbol already in ticker list! skipping appending...')
            return
        # Append symbol to internal list then save list
        self.tickers.append(symbol_clean)
        with open(self.tickers_filepath, 'w') as f:
            json.dump(self.tickers, f)
        self.logger.info('Added symbol %s to our list', symbol_clean)

        self.update(symbol)  # see if we need to download more data

    def update(self, ticker=None, backup=False):
        """Update the database based on the ticker, or all of the database.
        :param ticker: ticker symbol, defaults to None
        :type ticker: string
        :param backup: whether to backup old data or delete it, defaults to false
        :type backup: boolean

        :return: does not return anything, sets state

        .. notes:: If ticker is None it will go through all of the ticker list.
        """
        # download the day's new data
        path_var = self.tickers_filepath.parent / 'raw'
        file_list = path_var.glob('*_*.csv')  # grab all files in our database
        ticker_list = self.tickers
        if ticker is not None:
            ticker_list = [ticker.upper()]
        files_to_ignore = []
        self.logger.info('Removing old files...')
        for file in file_list:
            # build list of files to ignore and remove or backup old files
            if datetime.fromtimestamp(file.stat().st_mtime).date() == datetime.today().date()\
                    or file.name == 'TEST_DATA.csv':
                files_to_ignore.append(file.name.split('_')[0])
            elif backup:
                self.logger.debug('Moving %s to backup', file.name)
                file.rename(file.parent.parent / 'backup' / 'raw' / file.name)  # moving file to backup database
            else:
                self.logger.debug('Deleting %s', file.name)
                file.unlink()

        self.logger.info('Finished removing old files!')
        self.logger.info('Downloading files needed...')

        self.logger.debug('Here is the ticker list: %s', ticker_list)
        for ticker in ticker_list:
            if ticker in files_to_ignore:
                continue  # skip over files we don't need to update
            self.logger.debug('Starting download for: %s', ticker)
            response = _download_stock(ticker)
            if response:  # if we could not download any data, remove the ticker from our list
                self.logger.error('Couldn\'t download %s, removing from list...', ticker)
                self.remove_ticker(ticker)
            self.logger.debug('Finished download!')

        self.logger.info('Finished all downloads!')

    def remove_ticker(self, symbol):
        """Removes the specified ticker symbol from our list."""
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
