import os
import unittest
from src.features.feature_builder import FeatureBuilder
from src.models.model_manager import ModelManager
from src.data.data_loader import DataLoader
from keras import Sequential


class FeatureBuilderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.feat_builder = FeatureBuilder()
        self.feat_builder.read_data('TEST')

    def tearDown(self):
        del self.feat_builder

    def test_read(self):
        self.assertEqual(len(self.feat_builder.data), 7442)  # add assertion here

    def test_add_features(self):
        self.feat_builder.add_features()
        self.assertIsNotNone(self.feat_builder.transformed)

    def test_save_features(self):
        self.feat_builder.add_features()
        self.feat_builder.save_features('TEST')
        file = 'TEST.csv'
        self.assertTrue(file in os.listdir('../data/interim/'))
        os.remove('../data/interim/TEST.csv')

    def test_make_digestible(self):
        self.feat_builder.add_features()
        self.feat_builder.make_digestible()
        self.assertIsNotNone(self.feat_builder.train)
        self.assertIsNotNone(self.feat_builder.val)
        self.assertIsNotNone(self.feat_builder.test)


class ModelManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.model_man = ModelManager('TEST')

    def tearDown(self) -> None:
        del self.model_man

    def test_compile_model(self):
        model = self.model_man.compile_model((1, 1, 1))
        self.assertIsInstance(model, Sequential)
        del model

    def test_save_model(self):
        self.model_man.model = self.model_man.compile_model((1, 1, 1))
        self.model_man.save_model()
        filename = 'TEST-5-day-model.h5'
        self.assertTrue(filename in os.listdir('../models/'))
        os.remove('../models/' + filename)


class DataLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_loader = DataLoader()

    def tearDown(self) -> None:
        del self.data_loader

    def test_file_exists(self):
        filename = 'tickers.txt'
        self.assertTrue(filename in os.listdir('../data'))

    def test_ticker_adjustments(self):
        self.data_loader.add_ticker('TEST')
        self.assertTrue('TEST' in self.data_loader.tickers)
        self.data_loader.remove_ticker('TEST')
        self.assertTrue('TEST' not in self.data_loader.tickers)


if __name__ == '__main__':
    unittest.main()
