import unittest
from run_research import config
class ConfigTests(unittest.TestCase):
    def test_defaults(self): self.assertEqual(config({})['source'],'synthetic')
    def test_injection(self):
        with self.assertRaises(ValueError): config({'RESEARCH_SYMBOL':'BTCUSDT; echo bad'})
    def test_bad_cost(self):
        for value in ('nan','inf','-1','1'):
            with self.assertRaises(ValueError): config({'RESEARCH_FEE':value})
    def test_bounds(self):
        with self.assertRaises(ValueError): config({'RESEARCH_DAYS':'9999'})

    def test_csv_path(self):
        for path in ('/etc/passwd','../input.csv','data/../../input.csv','other/file.csv'):
            with self.assertRaises(ValueError): config({'RESEARCH_SOURCE':'csv','RESEARCH_DATA_FILE':path})
        self.assertEqual(config({'RESEARCH_SOURCE':'csv'})['data_file'],'data/candles.csv')
