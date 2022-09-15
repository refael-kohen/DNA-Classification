import os
from unittest import TestCase

import pandas as pd
from create_features import OrganizeFeatures

TEST_FOLDER = os.path.dirname(__file__)
TEST_DATA = os.path.abspath(os.path.join(TEST_FOLDER, 'test-data'))


class TestOrganizeFeatures(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convert_dict_to_data_frame(self):
        kmers_freq_list_dict = [dict({'AA': 2, 'CC': 3}), dict({'AA': 1, 'CC': 2})]
        df_observed = OrganizeFeatures(kmers_freq_list_dict).kmers_freq_df
        df_expected = pd.DataFrame([[2,3], [1,2]], columns=['AA','CC'])
        self.assertEqual(df_observed['AA'].tolist(), df_expected['AA'].tolist())
        self.assertEqual(df_observed['CC'].tolist(), df_expected['CC'].tolist())
