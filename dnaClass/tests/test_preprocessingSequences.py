import os
from unittest import TestCase

from create_features import PreprocessingSequences

TEST_FOLDER = os.path.dirname(__file__)
TEST_DATA = os.path.abspath(os.path.join(TEST_FOLDER, 'test-data'))


class TestPreprocessingSequences(TestCase):
    def setUp(self):
        fh = os.path.join(TEST_DATA, 'h3_small.pos')
        self.prepros = PreprocessingSequences(fh)

    def tearDown(self):
        pass

    def test_get_sequences(self):
        sequences = [seq for seq in self.prepros.get_sequences()]
        self.assertSequenceEqual(sequences, ['AATGA', 'CAATA'])
