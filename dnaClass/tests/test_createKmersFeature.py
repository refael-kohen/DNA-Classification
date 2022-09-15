import os
from unittest import TestCase

from create_features import CreateKmersFeature

TEST_FOLDER = os.path.dirname(__file__)
TEST_DATA = os.path.abspath(os.path.join(TEST_FOLDER, 'test-data'))


class TestCreateKmersFeature(TestCase):
    def setUp(self):
        self.kmers = CreateKmersFeature(max_k=2)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertDictEqual(self.kmers.kmers_frequencies_temp,
                             {'T': [], 'TT': [], 'TG': [], 'TC': [], 'TA': [], 'G': [], 'GT': [], 'GG': [], 'GC': [],
                              'GA': [], 'C': [], 'CT': [], 'CG': [], 'CC': [], 'CA': [], 'A': [], 'AT': [], 'AG': [],
                              'AC': [], 'AA': []})

    def test_get_kmer_rc(self):
        self.assertEqual(self.kmers.get_kmer_rc('AGCT'), 'AGCT')

    def test_get_kmer_comp(self):
        self.assertEqual(self.kmers.get_kmer_comp('AGCT'), 'TCGA')

    def test_create_all_kmers(self):
        all_kmers_key_value_expected = {('T', 'T'), ('TT', 'TT'), ('TG', 'TG'), ('TC', 'TC'), ('TA', 'TA'), ('G', 'G'),
                                        ('GT', 'GT'), ('GG', 'GG'), ('GC', 'GC'), ('GA', 'GA'), ('C', 'C'),
                                        ('CT', 'CT'), ('CG', 'CG'), ('CC', 'CC'), ('CA', 'CA'), ('A', 'A'),
                                        ('AT', 'AT'), ('AG', 'AG'), ('AC', 'AC'), ('AA', 'AA')}
        all_kmers_key_value = {(k, v) for k, v in self.kmers.kmers_trie.items()}
        self.assertSetEqual(all_kmers_key_value, all_kmers_key_value_expected)

    def test_find_frequencies(self):
        sequences_iterator = ['AAA', 'GTAC']
        kmer_freq = self.kmers.find_frequencies(sequences_iterator)
        self.assertDictEqual(kmer_freq,
                             {'T': [0, 1], 'TT': [0, 0], 'TG': [0, 0], 'TC': [0, 0], 'TA': [0, 1], 'G': [0, 1],
                              'GT': [0, 1], 'GG': [0, 0], 'GC': [0, 0], 'GA': [0, 0], 'C': [0, 1], 'CT': [0, 0],
                              'CG': [0, 0], 'CC': [0, 0], 'CA': [0, 0], 'A': [3, 1], 'AT': [0, 0], 'AG': [0, 0],
                              'AC': [0, 1], 'AA': [2, 0]})

    def test_combine_kmer_with_reverse_complement(self):
        sequences_iterator = ['AAA', 'GTAC']
        frequencies = self.kmers.find_frequencies(sequences_iterator)
        frequencies_rc = self.kmers.combine_kmer_with_reverse_complement(frequencies)
        self.assertDictEqual(frequencies_rc,
                             {'CA_TG': [0, 0], 'A_T': [3, 2], 'C_G': [0, 2], 'AA_TT': [2, 0], 'CC_GG': [0, 0],
                              'GA_TC': [0, 0], 'AC_GT': [0, 2], 'AG_CT': [0, 0]})

        # Meatime not in use
        # def test_combine_kmer_with_complement(self):
        #     sequences_iterator = ['AAA', 'TGAC']
        #     kmer_freq = self.kmers.find_frequencies(sequences_iterator)
        #     kmer_comp_freq = self.kmers.combine_kmer_with_complement(kmer_freq)
        #     self.assertDictEqual(kmer_comp_freq,
        #                          {'AG_TC': [0, 0], 'A_T': [3, 2], 'C_G': [0, 2], 'CA_GT': [0, 0], 'AT_TA': [0, 0],
        #                           'AA_TT': [2, 0], 'CC_GG': [0, 0], 'CG_GC': [0, 0], 'CT_GA': [0, 1], 'AC_TG': [0, 2]})
