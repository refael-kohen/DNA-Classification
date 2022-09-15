import itertools
import string
from copy import deepcopy

import ahocorasick
import matplotlib as mpl
import pandas as pd

mpl.use('Agg')  # make matplotlib non-interactive
import matplotlib.pyplot as plt

BASES = ['A', 'C', 'G', 'T']


class CreateKmersFeature(object):
    """Find frequencies of k-mers in the sequences

    Create all kmers until max_kmer_length length and find frequencies of k-mers in sequences that are feeded by an iterator

    Args:
        max_kmer_length       (int): maximum length of k-mer (create k-mer from 1 to k)
    Attributes:
        max_kmer_length       (int): maximum length of k-mer (create k-mer from 1 to k)
        kmers_trie  (object ahocorasick): Data structure of Ahocorasick
        sequences_num (int): number of sequences. Updated in find_frequencies method
        kmers_frequencies_temp (dict of str:list): Template of Dictionary of frequencies of kmers, contains all possible kmers. key: kmer, value: list of frquencies
    """

    def __init__(self, max_kmer_length):
        self.max_kmer_length = max_kmer_length
        self.kmers_trie = ahocorasick.Automaton()
        self.kmers_frequencies_temp = {}
        self.create_all_kmers()
        self.complement = string.maketrans("ACGT", "TGCA")

    def create_all_kmers(self):
        """Create data structure for frequencies of the kmers
        """
        for k in xrange(1, self.max_kmer_length + 1):
            for kmer in itertools.product(BASES, repeat=k):
                kmer = ''.join(kmer)
                self.kmers_frequencies_temp[kmer] = []
                self.kmers_trie.add_word(kmer, kmer)
        self.kmers_trie.make_automaton()

    def find_frequencies(self, sequences_iterator):
        """Find frequencies of k-mers in the sequences and store them in self.kmers_frequencies
        Args:
            sequences_iterator    (iter of str): iterator of sequences
        Return:
            kmers_frequencies (dict of str:int): Dictionary of frequencies of kmers. key: kmer, value: list of frquencies of the kmer in the sequences.
        """
        kmers_frequencies = deepcopy(self.kmers_frequencies_temp)
        for seq in sequences_iterator:
            for kmer in kmers_frequencies.keys():
                kmers_frequencies[kmer].append(0)
            for end_index, kmer in self.kmers_trie.iter(seq):
                kmers_frequencies[kmer][-1] += 1
        return kmers_frequencies

    #Meantime not in use.
    def get_kmer_rc(self, kmer):
        return kmer.translate(self.complement)[::-1]

    def get_kmer_comp(self, kmer):
        return kmer.translate(self.complement)

    def combine_kmer_with_reverse_complement(self, kmers_frequencies):
        """Combine the frequencies of the k-mers, each k-mer with its complement
        Args:
            frequencies                 (list of dict of str: int): item for each sequences of dictionary with key=kmer, value=frequency (created by find_frequencies method).
        Return:
            frequencies_combined_com    (list of dict of str: int): item for each sequences of dictionary with key=combined_kmer, value=combined_frequency.
        """
        kmer_comp_frequencies = {}
        kmer_used = {}
        for kmer, frequencies in kmers_frequencies.items():
            if kmer not in kmer_used:  # After each iteration we save the kmer and its reversed complement in kmer_used dict in order to not take it twice
                kmer_com = self.get_kmer_rc(kmer)
                if kmer == kmer_com:
                    continue
                kmer_combined = sorted([kmer + "_" + kmer_com, kmer_com + "_" + kmer])
                kmer_comp_frequencies[kmer_combined[0]] = []
                for i in xrange(len(frequencies)):
                    freq = frequencies[i]
                    freq_com = kmers_frequencies[kmer_com][i]
                    kmer_comp_frequencies[kmer_combined[0]].append(freq + freq_com)
                    kmer_used[kmer] = None
                    kmer_used[kmer_com] = None
        return kmer_comp_frequencies

class PreprocessingSequences(object):
    def __init__(self, filename):
        self.seq_file = open(filename)

    def get_sequences(self):
        for seq in self.seq_file:
            seq = seq.replace(' ', '').strip()  # remove whitespaces and endline
            if seq:  # not retrun empty lines
                yield seq


class OrganizeFeatures(object):
    def __init__(self, kmers_freq_dict):
        self.kmers_freq_dict = kmers_freq_dict
        self.kmers_freq_df = pd.DataFrame()
        self.convert_dict_to_data_frame()

    def convert_dict_to_data_frame(self):
        self.kmers_freq_df = pd.DataFrame(self.kmers_freq_dict)

    @staticmethod
    def create_histogram(title,df1,df2):
        # hist = df1.hist(bins=10)
        n, bins, patches = plt.hist(df1, 50, density=True, facecolor='r', alpha=0.75)
        n, bins, patches = plt.hist(df2, 50, density=True, facecolor='g', alpha=0.75)

        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title(title)
        # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.show()