import itertools
import os
import pickle
from time import gmtime, strftime

import matplotlib as mpl

mpl.use('Agg')  # make matplotlib non-interactive
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from pandas.plotting import scatter_matrix
from scipy import stats

plt.style.use('ggplot')

from create_features import PreprocessingSequences, CreateKmersFeature, OrganizeFeatures


class Preprocessing(object):
    """
    Find frequencies of k-mers in the sequences (dumping to file)
    Create hisograms for specific kmers (for now hard-coded). The histogram show the difference of the frequency of the kmer between pos and neg sets.
    Create t-test for all kmers between the pos and neg sets (dumping to file the most significant kmers -MAX_TTEST_KMER_NUM or MAX_TTEST_PVALUE parameter)
    Create scatter plots of frquencies in the sequences between each kmer to another for several significant kmers (union of pos and neg sets)(MAX_TTEST_KMER_NUM_PLOT parameter)

    Args:
        sample_name                     (str): sample name
        input_dir                       (str): folder with input files
        output_dir                      (str): existing folder for output files
        max_kmer_length                 (int): maximum length of k-mer (create k-mer from 1 to k)
        max_ttest_kmer_num_plot         (int): number of kmer that between them will be created the scatter plots.
        run_reversed_complement         (bool): consider reversed compement of kmers or not
        max_ttest_pvalue_to_compute     (float or None): maximum p-value that under it we will continue the training (only one of the parameters max_ttest_pvalue or max_ttest_kmer_num is not None)
        max_ttest_kmer_num_to_compute   (int or None): number of the most significat kmers that with them we will continue the training (only one of the parameters max_ttest_pvalue or max_ttest_kmer_num is not None)
    Attributes:
        kmers_freq_pos_df_fh            (str): path to file with frequencies of kmers of pos
        kmers_freq_neg_df_fh            (str): path to file with frequencies of kmers of neg
        kmers_freq_pos_rc_df_fh         (str): path to file with frequencies of the combined kmers (reversed complement with the origin) of pos
        kmers_freq_neg_rc_df_fh         (str): path to file with frequencies of the combined kmers (reversed complement with the origin) of neg
        kmers_freq_pos_df_ttest_fh      (str): path to file with frequencies of most signigicant kmers (dataframe) in pos
        kmers_freq_neg_df_ttest_fh      (str): path to file with frequencies of most signigicant kmers (dataframe) in neg
        kmers_freq_corr_fh_csv          (str): path to file with correlation between the most significant kmers (pos and neg combined)
        kmers_freq_corr_fh_png          (str): path to figure file of the kmers_freq_corr_fh_csv file
    """

    def __init__(self, sample_name, input_dir, output_dir, max_kmer_length, max_ttest_kmer_num_plot,
                 run_reversed_complement, max_ttest_pvalue_to_compute=None, max_ttest_kmer_num_to_compute=None):
        self.sample_name = sample_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_kmer_length = max_kmer_length
        self.max_ttest_kmer_num_plot = max_ttest_kmer_num_plot
        self.run_reversed_complement = run_reversed_complement
        self.max_ttest_pvalue_to_compute = max_ttest_pvalue_to_compute
        self.max_ttest_kmer_num_to_compute = max_ttest_kmer_num_to_compute
        self.max_ttest_to_compute = self.max_ttest_kmer_num_to_compute or self.max_ttest_pvalue_to_compute  # Using in output file names

        self.kmers_freq_pos_df_fh = os.path.join(self.output_dir, 'output-%s.pos-dataframe-max_k-%s'
                                                 % (sample_name,
                                                    str(self.max_kmer_length)))
        self.kmers_freq_neg_df_fh = os.path.join(self.output_dir, 'output-%s.neg-dataframe-max_k-%s'
                                                 % (sample_name,
                                                    str(self.max_kmer_length)))
        self.kmers_freq_pos_rc_df_fh = os.path.join(self.output_dir,
                                                    'output-%s.pos-rc-dataframe-max_k-%s' % (
                                                        sample_name, str(self.max_kmer_length)))
        self.kmers_freq_neg_rc_df_fh = os.path.join(self.output_dir,
                                                    'output-%s.neg-rc-dataframe-max_k-%s' % (
                                                        sample_name, str(self.max_kmer_length)))
        self.kmers_freq_pos_df_ttest_fh = os.path.join(self.output_dir,
                                                       'output-%s.pos-dataframe-ttest-%s-first-max_k-%s' % (
                                                           sample_name, str(self.max_ttest_to_compute),
                                                           str(self.max_kmer_length)))
        self.kmers_freq_neg_df_ttest_fh = os.path.join(self.output_dir,
                                                       'output-%s.neg-dataframe-ttest-%s-first-max_k-%s' % (
                                                           sample_name, str(self.max_ttest_to_compute),
                                                           str(self.max_kmer_length)))
        self.kmers_freq_corr_fh_csv = os.path.join(self.output_dir, 'output-%s-corr-max_k-%s.csv' % (
            str(self.max_kmer_length), sample_name))
        self.kmers_freq_corr_fh_png = os.path.join(self.output_dir, 'output-%s-corr-max_k-%s.png' % (
            str(self.max_kmer_length), sample_name))

    def create_featuers(self):
        """Find frequencies of k-mers in the sequences, and save the dataframe in files
        """
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Build trie of kmers'
        kmers = CreateKmersFeature(max_kmer_length=self.max_kmer_length)
        pos_file = os.path.join(self.input_dir, 'h3.pos')
        neg_file = os.path.join(self.input_dir, 'h3.neg')

        sequences_iter_pos = PreprocessingSequences(pos_file).get_sequences()
        sequences_iter_neg = PreprocessingSequences(neg_file).get_sequences()

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Find frequencies in sequences'
        kmers_freq_pos = kmers.find_frequencies(sequences_iter_pos)
        kmers_freq_neg = kmers.find_frequencies(sequences_iter_neg)

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Combine the kmers with reversed complement'
        kmers_freq_pos_rc = kmers.combine_kmer_with_reverse_complement(kmers_freq_pos)
        kmers_freq_neg_rc = kmers.combine_kmer_with_reverse_complement(kmers_freq_neg)

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Convert to dataframe'
        kmers_freq_pos_df = OrganizeFeatures(kmers_freq_pos).kmers_freq_df
        kmers_freq_neg_df = OrganizeFeatures(kmers_freq_neg).kmers_freq_df
        kmers_freq_pos_rc_df = OrganizeFeatures(kmers_freq_pos_rc).kmers_freq_df
        kmers_freq_neg_rc_df = OrganizeFeatures(kmers_freq_neg_rc).kmers_freq_df

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Dump dataframes to files'
        pickle.dump(kmers_freq_pos_df, open(self.kmers_freq_pos_df_fh, "wb"))
        pickle.dump(kmers_freq_neg_df, open(self.kmers_freq_neg_df_fh, "wb"))
        pickle.dump(kmers_freq_pos_rc_df, open(self.kmers_freq_pos_rc_df_fh, "wb"))
        pickle.dump(kmers_freq_neg_rc_df, open(self.kmers_freq_neg_rc_df_fh, "wb"))

    def load_df_from_file(self):
        kmers_freq_pos_df = pickle.load(open(self.kmers_freq_pos_df_fh, "rb"))
        kmers_freq_neg_df = pickle.load(open(self.kmers_freq_neg_df_fh, "rb"))
        kmers_freq_pos_rc_df = pickle.load(open(self.kmers_freq_pos_rc_df_fh, "rb"))
        kmers_freq_neg_rc_df = pickle.load(open(self.kmers_freq_neg_rc_df_fh, "rb"))
        return kmers_freq_pos_df, kmers_freq_neg_df, kmers_freq_pos_rc_df, kmers_freq_neg_rc_df

    def create_histograms(self):
        """Create hisograms for specific kmers (for now hard-coded). The histogram show the difference of the frequency of the kmer between pos and neg sets.
        """
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Create histograms'
        kmers_freq_pos_df, kmers_freq_neg_df, kmers_freq_pos_rc_df, kmers_freq_neg_rc_df = self.load_df_from_file()
        OrganizeFeatures.create_histogram("CGTA", kmers_freq_pos_rc_df["CGTA_TACG"], kmers_freq_neg_rc_df["CGTA_TACG"])
        OrganizeFeatures.create_histogram("AGG", kmers_freq_pos_df["AGG"], kmers_freq_neg_df["AGG"])

    def create_t_test(self):
        """
        Create t-test for all kmers between the pos and neg sets (dumping to file the most significant kmers -MAX_TTEST_KMER_NUM or MAX_TTEST_PVALUE parameter)
        Calculate the correlation between the most sigingicat kmers to each aother (pos and neg together)
        We want to continue only with the most significant and most non-correlated kmers
        """
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Feature selection according T-test'
        kmers_freq_pos_df, kmers_freq_neg_df, kmers_freq_pos_rc_df, kmers_freq_neg_rc_df = self.load_df_from_file()
        if self.run_reversed_complement:
            pos_df = kmers_freq_pos_df.join(kmers_freq_pos_rc_df)
            neg_df = kmers_freq_neg_df.join(kmers_freq_neg_rc_df)
        else:
            pos_df = kmers_freq_pos_df
            neg_df = kmers_freq_neg_df
        kmer_ttest = {}
        for kmer_pos_name in pos_df:
            kmer_ttest[kmer_pos_name] = stats.ttest_ind(pos_df[kmer_pos_name], neg_df[kmer_pos_name])
        sorted_items = sorted(kmer_ttest.items(), key=lambda items: items[1][1])  # sort according p-value
        if not self.max_ttest_kmer_num_to_compute:
            low_p_value = [(kmer, ttest) for kmer, ttest in sorted_items if
                           ttest[1] <= self.max_ttest_pvalue_to_compute]
            if not low_p_value:
                raise AttributeError('No kmer with p-value lower or equals to %s' % self.max_ttest_pvalue_to_compute)
        else:
            print 'not valid'
            low_p_value = sorted_items[:self.max_ttest_kmer_num_to_compute]

        low_kmers = [kmer for kmer, ttest in low_p_value]
        pos_df_low = pos_df[low_kmers]
        neg_df_low = neg_df[low_kmers]
        pickle.dump(pos_df_low, open(self.kmers_freq_pos_df_ttest_fh, "wb"))
        pickle.dump(neg_df_low, open(self.kmers_freq_neg_df_ttest_fh, "wb"))

        pos_neg = pd.concat([pos_df_low, neg_df_low], ignore_index=True)
        corr = pos_neg.corr()
        corr.to_csv(self.kmers_freq_corr_fh_csv)
        svm = sn.heatmap(corr)
        figure = svm.get_figure()
        figure.savefig(self.kmers_freq_corr_fh_png)
        return (self.kmers_freq_pos_df_ttest_fh, self.kmers_freq_neg_df_ttest_fh)

    def create_scatter_matrix(self):
        """
        Create scatter plots of frquencies in the sequences between each kmer to another for several significant kmers (union of pos and neg sets)(MAX_TTEST_KMER_NUM_PLOT parameter)
        The purpose is to see in eye what we see in csv file of the correlations
        """
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Create scatter matrixes'
        kmers_freq_pos_df_ttest = pickle.load(open(self.kmers_freq_pos_df_ttest_fh, "rb"))
        kmers_freq_neg_df_ttest = pickle.load(open(self.kmers_freq_neg_df_ttest_fh, "rb"))
        pos_neg = pd.concat([kmers_freq_pos_df_ttest, kmers_freq_neg_df_ttest])
        for columns in itertools.combinations(range(self.max_ttest_kmer_num_plot), 2):
            scatter_matrix(pos_neg.iloc[:, list(columns)], alpha=0.3)
            str_col = '_'.join(str(x) for x in columns)
            if not os.path.exists(os.path.join(self.output_dir, 'png')):
                os.makedirs(os.path.join(self.output_dir, 'png'))
            plt.savefig(os.path.join(self.output_dir, 'png',
                                     'output-h3.neg-dataframe-max_k-%s-%s.png' % (str(self.max_kmer_length), str_col)))
