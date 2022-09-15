import os
import pickle
from time import gmtime, strftime

import matplotlib as mpl

mpl.use('Agg')  # make matplotlib non-interactive
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from preprocess import Preprocessing
from training import Training

SCRIPTS_FOLDER = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_FOLDER, os.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data')

from utils import SAMPLE_NAME, MAX_KMER_LENGTH, MAX_TTEST_KMER_NUM_TO_COMPUTE, MAX_TTEST_TO_USE, \
    MAX_TTEST_PVALUE_TO_COMPUTE, MAX_TTEST_KMER_NUM_PLOT, RAMDOM_STATE, MAX_SAMPLES_TO_USE_POS_FOR_MODEL, \
    MAX_SAMPLES_TO_USE_NEG_FOR_MODEL, MAX_SAMPLES_TO_USE_POS_FOR_ALGO, MAX_SAMPLES_TO_USE_NEG_FOR_ALGO, \
    RUN_REVERSED_COMPLEMENT, FEATURE_SELECTION_SIZE, LDA_N_COMPONENTS, PCA_N_COMPONENTS

if __name__ == '__main__':
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' - Start run kmers ' + (
        'with' if RUN_REVERSED_COMPLEMENT else 'without') + ' reversed complement'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    prepross = Preprocessing(SAMPLE_NAME, INPUT_DIR, OUTPUT_DIR, MAX_KMER_LENGTH, MAX_TTEST_KMER_NUM_PLOT,
                             RUN_REVERSED_COMPLEMENT, MAX_TTEST_PVALUE_TO_COMPUTE, MAX_TTEST_KMER_NUM_TO_COMPUTE)
    # Each of the functions run only one time for creation of output files. The next function get its input from output file of the previous fucntion.
    # prepross.create_featuers()
    # prepross.create_histograms()
    # kmers_freq_pos_df_ttest_fh, kmers_freq_neg_df_ttest_fh = prepross.create_t_test()  # return the features with low ttest
    # prepross.create_scatter_matrix()

    # run the last step with this command:
    # bsub -q bio -e error.txt -o output.txt -n 48  -R "rusage[mem=300]" -R "span[hosts=1]"  "~/miniconda2/envs/dnaclass/bin/python find_feature.py"

    pos_df = pickle.load(open(prepross.kmers_freq_pos_df_ttest_fh, 'rb'))
    neg_df = pickle.load(open(prepross.kmers_freq_neg_df_ttest_fh, 'rb'))
    # FEATUER SELECTION
    Training(pos_df=pos_df, neg_df=neg_df, ttest_to_use_start=0, ttest_to_use_end=MAX_TTEST_TO_USE,
             max_samples_to_use_pos_for_model=MAX_SAMPLES_TO_USE_POS_FOR_MODEL, max_samples_to_use_neg_for_model=MAX_SAMPLES_TO_USE_NEG_FOR_MODEL,
             max_samples_to_use_pos_for_algo=MAX_SAMPLES_TO_USE_POS_FOR_ALGO,max_samples_to_use_neg_for_algo=MAX_SAMPLES_TO_USE_NEG_FOR_ALGO,
             output_dir=OUTPUT_DIR, random_state=RAMDOM_STATE, feature_selection_size=FEATURE_SELECTION_SIZE,
             run_pca=True, pca_n_components=PCA_N_COMPONENTS, run_lda=False, lda_n_components=LDA_N_COMPONENTS)

    # TRAINING
    # for i in xrange(10, MAX_TTEST_TO_USE, 10):
    #     if i+1 > pos_df.shape[1]:
    #         break
    #     Training(pos_df, neg_df, 0, i,
    #              MAX_SAMPLES_TO_USE_POS_FOR_MODEL, MAX_SAMPLES_TO_USE_NEG_FOR_MODEL, MAX_SAMPLES_TO_USE_POS_FOR_ALGO,
    #              MAX_SAMPLES_TO_USE_NEG_FOR_ALGO, OUTPUT_DIR, RAMDOM_STATE)
