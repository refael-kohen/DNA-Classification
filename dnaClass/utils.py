from time import gmtime, strftime

def logtime():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


RUN_REVERSED_COMPLEMENT=False
# consts of Training
RAMDOM_STATE=1
TEST_SIZE = 0.3  # Size of test set
PCA_N_COMPONENTS = 20
LDA_N_COMPONENTS = 20
OUTER_CV = 5  # The cross validation number of the outer loop of the nested cross-validation for algorithm selection.
N_JOBS_OUTER_CV = OUTER_CV  # Number of jobs for OUTER_CV
LEARN_CURVE_CV = 10  # Cross validation size for estimation on test set
N_JOBS_LEARN_CURVE_CV = LEARN_CURVE_CV
MODEL_SELECT_CV = 10  # Cross validation size for model selection
N_JOBS_MODEL_SELECT_CV = MODEL_SELECT_CV
FEATURE_SELECTION_SIZE=float(1)/5 #fraction of features to take after feature selection
# end of Training consts

# consts of Preprocessing
SAMPLE_NAME = 'h3'
MAX_KMER_LENGTH = 6
# significant kmers with low ttest. You can choose maximum p-value of t-test:MAX_TTEST_PVALUE or absolute value: MAX_TTEST_KMER_NUM
MAX_TTEST_PVALUE_TO_COMPUTE = 0.05  # float (for example: 0.0) Or None if MAX_TTEST_KMER_NUM is not None. The number of the results must be >= MAX_TTEST_KMER_NUM_PLOT
MAX_TTEST_KMER_NUM_TO_COMPUTE = None  # int (for example: 20) Or None if MAX_TTEST_PVALUE is not None. Must be >= MAX_TTEST_KMER_NUM_PLOT
MAX_TTEST_KMER_NUM_PLOT = 20  # Number of kmers to create plot of correlation between them.
MAX_TTEST_TO_USE = 100000 # How many features to use from the file that created with MAX_TTEST_PVALUE_TO_COMPUTE or MAX_TTEST_KMER_NUM_TO_COMPUTE parameters. The value need to be less or equal to what computed
MAX_SAMPLES_TO_USE_POS_FOR_MODEL = 100000000 # The max number of samples to use in positive set for model selection (hyperparameters of the model)
MAX_SAMPLES_TO_USE_NEG_FOR_MODEL = 100000000 # The max number of samples to use in negative set for model selection (hyperparameters of the model)
MAX_SAMPLES_TO_USE_POS_FOR_ALGO = 10000000 # The max number of samples to use in positive set for algorithm selection
MAX_SAMPLES_TO_USE_NEG_FOR_ALGO = 10000000 # The max number of samples to use in negative set for algorithm selection
# end of Preprocessing consts
