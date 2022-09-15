import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import TEST_SIZE, OUTER_CV, N_JOBS_OUTER_CV, LEARN_CURVE_CV, N_JOBS_LEARN_CURVE_CV, MODEL_SELECT_CV, \
    N_JOBS_MODEL_SELECT_CV, FEATURE_SELECTION_SIZE, PCA_N_COMPONENTS, LDA_N_COMPONENTS
from utils import logtime


class Training(object):
    """Training the model
    Args:
        pos_df                              (df): df of positive exmmples
        neg_df                              (df): df of negative examples
        project_root                        (str): path to root folder
        random_state                        (int): seed for random number generator. default=1
        ttest_to_use_start                  (int): start index of features to use. It start to takes the features with the lowest ttest values
        ttest_to_use_end                    (int): end index of features to use. It start to takes the features with the lowest ttest values
        max_samples_to_use_pos_for_model    (int): how many samples to use for the model selection (selecting hyperparameters) in pos set
        max_samples_to_use_neg_for_model    (int): how many samples to use for the model selection (selecting hyperparameters) in neg set
        max_samples_to_use_pos_for_algo     (int): how many samples to use for the selecting the best algorithm in pos set
        max_samples_to_use_neg_for_algo     (int): how many samples to use for the selecting the best algorithm in neg set
        project_root                        (stt): root directory of the project (for output path)
        feature_selection_size              (float): the fraction of the features to continue after the feature selection. The pipeline continue with the best features
        run_pca                             (bool): transfrom the data by pca before the training (after feature selection) - run pca or lda or none of them but not both
        pca_n_components                    (int): number of pc's of PCA
        run_lda                             (bool): transfrom the data by lda before the training (after feature selection) - run pca or lda or none of them but not both
        lda_n_components                    (int): number of components of LDA
    """

    def __init__(self, pos_df, neg_df, ttest_to_use_start, ttest_to_use_end, max_samples_to_use_pos_for_model,
                 max_samples_to_use_neg_for_model, max_samples_to_use_pos_for_algo, max_samples_to_use_neg_for_algo,
                 output_dir, random_state, feature_selection_size=FEATURE_SELECTION_SIZE, run_pca=False,
                 pca_n_components=PCA_N_COMPONENTS, run_lda=False, lda_n_components=LDA_N_COMPONENTS):
        self.pos_df_for_model = pos_df.iloc[:max_samples_to_use_pos_for_model, ttest_to_use_start:ttest_to_use_end]
        self.neg_df_for_model = neg_df.iloc[:max_samples_to_use_neg_for_model, ttest_to_use_start:ttest_to_use_end]
        self.pos_df_for_algo = pos_df.iloc[:max_samples_to_use_pos_for_algo, ttest_to_use_start:ttest_to_use_end]
        self.neg_df_for_algo = neg_df.iloc[:max_samples_to_use_neg_for_algo, ttest_to_use_start:ttest_to_use_end]
        self.output_dir = output_dir
        self.random_state = random_state
        self.feature_selection_size = feature_selection_size
        self.run_pca = run_pca
        self.run_lda = run_lda
        self.pca_n_components = pca_n_components
        self.lda_n_components = lda_n_components
        print logtime() + ' - Start to train the data'
        print logtime() + ' - For algorithm selection we use with %d and %d samples for pos and neg respectively' % (
            max_samples_to_use_pos_for_algo, max_samples_to_use_neg_for_algo)
        print logtime() + ' - For model selection we use with %d and %d samples for pos and neg respectively' % (
            max_samples_to_use_pos_for_model, max_samples_to_use_neg_for_model)
        print logtime() + ' - We used with features: %d-%d' % (ttest_to_use_start, ttest_to_use_end)
        self.X_model, self.y_model = self.combine_pos_neg(self.pos_df_for_model, self.neg_df_for_model)
        self.X_algo, self.y_algo = self.combine_pos_neg(self.pos_df_for_algo, self.neg_df_for_algo)
        self.feature_selection()  # change the self.
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_train_test()
        self.training()

    def combine_pos_neg(self, pos_df, neg_df):
        """Combine the positive and negative df to one numpy matrix, and create numpy array of labels
        Args:
            pos_df      (df): df of positive examples after filtering the required number of samples and featuers
            neg_df      (df): df of positive examples after filtering the required number of samples and featuers
        Return:
            data        (np matrix): combined positive and negative df
            label       (np array): labels of the data
        """
        print logtime() + ' - Combine positive and negative sets'
        pos_samp_num = pos_df.shape[0]
        neg_samp_num = neg_df.shape[0]
        labels_pos = [1] * pos_samp_num
        labels_neg = [0] * neg_samp_num
        labels = np.concatenate([labels_pos, labels_neg])
        data = pd.concat([pos_df, neg_df]).values.astype(np.float64)
        return (data, labels)

    def create_train_test(self):
        """Split the data to train and test sets
        Return:
            X_train     (np matrix): values of the train
            X_test      (np matrix): values of the test
            y_train     (np arrray): lables of the train
            y_test      (np arrray): lables of the test
        """
        print logtime() + ' - Split to train and test sets - test_size: %s' % TEST_SIZE
        return train_test_split(self.X_model, self.y_model, test_size=TEST_SIZE, random_state=self.random_state)

    def get_best_algorithm(self, outer_cv=OUTER_CV, n_jobs_outer_cv=N_JOBS_OUTER_CV):
        """Create nested cross-validation on ALL data with specific algorithm for getting the average score.
        Args:
            grid_search_algo_list (list of GridSearchCV object): list of GridSearchCV
        Returns:
            best_grid_search (GridSearchCV object): The best GridSearchCV object with the highest average score
        """
        print logtime() + ' - Select the best algorithm with nested-cross-validation'
        algorithm_scores = []
        for gs in self.grid_search_algo_list:
            print logtime() + ' - Start training algorithm %s' % gs
            # scores is outer_cv-length-list with one score for each cv
            scores = cross_val_score(gs, self.X_algo, self.y_algo, scoring='accuracy', cv=outer_cv,
                                     n_jobs=n_jobs_outer_cv)
            mean_scores = np.mean(scores)
            algorithm_scores.append(mean_scores)
            print logtime() + ' - Mean score: %.3f for algorithm: %s' % (mean_scores, gs.estimator)
        best_grid_search = self.grid_search_algo_list[np.argmax(algorithm_scores)]
        print logtime() + ' - The algorithm with the best score (%.3f) is %s' % (
            max(algorithm_scores), best_grid_search.estimator)
        return best_grid_search

    def feature_selection(self):
        print logtime() + ' - Start feature selection'
        pipe_lr = Pipeline([('scl', StandardScaler()),
                            ('lr', LogisticRegression(random_state=self.random_state))])
        features_scores = []
        for i in xrange(0, self.X_model.shape[1]):
            X_one_feature = self.X_model[:, i].reshape(-1, 1)
            scores = cross_val_score(pipe_lr, X_one_feature, self.y_model, scoring='roc_auc', cv=5, n_jobs=5)
            mean_scores = np.mean(scores)
            features_scores.append((i, mean_scores))
        sorted_features_scores = sorted(features_scores, key=lambda s: s[1], reverse=True)
        sorted_features_scores_indexes = [i for i, mean_scores in sorted_features_scores]
        best_features_indexes = sorted_features_scores_indexes[
                                :int(len(sorted_features_scores_indexes) * self.feature_selection_size)]
        # Take 20 percent of the featuers, who have the best scores
        self.X_model = self.X_model[:, best_features_indexes]
        self.X_algo = self.X_algo[:, best_features_indexes]
        pickle.dump(self.X_model, open('best_featuers.df', 'wb'))

    def get_best_model(self, grid_search_best_algo):
        """Run GridSearchCV on TRAIN set with the best algorithm to get the model with the best parameters
        Args:
            grid_search_best_algo   (GridSearchCV object): GridSearchCV object of the best algorithm that selected by get_best_algorithm method
        Returns:
            best_model  (sklearn model): The best model object
        """
        print logtime() + ' - Start model selection of the algorithm: %s' % (
            grid_search_best_algo.estimator)
        grid_search_best_algo = grid_search_best_algo.fit(self.X_train, self.y_train)
        print logtime() + ' - The model with the best score (%.3f) in cross-validation is: %s' % (
            grid_search_best_algo.best_score_, grid_search_best_algo.best_params_)
        best_model = grid_search_best_algo.best_estimator_
        best_model.fit(self.X_train, self.y_train)

        print logtime() + ' - The score of the best model on training is: %.3f' % (
            best_model.score(self.X_train, self.y_train))
        print logtime() + ' - The score of the best model on test is: %.3f' % (
            best_model.score(self.X_test, self.y_test))
        return best_model

    def estimate_model_on_test(self, best_model, cv=LEARN_CURVE_CV, n_jobs=N_JOBS_LEARN_CURVE_CV):
        """Run the best model on different sizes of training sets, and plot the training and test accurracy vs. training set size.
           The plot saved in file: "output/accuracy_vs_train_size.png"
         Args:
             best_model  (sklearn model): The best model object
         """
        print logtime() + ' - Start estimate trained model on test sets in veriety of sizes'
        train_sizes, train_scores, test_scores = learning_curve(estimator=best_model, X=self.X_train, y=self.y_train,
                                                                train_sizes=np.linspace(0.1, 1, 10), cv=cv,
                                                                n_jobs=n_jobs)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                 label='validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim([0.8, 1.0])
        output_graph = os.path.join(self.output_dir, 'accuracy_vs_train_size.png')
        plt.savefig(output_graph)
        print logtime() + ' - Run ended see accuraty vs. train size graph in %s' % output_graph
        # plt.show()

    @property
    def grid_search_algo_list(self):
        """List of GridSearchCV objects with different pipelines. Each of them represent specific algorithm with variety of hyperparameters
        Returns:
            algo_list   (list of GridSearchCV): List of GridSearchCV objects with different pipelines.

        """
        pca = [('pca', PCA(n_components=self.pca_n_components))] if self.run_pca else []
        pipe_lr = Pipeline([('scl', StandardScaler())] + pca +
                           [('lr', LogisticRegression(random_state=self.random_state))])
        MODEL_SELECT_LR_C = [10 ** int(C) for C in np.arange(-5, 5)]
        MODEL_SELECT_LR_PENALTY = ['l2', 'l1']
        param_grid = [{'lr__C': MODEL_SELECT_LR_C,
                       'lr__penalty': MODEL_SELECT_LR_PENALTY}]
        gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                             n_jobs=N_JOBS_MODEL_SELECT_CV)

        pipe_svm = Pipeline([('scl', StandardScaler())] + pca +
                            [('svm', SVC(random_state=self.random_state))])
        MODEL_SELECT_SVM_C = [10 ** int(C) for C in np.arange(-5, 5)]
        MODEL_SELECT_SVM_RBF_GAMMA = ['auto']
        param_grid = [{'svm__C': MODEL_SELECT_SVM_C,
                       'svm__kernel': ['linear']},
                      {'svm__C': MODEL_SELECT_SVM_C,
                       'svm__gamma': MODEL_SELECT_SVM_RBF_GAMMA,
                       'svm__kernel': ['rbf']}]
        gs_svm = GridSearchCV(estimator=pipe_svm, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                              n_jobs=N_JOBS_MODEL_SELECT_CV)

        pipe_tree = Pipeline(pca + [('tree', DecisionTreeClassifier(random_state=self.random_state))])
        MODEL_SELECT_TREE_CRITERION = ['entropy']
        MODEL_SELECT_TREE_DEPTH = [3]
        param_grid = [{'tree__criterion': MODEL_SELECT_TREE_CRITERION,
                       'tree__max_depth': MODEL_SELECT_TREE_DEPTH}]
        gs_tree = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                               n_jobs=N_JOBS_MODEL_SELECT_CV)

        tree = DecisionTreeClassifier(random_state=self.random_state, criterion='entropy', max_depth=1)
        pipe_ada = Pipeline(pca + [('adaboost', AdaBoostClassifier(base_estimator=tree, random_state=self.random_state))])
        param_grid = [{'n_estimators': [500],
                       'learning_rate': [0.1]}]
        gs_ada = GridSearchCV(estimator=pipe_ada, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                              n_jobs=N_JOBS_MODEL_SELECT_CV)

        pipe_forest = Pipeline(pca + [('forest', RandomForestClassifier(random_state=self.random_state))])
        MODEL_SELECT_FOREST_CRITERION = ['entropy']
        MODEL_SELECT_FOREST_N_ESTIMATORS = [10]
        # Check the parameters: max_depth, max_features
        param_grid = [{'criterion': MODEL_SELECT_FOREST_CRITERION,
                       'n_estimators': MODEL_SELECT_FOREST_N_ESTIMATORS}]
        gs_forest = GridSearchCV(estimator=pipe_forest, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                                 n_jobs=N_JOBS_MODEL_SELECT_CV)

        pipe_knn = Pipeline(pca + [('knn', KNeighborsClassifier())])
        MODEL_SELECT_KNN_N_NEIGHBORS = [5]
        MODEL_SELECT_KNN_P = [2]
        param_grid = [{'n_neighbors': MODEL_SELECT_KNN_N_NEIGHBORS,
                       'p': MODEL_SELECT_KNN_P}]
        gs_knn = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, scoring='accuracy', cv=MODEL_SELECT_CV,
                              n_jobs=N_JOBS_MODEL_SELECT_CV)

        algo_list = [gs_lr, gs_tree, gs_ada, gs_knn]
        # algo_list = [gs_lr, gs_svm, gs_tree, gs_ada, gs_forest, gs_knn]
        return algo_list

    def training(self):
        """Training model and estimate the accuracy on different sizes of training sets
        Save the model in 'output/best_model_object' file
        """
        print logtime() + ' - Start the training pipeline'
        best_grid_search = self.get_best_algorithm()
        best_model = self.get_best_model(best_grid_search)
        pickle.dump(best_model, open(os.path.join(self.output_dir, 'best_model_object'), "wb"))
        self.estimate_model_on_test(best_model)
