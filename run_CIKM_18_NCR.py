#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import traceback
from functools import partial

import numpy as np

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
from Conferences.CIKM.NCR_our_interface.ExampleDatasetProvided.Movielens1MReader import Movielens1MReader
from Conferences.CIKM.NCR_our_interface.NCRWrapper import NCRWrapper
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Recommender_import_list import *
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table


def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_DL_article_default=False, flag_DL_tune=False,
                               flag_print_results=False):
    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    if dataset_name == "movielens1m":
        dataset = Movielens1MReader(result_folder_path)

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return

    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_negative = dataset.URM_DICT["URM_negative"].copy()

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    metric_to_optimize = "HIT_RATE"
    cutoff_list_validation = [10]
    cutoff_list_test = [10]

    n_cases = 50
    n_random_starts = 15

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_negative, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_negative, cutoff_list=cutoff_list_test)

    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:
        article_hyperparameters = {
            "epochs": 50,
            "batch_size": 256,
            "num_factors": 8,
            "layers": (32, 32, 16, 8),
            "reg_mf": 0,
            "reg_layers": (0, 0, 0, 0),
            "num_negatives": 2,
            "learning_rate": 0.0005,
            "learner": "adam",
            "k": 2
        }

        # Do not modify earlystopping
        earlystopping_hyperparameters = {"validation_every_n": 1,
                                         "stop_on_validation": False,
                                         "lower_validations_allowed": 5,
                                         "evaluator_object": evaluator_validation,
                                         "validation_metric": metric_to_optimize,
                                         }

        # This is a simple version of the tuning code that is reported below and uses SearchSingleCase
        # You may use this for a simpler testing
        # recommender_instance = HERSWrapper(URM_train, UCM_train, ICM_train)
        #
        # recommender_instance.fit(**article_hyperparameters,
        #                          **earlystopping_hyperparameters)
        #
        # evaluator_test.evaluateRecommender(recommender_instance)

        # Fit the DL model, select the optimal number of epochs and save the result
        parameterSearch = SearchSingleCase(NCRWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
            FIT_KEYWORD_ARGS=earlystopping_hyperparameters)

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path=result_folder_path,
                               output_file_name_root=NCRWrapper.RECOMMENDER_NAME)

    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS - Nothing should be modified below this point
    ######

    if flag_baselines_tune:

        ################################################################################################
        ###### Collaborative Baselines

        collaborative_algorithm_list = [
            Random,
            TopPop,
            ItemKNNCFRecommender,
            PureSVDRecommender,
            SLIM_BPR_Cython,
        ]

        # Running hyperparameter tuning of baslines
        # See if the results are reasonable and comparable to baselines reported in the paper
        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train=URM_train,
                                                           URM_train_last_test=URM_train_last_test,
                                                           metric_to_optimize=metric_to_optimize,
                                                           evaluator_validation_earlystopping=evaluator_validation,
                                                           evaluator_validation=evaluator_validation,
                                                           evaluator_test=evaluator_test,
                                                           output_folder_path=result_folder_path,
                                                           resume_from_saved=True,
                                                           parallelizeKNN=False,
                                                           allow_weighting=True,
                                                           n_cases=n_cases,
                                                           n_random_starts=n_random_starts)

        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:
        n_validation_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)
        n_test_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)

        print_time_statistics_latex_table(result_folder_path=result_folder_path,
                                          dataset_name=dataset_name,
                                          algorithm_name=ALGORITHM_NAME,
                                          other_algorithm_list=[NCRWrapper],
                                          KNN_similarity_to_report_list=KNN_similarity_to_report_list,
                                          n_validation_users=n_validation_users,
                                          n_test_users=n_test_users,
                                          n_decimals=2)

        print_results_latex_table(result_folder_path=result_folder_path,
                                  algorithm_name=ALGORITHM_NAME,
                                  file_name_suffix="article_metrics_",
                                  dataset_name=dataset_name,
                                  metrics_to_report_list=["HIT_RATE", "NDCG"],
                                  cutoffs_to_report_list=cutoff_list_test,
                                  other_algorithm_list=[NCRWrapper],
                                  KNN_similarity_to_report_list=KNN_similarity_to_report_list)

        print_results_latex_table(result_folder_path=result_folder_path,
                                  algorithm_name=ALGORITHM_NAME,
                                  file_name_suffix="all_metrics_",
                                  dataset_name=dataset_name,
                                  metrics_to_report_list=["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST",
                                                          "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                  cutoffs_to_report_list=cutoff_list_validation,
                                  other_algorithm_list=[NCRWrapper],
                                  KNN_similarity_to_report_list=KNN_similarity_to_report_list)


if __name__ == '__main__':

    ALGORITHM_NAME = "cikm"
    CONFERENCE_NAME = "NCR"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune', help="Baseline hyperparameter search", type=bool, default=False)
    parser.add_argument('-a', '--DL_article_default', help="Train the DL model with article hyperparameters", type=bool, default=True)
    parser.add_argument('-p', '--print_results', help="Print results", type=bool, default=False)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = ["movielens1m"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_DL_article_default=input_flags.DL_article_default,
                                   flag_print_results=input_flags.print_results,
                                   )
