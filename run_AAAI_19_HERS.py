#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import traceback
from functools import partial

import numpy as np

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.DeliciousHetrec2011ColdItemsReader import DeliciousHetrec2011ColdItemsReader
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.DeliciousHetrec2011ColdUsersReader import DeliciousHetrec2011ColdUsersReader
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.DeliciousHetrec2011Reader import DeliciousHetrec2011Reader
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.LastFMHetrec2011ColdItemsReader import LastFMHetrec2011ColdItemsReader
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.LastFMHetrec2011ColdUsersReader import LastFMHetrec2011ColdUsersReader
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from Conferences.AAAI.HERS_our_interface.HERSWrapper import HERSWrapper
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.run_parameter_search import runParameterSearch_Content, runParameterSearch_Hybrid, runParameterSearch_Collaborative
from Recommender_import_list import *
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table


def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_DL_article_default=False, flag_DL_tune=False,
                               flag_print_results=False):
    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    if dataset_name == "delicious-hetrec2011":
        dataset = DeliciousHetrec2011Reader(result_folder_path)

    elif dataset_name == "delicious-hetrec2011-cold-users":
        dataset = DeliciousHetrec2011ColdUsersReader(result_folder_path)

    elif dataset_name == "delicious-hetrec2011-cold-items":
        dataset = DeliciousHetrec2011ColdItemsReader(result_folder_path)

    elif dataset_name == "lastfm-hetrec2011":
        dataset = LastFMHetrec2011Reader(result_folder_path)

    elif dataset_name == "lastfm-hetrec2011-cold-users":
        dataset = LastFMHetrec2011ColdUsersReader(result_folder_path)

    elif dataset_name == "lastfm-hetrec2011-cold-items":
        dataset = LastFMHetrec2011ColdItemsReader(result_folder_path)

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return

    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_negative = dataset.URM_DICT["URM_negative"].copy()
    UCM_train = dataset.UCM_DICT["UCM"].copy()
    ICM_train = dataset.ICM_DICT["ICM"].copy()

    if dataset_name == "delicious-hetrec2011" or dataset_name == "lastfm-hetrec2011":
        URM_train_last_test = URM_train + URM_validation

        # Ensure IMPLICIT data and disjoint test-train split
        assert_implicit_data([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    else:
        URM_train_last_test = URM_train

        # Ensure IMPLICIT data and disjoint test-train split
        assert_implicit_data([URM_train, URM_test])
        assert_disjoint_matrices([URM_train, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    metric_to_optimize = "MAP"
    cutoff_list_validation = [5, 10, 20]
    cutoff_list_test = [5, 10, 20]

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
            "pretrain_samples": 3,
            "pretrain_batch_size": 200,
            "pretrain_iterations": 5,
            "embed_len": 128,
            "topK": 10,
            "fliter_theta": 16,
            "aggre_theta": 64,
            "batch_size": 400,
            "samples": 3,
            "margin": 20,
            "epochs": 30,
            "iter_without_att": 5,
            "directed": False,
        }

        # Do not modify earlystopping
        earlystopping_hyperparameters = {"validation_every_n": 5,
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
        parameterSearch = SearchSingleCase(HERSWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_train, ICM_train],
            FIT_KEYWORD_ARGS=earlystopping_hyperparameters)

        if dataset_name == "delicious-hetrec2011" or dataset_name == "lastfm-hetrec2011":
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test

            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test=recommender_input_args_last_test,
                                   fit_hyperparameters_values=article_hyperparameters,
                                   output_folder_path=result_folder_path,
                                   output_file_name_root=HERSWrapper.RECOMMENDER_NAME)
        else:
            parameterSearch.search(recommender_input_args,
                                   fit_hyperparameters_values=article_hyperparameters,
                                   output_folder_path=result_folder_path,
                                   output_file_name_root=HERSWrapper.RECOMMENDER_NAME)

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
        ###### Content Baselines

        for ICM_name, ICM_object in dataset.ICM_DICT.items():

            try:

                runParameterSearch_Content(ItemKNNCBFRecommender,
                                           URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test,
                                           output_folder_path=result_folder_path,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           ICM_name=ICM_name,
                                           ICM_object=ICM_object.copy(),
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)

            except Exception as e:

                print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
                traceback.print_exc()

        ################################################################################################
        ###### Hybrid

        for ICM_name, ICM_object in dataset.ICM_DICT.items():

            try:

                runParameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                          URM_train=URM_train,
                                          URM_train_last_test=URM_train_last_test,
                                          metric_to_optimize=metric_to_optimize,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test,
                                          output_folder_path=result_folder_path,
                                          parallelizeKNN=False,
                                          allow_weighting=True,
                                          ICM_name=ICM_name,
                                          ICM_object=ICM_object.copy(),
                                          n_cases=n_cases,
                                          n_random_starts=n_random_starts)

            except Exception as e:

                print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
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
                                          other_algorithm_list=[HERSWrapper],
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
                                  other_algorithm_list=[HERSWrapper],
                                  KNN_similarity_to_report_list=KNN_similarity_to_report_list)

        print_results_latex_table(result_folder_path=result_folder_path,
                                  algorithm_name=ALGORITHM_NAME,
                                  file_name_suffix="all_metrics_",
                                  dataset_name=dataset_name,
                                  metrics_to_report_list=["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST",
                                                          "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                  cutoffs_to_report_list=cutoff_list_validation,
                                  other_algorithm_list=[HERSWrapper],
                                  KNN_similarity_to_report_list=KNN_similarity_to_report_list)


if __name__ == '__main__':

    ALGORITHM_NAME = "hers"
    CONFERENCE_NAME = "aaai19"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune', help="Baseline hyperparameter search", type=bool, default=False)
    parser.add_argument('-a', '--DL_article_default', help="Train the DL model with article hyperparameters", type=bool, default=True)
    parser.add_argument('-p', '--print_results', help="Print results", type=bool, default=False)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = ["delicious-hetrec2011", "delicious-hetrec2011-cold-users", "delicious-hetrec2011-cold-items",
                    "lastfm-hetrec2011", "lastfm-hetrec2011-cold-users", "lastfm-hetrec2011-cold-items"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_DL_article_default=input_flags.DL_article_default,
                                   flag_print_results=input_flags.print_results,
                                   )
