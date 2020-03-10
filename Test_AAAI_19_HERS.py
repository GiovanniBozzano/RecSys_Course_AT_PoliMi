#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import unittest

import numpy as np
import scipy.sparse as sps

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
from Conferences.AAAI.HERS_our_interface.ExampleDatasetProvided.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from Conferences.AAAI.HERS_our_interface.HERSWrapper import HERSWrapper


class MyTestCase(unittest.TestCase):

    def test_compute_item_score(self):
        recommender_instance, URM_train, UCM_train, ICM_train, URM_test, URM_negative = get_data_and_rec_instance(HERSWrapper)

        n_users, n_items = URM_train.shape

        recommender_instance.fit()

        users_to_evaluate_mask = np.zeros(n_users, dtype=np.bool)
        rows = URM_test.indptr
        num_ratings = np.ediff1d(rows)
        new_mask = num_ratings > 0
        users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)
        user_id_list = np.arange(n_users, dtype=np.int)[users_to_evaluate_mask]

        URM_items_to_rank = sps.csr_matrix(URM_test.copy().astype(np.bool)) + sps.csr_matrix(URM_negative.copy().astype(np.bool))
        URM_items_to_rank.eliminate_zeros()
        URM_items_to_rank.data = np.ones_like(URM_items_to_rank.data)

        for test_user in user_id_list:
            start_pos = URM_items_to_rank.indptr[test_user]
            end_pos = URM_items_to_rank.indptr[test_user + 1]
            items_to_compute = URM_items_to_rank.indices[start_pos:end_pos]
            item_scores = recommender_instance._compute_item_score(user_id_array=np.atleast_1d(test_user),
                                                                   items_to_compute=items_to_compute)

            self.assertEqual(item_scores.shape, (1, n_items), "item_scores shape not correct, contains more users than in user_id_array")
            self.assertFalse(np.any(np.isnan(item_scores)), "item_scores contains np.nan values")

            # Check items not in list have a score of -np.inf
            item_id_not_to_compute = np.ones(n_items, dtype=np.bool)
            item_id_not_to_compute[items_to_compute] = False

            self.assertTrue(np.all(np.isneginf(item_scores[:, item_id_not_to_compute])), "item_scores contains scores for items that should not be computed")

    def test_save_and_load(self):
        recommender_class = HERSWrapper

        recommender_instance_original, URM_train, UCM_train, ICM_train, URM_test, URM_negative = get_data_and_rec_instance(recommender_class)
        n_users, n_items = URM_train.shape

        evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_negative, [50], exclude_seen=True)

        folder_path = "./temp_folder/"
        file_name = "temp_file"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        recommender_instance_original.fit()
        recommender_instance_original.save_model(folder_path=folder_path, file_name=file_name)

        results_run_original, _ = evaluator_test.evaluateRecommender(recommender_instance_original)

        recommender_instance_loaded = recommender_class(URM_train, UCM_train, ICM_train)
        recommender_instance_loaded.load_model(folder_path=folder_path, file_name=file_name)

        results_run_loaded, _ = evaluator_test.evaluateRecommender(recommender_instance_loaded)

        print("Result original: {}\n".format(results_run_original))
        print("Result loaded: {}\n".format(results_run_loaded))

        users_to_evaluate_mask = np.zeros(n_users, dtype=np.bool)
        rows = URM_test.indptr
        num_ratings = np.ediff1d(rows)
        new_mask = num_ratings > 0
        users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)
        user_id_list = np.arange(n_users, dtype=np.int)[users_to_evaluate_mask]

        URM_items_to_rank = sps.csr_matrix(URM_test.copy().astype(np.bool)) + sps.csr_matrix(URM_negative.copy().astype(np.bool))
        URM_items_to_rank.eliminate_zeros()
        URM_items_to_rank.data = np.ones_like(URM_items_to_rank.data)

        for test_user in user_id_list:
            start_pos = URM_items_to_rank.indptr[test_user]
            end_pos = URM_items_to_rank.indptr[test_user + 1]
            items_to_compute = URM_items_to_rank.indices[start_pos:end_pos]

            item_scores_original = recommender_instance_original._compute_item_score(user_id_array=np.atleast_1d(test_user),
                                                                                     items_to_compute=items_to_compute)

            item_scores_loaded = recommender_instance_loaded._compute_item_score(user_id_array=np.atleast_1d(test_user),
                                                                                 items_to_compute=items_to_compute)

            self.assertTrue(np.allclose(item_scores_original, item_scores_loaded), "item_scores of the fitted model and of the loaded model are different")

        shutil.rmtree(folder_path, ignore_errors=True)


def get_data_and_rec_instance(recommender_class):
    result_folder_path = "temp_folder/"

    dataset = LastFMHetrec2011Reader(result_folder_path)
    shutil.rmtree(result_folder_path, ignore_errors=True)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_negative = dataset.URM_DICT["URM_negative"].copy()
    UCM_train = dataset.UCM_DICT["UCM"].copy()
    ICM_train = dataset.ICM_DICT["ICM"].copy()

    URM_train = URM_train + URM_validation

    recommender_instance = recommender_class(URM_train, UCM_train, ICM_train)

    return recommender_instance, URM_train, UCM_train, ICM_train, URM_test, URM_negative


if __name__ == '__main__':
    unittest.main()
