#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.sparse as sps

from Conferences.CIKM.NCR_github.DataSet import DataSet
from Data_manager.load_and_save_data import load_data_dict_zip, save_data_dict_zip


class Movielens1MReader(object):
    URM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(Movielens1MReader, self).__init__()

        pre_splitted_path += "movielens1m_data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = "Conferences/CIKM/NCR_github/data/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Movielens1MReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            print("Movielens1MReader: Pre-splitted data not found, building new one")

            dataset = DataSet()
            dataset.loadClicks(original_data_path + 'ml1m.txt', 10, 10)
            train_data, validation_data, test_data, negative_data = dataset.trainMatrix, dataset.validRatings, dataset.testRatings, dataset.testNegatives

            URM_train = train_data.tocsr()

            user_list = [pair[0] for pair in validation_data]
            item_list = [pair[1] for pair in validation_data]
            interactions = list(np.ones(len(item_list)))
            URM_validation = sps.coo_matrix((interactions, (user_list, item_list)), shape=(dataset.nUsers, dataset.nItems), dtype=np.int32)
            URM_validation = URM_validation.tocsr()

            user_list = [pair[0] for pair in test_data]
            item_list = [pair[1] for pair in test_data]
            interactions = list(np.ones(len(item_list)))
            URM_test = sps.coo_matrix((interactions, (user_list, item_list)), shape=(dataset.nUsers, dataset.nItems), dtype=np.int32)
            URM_test = URM_test.tocsr()

            user_list = np.concatenate([np.full(len(items), i) for i, items in enumerate(negative_data)])
            item_list = [item for items in negative_data for item in items]
            interactions = list(np.ones(len(item_list)))
            URM_negative = sps.coo_matrix((interactions, (user_list, item_list)), shape=(dataset.nUsers, dataset.nItems), dtype=np.int32)
            URM_negative = URM_negative.tocsr()

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_validation": URM_validation,
                "URM_test": URM_test,
                "URM_negative": URM_negative,
            }

            save_data_dict_zip(self.URM_DICT, {}, pre_splitted_path, pre_splitted_filename)

            print("Movielens1MReader: loading complete")
