#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.sparse as sps

from Conferences.AAAI.HERS_our_interface.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Conferences.AAAI.HERS_our_interface.split_train_validation import split_data_train_validation_test_negative_user_wise


class DeliciousHetrec2011Reader(object):
    URM_DICT = {}
    UCM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(DeliciousHetrec2011Reader, self).__init__()

        pre_splitted_path += "delicious_data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = "Conferences/AAAI/HERS_github/datasets/book/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("DeliciousHetrec2011Reader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            print("DeliciousHetrec2011Reader: Pre-splitted data not found, building new one")

            ratings_path = original_data_path + "book_rating.txt"
            users_net_path = original_data_path + "book_userNet.txt"
            items_net_path = original_data_path + "book_itemNet.txt"

            all_data = np.loadtxt(ratings_path, dtype=np.int32)
            G_users = np.loadtxt(users_net_path, dtype=np.int32)
            G_items = np.loadtxt(items_net_path, dtype=np.int32)

            users_amount = max(G_users[:, 0])
            items_amount = max(G_items[:, 0])

            user_list_1 = np.asarray(list(G_users[:, 0])) - 1
            user_list_2 = np.asarray(list(G_users[:, 1])) - 1
            interactions = list(np.ones(len(user_list_2)))
            UCM = sps.coo_matrix((interactions, (user_list_1, user_list_2)), shape=(users_amount, users_amount), dtype=np.int32)
            UCM = UCM.tocsr()

            item_list_1 = np.asarray(list(G_items[:, 0])) - 1
            item_list_2 = np.asarray(list(G_items[:, 1])) - 1
            interactions = list(np.ones(len(item_list_2)))
            ICM = sps.coo_matrix((interactions, (item_list_1, item_list_2)), shape=(items_amount, items_amount), dtype=np.int32)
            ICM = ICM.tocsr()

            user_list = np.asarray(list(all_data[:, 0])) - 1
            item_list = np.asarray(list(all_data[:, 1])) - 1
            interactions = list(np.ones(len(item_list)))
            URM_all = sps.coo_matrix((interactions, (user_list, item_list)), shape=(users_amount, items_amount), dtype=np.int32)
            URM_all = URM_all.tocsr()
            URM_train, URM_validation, URM_test, URM_negative = split_data_train_validation_test_negative_user_wise(URM_all, negative_items_per_positive=10)

            self.ICM_DICT = {
                "ICM": ICM,
            }

            self.UCM_DICT = {
                "UCM": UCM,
            }

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_validation": URM_validation,
                "URM_test": URM_test,
                "URM_negative": URM_negative,
            }

            save_data_dict_zip(self.URM_DICT, self.UCM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

            print("DeliciousHetrec2011Reader: loading complete")
