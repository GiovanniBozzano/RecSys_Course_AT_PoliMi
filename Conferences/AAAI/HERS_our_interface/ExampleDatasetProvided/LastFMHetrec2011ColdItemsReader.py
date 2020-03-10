#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os

import numpy as np
import scipy.sparse as sps

from Conferences.AAAI.HERS_our_interface.load_and_save_data import load_data_dict_zip, save_data_dict_zip


class LastFMHetrec2011ColdItemsReader(object):
    URM_DICT = {}
    UCM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(LastFMHetrec2011ColdItemsReader, self).__init__()

        pre_splitted_path += "lastfm_cs_items_data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = "Conferences/AAAI/HERS_github/datasets/lastfm/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("LastFMHetrec2011Reader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            print("LastFMHetrec2011ColdItemsReader: Pre-splitted data not found, building new one")

            users_net_path = original_data_path + "lastfm_userNet.txt"
            items_net_path = original_data_path + "lastfm_itemNet.txt"
            train_path = original_data_path + "lastfm_rating_train_cold_item_reverse.txt"
            test_path = original_data_path + "lastfm_rating_test_cold_item.txt"
            test_path_neg = original_data_path + "lastfm_rating_test_cold_item_neg.txt"

            users_data = np.loadtxt(items_net_path, dtype=np.int32)
            items_data = np.loadtxt(users_net_path, dtype=np.int32)
            train_data = np.loadtxt(train_path, dtype=np.int32)
            test_data = np.loadtxt(test_path, dtype=np.int32)

            users_amount = max(users_data[:, 0])
            items_amount = max(items_data[:, 0])

            user_list_1 = np.asarray(list(users_data[:, 0])) - 1
            user_list_2 = np.asarray(list(users_data[:, 1])) - 1
            interactions = list(np.ones(len(user_list_2)))
            UCM = sps.coo_matrix((interactions, (user_list_1, user_list_2)), shape=(users_amount, users_amount), dtype=np.int32)
            UCM = UCM.tocsr()

            item_list_1 = np.asarray(list(items_data[:, 0])) - 1
            item_list_2 = np.asarray(list(items_data[:, 1])) - 1
            interactions = list(np.ones(len(item_list_2)))
            ICM = sps.coo_matrix((interactions, (item_list_1, item_list_2)), shape=(items_amount, items_amount), dtype=np.int32)
            ICM = ICM.tocsr()

            user_list = np.asarray(list(train_data[:, 0])) - 1
            item_list = np.asarray(list(train_data[:, 1])) - 1
            interactions = list(np.ones(len(item_list)))
            URM_train = sps.coo_matrix((interactions, (user_list, item_list)), shape=(users_amount, items_amount), dtype=np.int32)
            URM_train = URM_train.tocsr()

            user_list = np.asarray(list(test_data[:, 0])) - 1
            item_list = np.asarray(list(test_data[:, 1])) - 1
            interactions = list(np.ones(len(item_list)))
            URM_test = sps.coo_matrix((interactions, (user_list, item_list)), shape=(users_amount, items_amount), dtype=np.int32)
            URM_test = URM_test.tocsr()

            with open(test_path_neg, "r") as f:
                negative_nodes = np.asarray(list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)))
            user_list = []
            item_list = []
            for j in range(len(negative_nodes)):
                user_array = np.empty(len(negative_nodes[j][1:]))
                user_array[:] = negative_nodes[j][0]
                user_list = np.concatenate((user_list, user_array))
                item_list = np.concatenate((item_list, negative_nodes[j][1:]))
            user_list = user_list - 1
            item_list = item_list - 1
            interactions = list(np.ones(len(item_list)))
            URM_negative = sps.coo_matrix((interactions, (user_list, item_list)), shape=(users_amount, items_amount), dtype=np.int32)
            URM_negative = URM_negative.tocsr()

            self.ICM_DICT = {
                "ICM": ICM,
            }

            self.UCM_DICT = {
                "UCM": UCM,
            }

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_validation": URM_test,
                "URM_test": URM_test,
                "URM_negative": URM_negative,
            }

            save_data_dict_zip(self.URM_DICT, self.UCM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

            print("LastFMHetrec2011ColdItemsReader: loading complete")
