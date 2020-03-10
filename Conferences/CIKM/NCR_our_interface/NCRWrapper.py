#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop, Adagrad

from Base.BaseRecommender import BaseRecommender as BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.DataIO import DataIO
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Conferences.CIKM.NCR_github.NeuPR import get_model, get_train_instances


class NCRWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):
    RECOMMENDER_NAME = "NCRWrapper"

    def __init__(self, URM_train, verbose=True):

        super(NCRWrapper, self).__init__(URM_train, verbose=verbose)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices

        for user_index in range(len(user_id_array)):
            negative_items = item_indices[1:].astype(np.int32)
            positive_items = np.full(len(negative_items), item_indices[:1], dtype=np.int32)

            users = np.full(len(negative_items), user_id_array[user_index], dtype=np.int32)
            prediction_1 = self.model.predict([users, positive_items, negative_items], batch_size=101, verbose=0)
            prediction_2 = self.model.predict([users, negative_items, positive_items], batch_size=101, verbose=0)
            prediction = prediction_1 - prediction_2
            prediction = - prediction
            prediction = np.insert(prediction, 0, 0)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = prediction.ravel()
            else:
                item_scores[user_index, :] = prediction.ravel()

        return item_scores

    def fit(self,

            epochs=0,
            batch_size=256,
            num_factors=8,
            layers=(32, 32, 16, 8),
            reg_mf=0,
            reg_layers=(0, 0, 0, 0),
            num_negatives=2,
            learning_rate=0.0005,
            learner='adam',
            k=2,

            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        # The following code contains various operations needed by another wrapper

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_factors = num_factors
        self.layers = layers
        self.reg_mf = reg_mf
        self.reg_layers = reg_layers
        self.num_negatives = num_negatives
        self.learning_rate = learning_rate
        self.learner = learner
        self.k = k

        keras.backend.clear_session()

        self.model = get_model(self.n_users, self.n_items, num_factors, [e * k for e in layers], reg_layers, reg_mf)
        if learner.lower() == "adagrad":
            self.model.compile(optimizer=Adagrad(lr=learning_rate), loss="binary_crossentropy")
        elif learner.lower() == "rmsprop":
            self.model.compile(optimizer=RMSprop(lr=learning_rate), loss="binary_crossentropy")
        elif learner.lower() == "adam":
            self.model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
        else:
            self.model.compile(optimizer=SGD(lr=learning_rate), loss="binary_crossentropy")

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Tranining complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        user_input, item_input_pos, item_input_neg, labels = get_train_instances(self.URM_train.todok(), self.num_negatives)
        training = self.model.fit([np.array(user_input, dtype=np.int32), np.array(item_input_pos, dtype=np.int32), np.array(item_input_neg, dtype=np.int32)],
                                  np.array(labels), batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
        print("Loss: ", training.history['loss'][0])

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.save_weights(folder_path + file_name + "_weights")

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_factors": self.num_factors,
            "layers": self.layers,
            "reg_mf": self.reg_mf,
            "reg_layers": self.reg_layers,
            "num_negatives": self.num_negatives,
            "learning_rate": self.learning_rate,
            "learner": self.learner,
            "k": self.k,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        self.model = get_model(self.n_users, self.n_items, self.num_factors, [e * self.k for e in self.layers], self.reg_layers, self.reg_mf)
        if self.learner.lower() == "adagrad":
            self.model.compile(optimizer=Adagrad(lr=self.learning_rate), loss="binary_crossentropy")
        elif self.learner.lower() == "rmsprop":
            self.model.compile(optimizer=RMSprop(lr=self.learning_rate), loss="binary_crossentropy")
        elif self.learner.lower() == "adam":
            self.model.compile(optimizer=Adam(lr=self.learning_rate), loss="binary_crossentropy")
        else:
            self.model.compile(optimizer=SGD(lr=self.learning_rate), loss="binary_crossentropy")
        self.model.load_weights(folder_path + file_name + "_weights")

        self._print("Loading complete")
