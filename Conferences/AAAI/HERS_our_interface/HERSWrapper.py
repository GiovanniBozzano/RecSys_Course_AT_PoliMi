#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import keras
import networkx as nx
import numpy as np
from keras.regularizers import l2
from sklearn.utils import shuffle

from Base.BaseRecommender import BaseRecommender as BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.DataIO import DataIO
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
from Conferences.AAAI.HERS_github.model.RSbatch import TripletGenerator
from Conferences.AAAI.HERS_github.model.losses import max_margin_loss
from Conferences.AAAI.HERS_github.model.mlmr import mlmf
from Conferences.AAAI.HERS_github.model.scorer import inner_prod_scoremodel
from Conferences.AAAI.HERS_github.model.socialRC import score_connection
from Conferences.AAAI.HERS_github.model.srs_model import NetworkRS


class HERSWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):
    RECOMMENDER_NAME = "HERSWrapper"

    def __init__(self, URM_train, UCM_train, ICM_train, verbose=True):

        super(HERSWrapper, self).__init__(URM_train, verbose=verbose)

        assert self.n_users == UCM_train.shape[0], "{}: URM_train has {} users but UCM_train has {}".format(self.RECOMMENDER_NAME, self.n_users, UCM_train.shape[0])

        self.UCM_train = check_matrix(UCM_train.copy(), 'csr', dtype=np.float32)
        self.UCM_train.eliminate_zeros()

        self._cold_user_CBF_mask = np.ediff1d(self.UCM_train.indptr) == 0

        if self._cold_user_CBF_mask.any():
            print("{}: UCM Detected {} ({:.2f} %) cold users.".format(
                self.RECOMMENDER_NAME, self._cold_user_CBF_mask.sum(), self._cold_user_CBF_mask.sum() / self.n_users * 100))

        assert self.n_items == ICM_train.shape[0], "{}: URM_train has {} items but ICM_train has {}".format(self.RECOMMENDER_NAME, self.n_items, ICM_train.shape[0])

        self.ICM_train = check_matrix(ICM_train.copy(), 'csr', dtype=np.float32)
        self.ICM_train.eliminate_zeros()

        self._cold_item_CBF_mask = np.ediff1d(self.ICM_train.indptr) == 0

        if self._cold_item_CBF_mask.any():
            print("{}: ICM Detected {} ({:.2f} %) items with no features.".format(
                self.RECOMMENDER_NAME, self._cold_item_CBF_mask.sum(), self._cold_item_CBF_mask.sum() / self.n_items * 100))

        self.G_ui = np.swapaxes(np.asarray(self.URM_train.nonzero(), dtype=np.int32), 0, 1)
        self.G_user = nx.convert_matrix.from_scipy_sparse_matrix(self.UCM_train, create_using=nx.DiGraph())
        self.G_user = self.G_user.to_undirected()
        self.G_user.remove_nodes_from(list(nx.isolates(self.G_user)))

        self.G_item = nx.convert_matrix.from_scipy_sparse_matrix(self.ICM_train, create_using=nx.DiGraph())
        self.G_item = self.G_item.to_undirected()
        self.G_item.remove_nodes_from(list(nx.isolates(self.G_item)))

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

    def _get_cold_user_mask(self):
        return self._cold_user_CBF_mask

    def _get_cold_item_mask(self):
        return self._cold_item_CBF_mask

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices

        for user_index in range(len(user_id_array)):
            from_rep = self.user_rep[None, user_id_array[user_index] - 1]
            to_rep = self.item_rep[item_indices - 1]
            item_score_user = score_connection(np.repeat(from_rep, to_rep.shape[0], axis=0), to_rep, self.model.score_model)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores

    def fit(self,

            pretrain_samples=3,
            pretrain_batch_size=200,
            pretrain_iterations=0,
            embed_len=128,
            topK=10,
            fliter_theta=16,
            aggre_theta=64,
            batch_size=400,
            samples=3,
            margin=20,
            epochs=30,
            iter_without_att=5,
            directed=False,

            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        # The following code contains various operations needed by another wrapper

        self.embed_len = embed_len
        self.topK = topK
        self.fliter_theta = fliter_theta
        self.aggre_theta = aggre_theta
        self.batch_size = batch_size
        self.samples = samples
        self.margin = margin
        self.epochs = epochs
        self.iter_without_att = iter_without_att
        self.directed = directed

        self.pretrain_model(pretrain_samples,
                            pretrain_batch_size,
                            pretrain_iterations,
                            margin)

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        self.item_rep = self.get_item_rep()
        self.user_rep = self.get_user_rep()

        print("{}: Tranining complete".format(self.RECOMMENDER_NAME))

    def pretrain_model(self,
                       pretrain_samples,
                       pretrain_batch_size,
                       pretrain_iterations,
                       margin):

        keras.backend.clear_session()

        loss = max_margin_loss

        # score_model = nn_scoremodel((embed_len,), embed_len, score_act=None)
        score_model = inner_prod_scoremodel((self.embed_len,), score_rep_norm=False)
        # score_model = fm_scoremodel((embed_len,), score_rep_norm=False, score_act=None)

        pretrain_model = mlmf(nb_user=self.n_users + 1, nb_item=self.n_items + 1, embed_dim=self.embed_len, score_model=score_model, reg=l2(1e-7))
        pretrain_model.contrast_model.compile(loss=loss, optimizer='adam')

        self.edges = self.G_ui
        num_edges = len(self.edges)

        pretrain_batch_num = math.ceil(num_edges / pretrain_batch_size)

        for i in range(pretrain_iterations):
            shuffle(self.edges)
            train_loss = 0
            for s in range(pretrain_samples):
                for j in range(pretrain_batch_num):
                    edge_batch = np.array(self.edges[j * pretrain_batch_size:min(num_edges, (j + 1) * pretrain_batch_size)])
                    batch_node_array, positive_batch_array, negative_batch_array = \
                        (edge_batch[:, 0], edge_batch[:, 1], np.random.randint(low=1, high=self.n_items, size=len(edge_batch)))
                    train_loss_temp = pretrain_model.contrast_model.train_on_batch(
                        x=[batch_node_array, positive_batch_array, negative_batch_array, ],
                        y=margin * np.ones([len(edge_batch)])
                    )
                    train_loss += train_loss_temp

                print("[#pretrain_epoch=%d/%d], sample=%d" % (i + 1, pretrain_iterations, s + 1))
            print("[#pretrain_epoch=%d/%d] ended, loss=%.5f" % (i + 1, pretrain_iterations, train_loss / (pretrain_batch_num * pretrain_samples)))

        self.model = NetworkRS(self.n_users, self.n_items, self.embed_len, score_model,
                               self.topK, self.topK, embed_regularizer=l2(5e-7), directed=self.directed,
                               mem_filt_alpha=self.fliter_theta, mem_agg_alpha=self.aggre_theta,
                               user_mask=None)
        self.model.triplet_model.compile(loss=loss, optimizer='adam')
        self.model.user_embed.set_weights(pretrain_model.user_emb.get_weights())
        self.model.item_embed.set_weights(pretrain_model.item_emb.get_weights())

        self.batchGenerator = TripletGenerator(self.G_user, self.model, self.G_ui, self.G_item)

    def _prepare_model_for_validation(self):
        self.item_rep = self.get_item_rep()
        self.user_rep = self.get_user_rep()

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        self.edges = shuffle(self.edges)
        train_loss = 0

        num_edges = len(self.edges)
        batch_num = math.ceil(num_edges / self.batch_size)

        spl = self.samples if currentEpoch < self.iter_without_att else self.samples
        for s in range(spl):
            for j in range(batch_num):
                edge_batch = self.edges[j * self.batch_size:min(num_edges, (j + 1) * self.batch_size)]
                batch_node, positive_batch, negative_batch, first_batch_data, second_batch_data, positive_first_batch, negative_first_batch = \
                    self.batchGenerator.generate_triplet_batch(edge_batch=edge_batch, topK=self.topK, attention_sampling=currentEpoch >= self.iter_without_att)

                batch_node_array = np.asarray(batch_node)
                positive_batch_array = np.asarray(positive_batch)
                negative_batch_array = np.asarray(negative_batch)
                train_loss_temp = self.model.triplet_model.train_on_batch(
                    x=[batch_node_array, first_batch_data, second_batch_data,
                       positive_batch_array, positive_first_batch,
                       negative_batch_array, negative_first_batch],
                    y=self.margin * np.ones((len(batch_node),))
                )
                train_loss += train_loss_temp

                if (j + 1) % 100 == 0:
                    print("[#epoch=%d/%d], batch %d/%d, sample=%d" % (currentEpoch + 1, self.epochs, j + 1, batch_num, s + 1))
        print("[#epoch=%d/%d] ended, loss=%.5f" % (currentEpoch + 1, self.epochs, train_loss / (batch_num * spl)))

        self.batchGenerator.clear_node_cache()

    def get_user_rep(self):
        node_size = self.G_user.number_of_nodes()
        memory_output = np.zeros((node_size + 1, self.embed_len))

        node_list = list(self.G_user.nodes())
        num_node = len(node_list)
        nb_batch = math.ceil(len(node_list) / self.batch_size)
        for j in range(nb_batch):
            batch_node = node_list[j * self.batch_size:min(num_node, (j + 1) * self.batch_size)]
            first_batch_data, second_batch_data = self.batchGenerator.get_batch_data_topk(batch_node=batch_node, topK=self.topK)
            memory_out = self.model.user_model.predict_on_batch([np.array(batch_node), first_batch_data, second_batch_data])
            memory_output[batch_node, :] = memory_out

        return memory_output[1:]

    def get_item_rep(self):
        node_list = list(self.G_item.nodes())
        node_size = len(node_list)
        memory_output = np.zeros((node_size + 1, self.embed_len))
        nb_batch = math.ceil(len(node_list) / self.batch_size)

        for j in range(nb_batch):
            batch_node = node_list[j * self.batch_size:min(node_size, (j + 1) * self.batch_size)]
            first_batch_data, _ = self.batchGenerator.itemGenerate.get_batch_data_topk(batch_node=batch_node, topK=self.topK, predict_batch_size=100)

            memory_out = self.model.item_model.predict_on_batch([np.array(batch_node), first_batch_data])
            memory_output[batch_node, :] = memory_out

        return memory_output[1:]

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.first_model.save_weights(folder_path + file_name + "_first_model_weights")
        self.model.second_model.save_weights(folder_path + file_name + "_second_model_weights")
        self.model.triplet_model.save_weights(folder_path + file_name + "_triplet_model_weights")

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embed_len": self.embed_len,
            "topK": self.topK,
            "fliter_theta": self.fliter_theta,
            "aggre_theta": self.aggre_theta,
            "batch_size": self.batch_size,
            "samples": self.samples,
            "margin": self.margin,
            "epochs": self.epochs,
            "iter_without_att": self.iter_without_att,
            "directed": self.directed,
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

        # score_model = nn_scoremodel((embed_len,), embed_len, score_act=None)
        score_model = inner_prod_scoremodel((self.embed_len,), score_rep_norm=False)
        # score_model = fm_scoremodel((embed_len,), score_rep_norm=False, score_act=None)

        loss = max_margin_loss

        self.model = NetworkRS(self.n_users, self.n_items, self.embed_len, score_model,
                               self.topK, self.topK, embed_regularizer=l2(5e-7), directed=self.directed,
                               mem_filt_alpha=self.fliter_theta, mem_agg_alpha=self.aggre_theta,
                               user_mask=None)

        self.model.first_model.load_weights(folder_path + file_name + "_first_model_weights")
        self.model.second_model.load_weights(folder_path + file_name + "_second_model_weights")
        self.model.triplet_model.load_weights(folder_path + file_name + "_triplet_model_weights")

        self.model.triplet_model.compile(loss=loss, optimizer='adam')

        self.batchGenerator = TripletGenerator(self.G_user, self.model, self.G_ui, self.G_item)

        self.item_rep = self.get_item_rep()
        self.user_rep = self.get_user_rep()

        self._print("Loading complete")
