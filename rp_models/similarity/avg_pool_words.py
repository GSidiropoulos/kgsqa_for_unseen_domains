import keras
from keras import backend as K
from keras.layers import Embedding, AveragePooling1D, GlobalMaxPooling1D, Lambda, \
    Dot, Concatenate, Dense
from keras.models import Input, Model

import layers.accuracy_sim_models as sim
import losses.hinge
from layers.custom_layer import CosineSim
from rp_models.similarity.sim_words import SimilarityModelWords


class AvgPoolingWords(SimilarityModelWords):

    def create_model(self, pool_size_l=3, pool_size_q=3, optimizer='adam', metric='cos', lr=0.001):
        question = Input(shape=(self._max_seq_len,))

        pos_label = Input(shape=(self._max_label_words_len,))
        neg_label = Input(shape=(self._max_label_words_len,))
        all_label = Input(shape=(None, self._max_label_words_len,))

        all_label_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0] * K.shape(all_label)[1], self._max_label_words_len)))(
            all_label)

        shared_emb = Embedding(self._nb_words, self._embed_dim,
                               weights=[self._embedding_matrix], mask_zero=False, trainable=True)

        shared_label_encoder = AveragePooling1D(pool_size=pool_size_l, padding='same')  # GlobalMaxPooling1D()#
        shared_question_encoder = AveragePooling1D(pool_size=pool_size_q, padding='same')

        # define metric layer if metric is mlp
        if metric == 'mlp':
            metric_layer_1 = Dense(units=128, activation='softplus')
            metric_layer_2 = Dense(units=64, activation='softplus')
            metric_layer_3 = Dense(units=1, activation='softplus')

        question_emb = shared_emb(question)
        pos_label_emb = shared_emb(pos_label)
        neg_label_emb = shared_emb(neg_label)
        all_label_emb = shared_emb(all_label_rshp)

        enc_question = shared_question_encoder(question_emb)
        enc_pos_label = shared_label_encoder(pos_label_emb)
        enc_neg_label = shared_label_encoder(neg_label_emb)
        enc_all_label = shared_label_encoder(all_label_emb)

        enc_question = GlobalMaxPooling1D()(enc_question)
        enc_pos_label = GlobalMaxPooling1D()(enc_pos_label)
        enc_neg_label = GlobalMaxPooling1D()(enc_neg_label)
        enc_all_label = GlobalMaxPooling1D()(enc_all_label)

        enc_all_label_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0], K.shape(all_label)[1], self._embed_dim)))(
            enc_all_label)

        if metric == 'cos':
            distance_pos = Dot(axes=1, normalize=True)([enc_question, enc_pos_label])
            distance_neg = Dot(axes=1, normalize=True)([enc_question, enc_neg_label])
            distance_all = CosineSim()([enc_question, enc_all_label_rshp])
        elif metric == 'mlp':
            distance_pos = Concatenate(axis=-1)([enc_question, enc_pos_label])
            distance_pos = metric_layer_1(distance_pos)
            distance_pos = metric_layer_2(distance_pos)
            distance_pos = metric_layer_3(distance_pos)

            distance_neg = Concatenate(axis=-1)([enc_question, enc_neg_label])
            distance_neg = metric_layer_1(distance_neg)
            distance_neg = metric_layer_2(distance_neg)
            distance_neg = metric_layer_3(distance_neg)

            enc_question_rshp = Lambda(lambda x: K.repeat(x, K.shape(all_label)[1]))(enc_question)
            enc_l_q_conc = Concatenate(axis=-1)([enc_question_rshp, enc_all_label_rshp])
            enc_l_q_conc_rshp = Lambda(
                lambda x: K.reshape(x, shape=(K.shape(all_label)[0] * K.shape(all_label)[1], 2 * self._embed_dim)))(
                enc_l_q_conc)

            distance_all = metric_layer_1(enc_l_q_conc_rshp)
            distance_all = metric_layer_2(distance_all)
            distance_all = metric_layer_3(distance_all)
            distance_all = Lambda(lambda x: K.reshape(x, shape=(K.shape(all_label)[0], K.shape(all_label)[1])))(
                distance_all)

        ranking_loss = losses.hinge.hinge_loss(distance_pos, distance_neg)
        acc = sim.accuracy(distance_pos, distance_neg)

        model = Model([question, pos_label, neg_label, all_label], distance_all)
        adam = keras.optimizers.Adam(lr=lr)

        model.compile(loss=ranking_loss, optimizer=adam, metrics=[acc])

        self.model = model

    def grid_search(self, x_train, y_train, pos_train, neg_train, x_valid, y_valid, pos_valid, neg_valid,
                    predicates_per_question_valid, param_grid):
        models = dict()
        for opt in param_grid['optimizer']:
            for lr in param_grid['lr']:
                for pool_size_l in param_grid['pool_size_l']:
                    for pool_size_q in param_grid['pool_size_q']:
                        for epoch in param_grid['epoch']:
                            for batch in param_grid['batch_size']:
                                print('Optimizer: ', opt, ' Pool size_q: ', pool_size_q, ' Pool size_l: ', pool_size_l,
                                      ' Epochs: ', epoch, 'Batch: ', batch, 'Learning rate: ', lr)
                                self.create_model(pool_size_l=pool_size_l, pool_size_q=pool_size_q, optimizer=opt,
                                                  metric='cos', lr=lr)
                                self.train(x_train, y_train, pos_train, neg_train, x_valid, y_valid, pos_valid,
                                           neg_valid, epoch, batch)

                                pred = self.evaluate_test(x_valid, y_valid, pos_valid, neg_valid)
                                res = self.relation_pred_wrt_cand(pred, y_valid, predicates_per_question_valid)
                                #              print('Result with constraint: ', res)
                                #              models[opt+str(pool_size)+str(epoch)] = self.model
                                self.clear()

        return models
