import keras
from keras import backend as K
from keras.layers import LSTM, Embedding, Dot, Lambda, Concatenate, GlobalMaxPooling1D
from keras.models import Input, Model

import layers.accuracy_sim_models as sim
import losses.hinge
from layers.custom_layer import CosineSim
from rp_models.similarity.sim_words_names import SimilarityModelWordsNames


class LSTMWordsNames(SimilarityModelWordsNames):

    def create_model(self, lr, units_l, units_q, train_emb):

        question = Input(shape=(self._max_seq_len,))
        pos_label = Input(shape=(self._max_label_words_len,))
        neg_label = Input(shape=(self._max_label_words_len,))
        pos_label_names = Input(shape=(self._max_label_names_len,))
        neg_label_names = Input(shape=(self._max_label_names_len,))
        all_label = Input(shape=(None, self._max_label_words_len,))
        all_label_names = Input(shape=(None, self._max_label_names_len,))

        all_label_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0] * K.shape(all_label)[1], self._max_label_words_len)))(
            all_label)
        all_label_names_rshp = Lambda(lambda x: K.reshape(x, shape=(
            K.shape(all_label_names)[0] * K.shape(all_label_names)[1], self._max_label_names_len)))(all_label_names)

        shared_emb = Embedding(self._nb_words, self._embed_dim,
                               weights=[self._embedding_matrix], mask_zero=False, trainable=train_emb)

        question_emb = shared_emb(question)
        pos_label_emb = shared_emb(pos_label)
        neg_label_emb = shared_emb(neg_label)
        all_label_emb = shared_emb(all_label_rshp)
        pos_label_names_emb = shared_emb(pos_label_names)
        neg_label_names_emb = shared_emb(neg_label_names)
        all_label_names_emb = shared_emb(all_label_names_rshp)

        shared_question_encoder = LSTM(units=units_q, return_sequences=True)
        shared_label_encoder = LSTM(units=units_l, return_sequences=True)

        enc_question = shared_question_encoder(question_emb)
        enc_pos_label = shared_label_encoder(pos_label_emb)
        enc_neg_label = shared_label_encoder(neg_label_emb)
        enc_all_label = shared_label_encoder(all_label_emb)

        enc_pos_label_names = shared_label_encoder(pos_label_names_emb)
        enc_neg_label_names = shared_label_encoder(neg_label_names_emb)
        enc_all_label_names = shared_label_encoder(all_label_names_emb)

        enc_question = GlobalMaxPooling1D()(enc_question)

        enc_pos_label_name_word = Concatenate(axis=1)([enc_pos_label_names, enc_pos_label])
        print(enc_pos_label_name_word)
        print(enc_pos_label)
        enc_pos_label_name_word = GlobalMaxPooling1D()(
            enc_pos_label_name_word)  # GlobalAveragePooling1D()(enc_pos_label_name_word)#

        enc_neg_label_name_word = Concatenate(axis=1)([enc_neg_label_names, enc_neg_label])
        enc_neg_label_name_word = GlobalMaxPooling1D()(
            enc_neg_label_name_word)  # GlobalAveragePooling1D()(enc_neg_label_name_word)#

        enc_all_label_name_word = Concatenate(axis=1)([enc_all_label_names, enc_all_label])
        enc_all_label_name_word = GlobalMaxPooling1D()(
            enc_all_label_name_word)  # GlobalAveragePooling1D()(enc_all_label_name_word)#

        enc_all_names_words_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0], K.shape(all_label)[1], units_q)))(
            enc_all_label_name_word)

        ### Similarity ###
        distance_pos = Dot(axes=1, normalize=True)([enc_question, enc_pos_label_name_word])
        distance_neg = Dot(axes=1, normalize=True)([enc_question, enc_neg_label_name_word])
        distance_all = CosineSim()([enc_question, enc_all_names_words_rshp])

        #    # other way to compute distancce_all
        #    # results in the exact same results
        #    enc_question = Lambda(lambda x: K.expand_dims(x,axis=1))(enc_question)
        #    distance_all = Dot(axes=2,normalize=True)([enc_question, enc_all_names_words_rshp])
        #    distance_all = Lambda(lambda x: K.sum(x,axis=1))(distance_all)

        ### loss and accuracy ###
        ranking_loss = losses.hinge.hinge_loss(distance_pos, distance_neg)
        acc = sim.accuracy(distance_pos, distance_neg)

        model = Model([question, pos_label, neg_label, all_label, pos_label_names, neg_label_names, all_label_names],
                      distance_all)
        adam = keras.optimizers.Adam(lr=lr)
        model.compile(loss=ranking_loss, optimizer=adam, metrics=[acc])

        self.model = model

    def grid_search(self, x_train, y_train, pos_train, neg_train, pos_name_level_train, neg_name_level_train,
                    x_valid, y_valid, pos_valid, neg_valid, pos_name_level_valid, neg_name_level_valid,
                    predicates_per_question_valid, param_grid):

        for batch in param_grid['batch_size']:
            for epoch in param_grid['epoch']:
                for units_q in param_grid['units_q']:
                    for units_l in param_grid['units_l']:
                        for train_emb in param_grid['train_emb']:
                            for lr in param_grid['lr']:
                                print('Batch: ', batch, ' epoch: ', epoch, ' units_q: ', units_q, ' units_l: ', units_l,
                                      ' train_emb: ', train_emb, ' lr: ', lr)

                                self.create_model(lr=lr, units_l=units_l, units_q=units_q, train_emb=train_emb)
                                self.train(x_train, y_train, pos_train, neg_train, pos_name_level_train,
                                           neg_name_level_train,
                                           x_valid, y_valid, pos_valid, neg_valid, pos_name_level_valid,
                                           neg_name_level_valid, epoch, batch)

                                r = self.evaluate_wrt_sbj_entity(x_valid, y_valid, pos_valid, neg_valid,
                                                                 pos_name_level_valid, neg_name_level_valid,
                                                                 predicates_per_question_valid)
                                self.clear()
