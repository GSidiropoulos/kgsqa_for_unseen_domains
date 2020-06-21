from keras import backend as K
from keras.layers import LSTM, Embedding, Dot, Lambda
from keras.models import Input, Model

import layers.accuracy_sim_models as sim
import losses.hinge
from layers.custom_layer import CosineSim
from rp_models.similarity.sim_words import SimilarityModelWords


class LSTMWords(SimilarityModelWords):

    def create_model(self, lr, units_l, units_q, train_emb):
        question = Input(shape=(self._max_seq_len,))

        pos_label = Input(shape=(self._max_label_words_len,))
        neg_label = Input(shape=(self._max_label_words_len,))
        all_label = Input(shape=(None, self._max_label_words_len,))

        all_label_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0] * K.shape(all_label)[1], self._max_label_words_len)))(
            all_label)

        shared_emb = Embedding(self._nb_words, self._embed_dim,
                               weights=[self._embedding_matrix], mask_zero=True, trainable=train_emb)

        question_emb = shared_emb(question)
        pos_label_emb = shared_emb(pos_label)
        neg_label_emb = shared_emb(neg_label)
        all_label_emb = shared_emb(all_label_rshp)

        shared_question_encoder = LSTM(units=units_q, return_state=True)
        shared_label_encoder = LSTM(units=units_l, return_state=True)

        _, enc_question, _ = shared_question_encoder(question_emb)
        _, enc_pos_label, _ = shared_label_encoder(pos_label_emb)
        _, enc_neg_label, _ = shared_label_encoder(neg_label_emb)
        _, enc_all_label, _ = shared_label_encoder(all_label_emb)

        enc_all_label_rshp = Lambda(
            lambda x: K.reshape(x, shape=(K.shape(all_label)[0], K.shape(all_label)[1], units_q)))(enc_all_label)

        distance_pos = Dot(axes=1, normalize=True)([enc_question, enc_pos_label])
        distance_neg = Dot(axes=1, normalize=True)([enc_question, enc_neg_label])
        distance_all = CosineSim()([enc_question, enc_all_label_rshp])
        ranking_loss = losses.hinge.hinge_loss(distance_pos, distance_neg)
        acc = sim.accuracy(distance_pos, distance_neg)

        model = Model([question, pos_label, neg_label, all_label], distance_all)

        model.compile(loss=ranking_loss, optimizer='adam', metrics=[acc])

        self.model = model

    def grid_search(self, x_train, y_train, pos_train, neg_train, x_valid, y_valid, pos_valid, neg_valid,
                    predicates_per_question_valid,
                    x_test, y_test, pos_test, neg_test, predicates_per_question_test, param_grid):
        models = dict()
        for batch in param_grid['batch_size']:
            for epoch in param_grid['epoch']:
                for units_q in param_grid['units_q']:
                    for units_l in param_grid['units_l']:
                        for train_emb in param_grid['train_emb']:
                            for lr in param_grid['lr']:
                                print('Batch: ', batch, ' epoch: ', epoch, ' units_q: ', units_q, ' units_l: ', units_l,
                                      ' train_emb: ', train_emb, ' lr: ', lr)
                                self.create_model(lr=lr, units_l=units_l, units_q=units_q, train_emb=train_emb)
                                self.train(x_train, y_train, pos_train, neg_train, x_valid, y_valid, pos_valid,
                                           neg_valid, epoch, batch)

                                #                   pred = self.evaluate_test(x_valid,y_valid,pos_valid,neg_valid)
                                #                   res = self.relation_pred_wrt_cand(pred, y_valid, predicates_per_question_valid)
                                self.evaluate_wrt_sbj_entity(x_test, y_test, pos_test, neg_test,
                                                             predicates_per_question_test)
                                self.model.save_weights('model' + str(units_q) + str(batch) + str(epoch) + '.h5')
                                #                   pred = self.evaluate_test(x_test,y_test,pos_test,neg_test)
                                #                   res = self.relation_pred_wrt_cand(pred, y_test, predicates_per_question_test)

                                self.clear()

            return models
