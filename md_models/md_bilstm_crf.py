import numpy as np
import keras.optimizers as optimizers
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from md_models.mention_detection import MentionDetection


class MDBiLSTMCRF(MentionDetection):

    def create_model(self, layers=2, rec_dropout=[0.2, 0.2], rnn_dropout=[0.5, 0.5], units=[400, 400], lr=0.001,
                     train_emb=False):
        input = Input(shape=(self._max_seq_len,))
        x_rnn = Embedding(self._nb_words, self._embed_dim,
                          weights=[self._embedding_matrix], mask_zero=True, trainable=train_emb)(input)

        for layer in range(layers):
            x_rnn = LSTM(units=units[layer], return_sequences=True,
                         recurrent_dropout=rec_dropout[layer], dropout=rnn_dropout[layer])(x_rnn)

        crf_inp = TimeDistributed(Dense(units[-1], activation="relu"))(x_rnn)
        crf = CRF(self._n_tags)  # CRF layer
        out = crf(crf_inp)  # output

        model = Model(input, out)
        model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.viterbi_acc])

        self.model = model

    def load_pretrained_model(self, path_model, layers=2, rec_dropout=[0.2, 0.2], rnn_dropout=[0.5, 0.5], units=[400, 400], lr=0.001, train_emb=False):
        """
        Method that loads a pretrained CRF on top of BiLSTM model. However, the loading is a bit tricky.
        To successfully load the model you need to do a naive train at first in order to load the model afterwards.
        """

        print("BiLSTM with CRF...")
        self.create_model(layers, rec_dropout, rnn_dropout, units, lr, train_emb)

        # model.summary()

        # define callbacks
        print("Dummy train to init model...")
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=1)
        callbacks_list = [early_stopping]

        # model training with batch normalization
        hist = self.model.fit(np.random.randint(low=2, high=105, size=(1, 36)), np.zeros(shape=(1, 36, 2)))

        print("Loading model...")
        # Load model
        save_load_utils.load_all_weights(self.model, path_model)
        print("Done!")

        #self.model = model
