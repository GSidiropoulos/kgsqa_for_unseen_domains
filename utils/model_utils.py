import random as rn
import gensim
import keras
import numpy as np


class ModelUtils(object):

    @staticmethod
    def pad_seq(sequences, max_len):
        padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype="int32", padding="post",
                                                            value=0)
        return padded

    @staticmethod
    def emb_matrix(ix2word,
                   path_pretrained_emb="data/embeddings/GoogleNews-vectors-negative300.bin",
                   random_init_unseen=True):
        """
        :param ix2word: dictionary of words ids to words
        :param path_pretrained_emb: path to word2vec file
        :param random_init_unseen: what to do for unseen words, random or zeros
        :return: embedding matrix
        """
        # Load Google"s pre-trained Word2Vec model.
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_pretrained_emb, binary=True)

        count_n = 0
        embedding_matrix = np.zeros((len(ix2word) + 1, 300))
        for word_id, word in ix2word.items():
            word_exist = True

            if word not in w2v_model.vocab:
                word = word.title()
                if word not in w2v_model.vocab:
                    word_exist = False

            if word_exist:
                # words not found in embedding index will be initialized all-zeros.
                embedding_vector = w2v_model.get_vector(word)
                embedding_matrix[word_id] = embedding_vector

            elif random_init_unseen:
                count_n += 1
                # words not found in embedding index will be initialized from
                # uniform distribution
                embedding_matrix[word_id] = np.random.uniform(low=-0.25, high=0.25, size=300)

        print("Embedding matrix shape: ", embedding_matrix.shape)

        return embedding_matrix

    @staticmethod
    def predicate_intersection_wrt_words(predicate_words, in_domain=True):
        """
        Predicates in which the text of their label intersect
        :param predicate_words: list of lists where each list consists of predicate tokens
        :param in_domain: if true then intersection to be the domain,
         else at least one token matched between predicates
        :return: Dictionary with predicate_id -> predicate_ids (intersection)
        """

        predicates_intersect = dict()
        for k, label_words in enumerate(predicate_words):
            label_words = label_words if in_domain else set(label_words)

            label_with_at_least_one_common_words = []
            for k_, label_words_ in enumerate(predicate_words):
                if k != k_:
                    label_words_ = label_words_ if in_domain else set(label_words_)

                    if in_domain:
                        if label_words[0] == label_words_[0]:
                            label_with_at_least_one_common_words.append(k_)
                    else:
                        if len(label_words.intersection(label_words_)) != 0:
                            label_with_at_least_one_common_words.append(k_)

            predicates_intersect[k] = label_with_at_least_one_common_words

        return predicates_intersect

    @staticmethod
    def pred_inters_wrt_keywords(predicate_words, pred2keyword):
        """
        Predicates in which the text of their label intersect
        :param predicate_words: list of list with words of predicate label
        :param pred2keyword: dictionary with predicate_id-> list of keywords
        :return: Dictionary with predicate_id -> predicate_ids (intersection)
        """

        predicates_intersect = dict()
        for p in pred2keyword.keys():
            label_words = set(predicate_words[p])

            label_with_at_least_one_common_words = []
            for p_ in pred2keyword.keys():
                if p != p_:
                    label_words_ = set(predicate_words[p_])
                    if len(label_words.intersection(label_words_)) != 0:
                        label_with_at_least_one_common_words.append(p_)

            predicates_intersect[p] = label_with_at_least_one_common_words

        return predicates_intersect

    @staticmethod
    def create_negatives(x, y, q_len, p_len, num_neg_total, num_neg_intersection, labels_intersect, pred_word_level):
        """
        :param x: data
        :param y: targets
        :param q_len: token level max length of questions
        :param p_len: token level max length of predicates
        :param num_neg_total: number of negatives
        :param num_neg_intersection: number indicating how many negatives should come from
        the intersection, intersection means either the same domain as the target or just lexical overlapping
        :param labels_intersect: predicates which are in the intersection
        :param pred_word_level: word/token level predicate
        :return: data, targets, positive predicate and negative predicate
        """
        pos_labels = []
        neg_labels = []
        x_ext = []
        y_ext = []
        for k, v in enumerate(y):
            pos_labels.extend([pred_word_level[v]] * num_neg_total)

            num_neg = num_neg_intersection if len(labels_intersect[v]) >= num_neg_intersection else len(
                labels_intersect[v])
            neg_samples = rn.sample(labels_intersect[v], num_neg)
            neg_samples.extend(
                rn.sample([i_ for i_ in range(1837)
                           if i_ not in neg_samples + [v]], num_neg_total - len(neg_samples)))
            neg_label = [pred_word_level[s] for s in neg_samples]
            neg_label = ModelUtils.pad_seq(neg_label, p_len)

            x_ext.extend(x[k] * np.ones((num_neg_total, q_len)))
            y_ext.extend(y[k] * np.ones((num_neg_total,)))
            neg_labels.extend(neg_label)

        pos_labels = ModelUtils.pad_seq(pos_labels, p_len)
        neg_labels = np.array(neg_labels)

        x = np.array(x_ext)
        y = np.array(y_ext)

        return x, y, pos_labels, neg_labels

    @staticmethod
    def create_negatives2(x, y, q_len, p_w_len, p_n_len, num_neg_total, num_neg_intersection, labels_intersect,
                          pred_word_level, pred_name_level):
        """
        Creates negative samples by taking into account both word and name level
        of the predicate label, extends dataset and returns 2D arrays.

        :param x: data
        :param y: targets
        :param q_len: token level max length of questions
        :param p_w_len: token level max length of predicates
        :param p_n_len: name level max length of predicates
        :param num_neg_total: number of negatives
        :param num_neg_intersection: number indicating how many negatives should come from
        the intersection, intersection means either the same domain as the target or just lexical overlapping
        :param labels_intersect: predicates which are in the intersection
        :param pred_word_level: word/token level predicate
        :param pred_name_level: name level predicate
        :return: (DATASET_SAMPLES*NUM_NEG, LEN)
        """

        pos_pred_w_l = []  # word level
        neg_pred_w_l = []

        pos_pred_n_l = []  # name level
        neg_pred_n_l = []

        x_ext = []
        y_ext = []
        for k, v in enumerate(y):
            pos_pred_w_l.extend([pred_word_level[v]] * num_neg_total)
            pos_pred_n_l.extend([pred_name_level[v]] * num_neg_total)

            num_neg = num_neg_intersection if len(labels_intersect[v]) >= num_neg_intersection else len(
                labels_intersect[v])
            neg_samples = rn.sample(labels_intersect[v], num_neg)
            neg_samples.extend(
                rn.sample([i_ for i_ in range(1837)
                           if i_ not in neg_samples + [v]], num_neg_total - len(neg_samples)))
            neg_w_l = [pred_word_level[s] for s in neg_samples]
            neg_w_l = ModelUtils.pad_seq(neg_w_l, p_w_len)

            neg_n_l = [pred_name_level[s] for s in neg_samples]
            neg_n_l = ModelUtils.pad_seq(neg_n_l, p_n_len)

            x_ext.extend(x[k] * np.ones((num_neg_total, q_len)))
            y_ext.extend(y[k] * np.ones((num_neg_total,)))

            neg_pred_w_l.extend(neg_w_l)
            neg_pred_n_l.extend(neg_n_l)

        pos_pred_w_l = ModelUtils.pad_seq(pos_pred_w_l, p_w_len)
        pos_pred_n_l = ModelUtils.pad_seq(pos_pred_n_l, p_n_len)

        neg_pred_w_l = np.array(neg_pred_w_l)
        neg_pred_n_l = np.array(neg_pred_n_l)

        x = np.array(x_ext)
        y = np.array(y_ext)

        return x, y, pos_pred_w_l, neg_pred_w_l, pos_pred_n_l, neg_pred_n_l

    @staticmethod
    def create_negatives2_onego(x, y, q_len, p_w_len, p_n_len, num_neg_total, num_neg_intersection, labels_intersect,
                                pred_word_level, pred_name_level):
        """
        Creates negative samples by taking into account both word and name level
        of the predicate label, Creates 3D arrays for negatives and 2D for positives,
        since we have only one positive predicate and multiple negatives. Without extending x and y.

        :return:(DATASET_SAMPLES,NUM_NEG,LEN_WORD_LEVEL), (DATASET_SAMPLES,LEN_WORD_LEVEL)
                (DATASET_SAMPLES,NUM_NEG,LEN_NAME_LEVEL), (DATASET_SAMPLES,LEN_NAME_LEVEL)
        """

        pos_pred_w_l = []  # word level
        neg_pred_w_l = []

        pos_pred_n_l = []  # name level
        neg_pred_n_l = []

        for k, v in enumerate(y):
            pos_pred_w_l.append(pred_word_level[v])
            pos_pred_n_l.append(pred_name_level[v])

            num_neg = num_neg_intersection if len(labels_intersect[v]) >= num_neg_intersection else len(
                labels_intersect[v])
            neg_samples = rn.sample(labels_intersect[v], num_neg)
            neg_samples.extend(
                rn.sample([i_ for i_ in range(1837)
                           if i_ not in neg_samples + [v]], num_neg_total - len(neg_samples)))
            neg_w_l = [pred_word_level[s] for s in neg_samples]
            neg_w_l = ModelUtils.pad_seq(neg_w_l, p_w_len)

            neg_n_l = [pred_name_level[s] for s in neg_samples]
            neg_n_l = ModelUtils.pad_seq(neg_n_l, p_n_len)

            neg_pred_w_l.append(neg_w_l)
            neg_pred_n_l.append(neg_n_l)

        pos_pred_w_l = ModelUtils.pad_seq(pos_pred_w_l, p_w_len)
        pos_pred_n_l = ModelUtils.pad_seq(pos_pred_n_l, p_n_len)

        neg_pred_w_l = np.array(neg_pred_w_l)
        neg_pred_n_l = np.array(neg_pred_n_l)

        return pos_pred_w_l, neg_pred_w_l, pos_pred_n_l, neg_pred_n_l

    @staticmethod
    def create_negatives_keywords(x, y, q_len, p_w_len, k_len, num_neg_total, num_neg_intersection, labels_intersect,
                                  labels_intersect_key, pred_word_level, predicate2keywords):
        """
        Creates negative samples by taking into account both word level
        of the predicate label and keywords of the predicate, extends dataset and returns 2D arrays.

        :param x: data
        :param y: targets
        :param q_len: token level max length of questions
        :param p_w_len: name level max length of predicate
        :param k_len: token level max length of keywords
        :param num_neg_total: number of all negatives
        :param num_neg_intersection: number indicating how many negatives should come from
        the intersection, intersection means either the same domain as the target or just lexical overlapping
        :param labels_intersect: predicate which are in the intersection
        :param labels_intersect_key: keywords for predicates which are in the intersection
        :param pred_word_level: predicate word level
        :param predicate2keywords: dictionary from predicates->keywords
        :return: (DATASET_SAMPLES*NUM_NEG, LEN)
        """

        pos_pred_w_l = []  # word level
        neg_pred_w_l = []

        pos_keyword = []  # keywords
        neg_keyword = []

        x_ext = []
        y_ext = []
        for k, v in enumerate(y):
            pos_pred_w_l.extend([pred_word_level[v]] * num_neg_total)

            if v in labels_intersect_key.keys():
                pos_keyword.extend([predicate2keywords[v]] * num_neg_total)

                num_neg = num_neg_intersection if len(labels_intersect_key[v]) >= num_neg_intersection else len(
                    labels_intersect_key[v])
                neg_samples = rn.sample(labels_intersect_key[v], num_neg)
                neg_samples.extend(
                    rn.sample([i_ for i_ in labels_intersect_key.keys()
                               if i_ not in neg_samples + [v]], num_neg_total - len(neg_samples)))
                neg_w_l = [pred_word_level[s] for s in neg_samples]
                neg_w_l = ModelUtils.pad_seq(neg_w_l, p_w_len)

                neg_n_l = [predicate2keywords[s] for s in neg_samples]
                #        print("---",neg_n_l,"---")
                neg_n_l = ModelUtils.pad_seq(neg_n_l, k_len)

                x_ext.extend(x[k] * np.ones((num_neg_total, q_len)))
                y_ext.extend(y[k] * np.ones((num_neg_total,)))

                neg_pred_w_l.extend(neg_w_l)
                neg_keyword.extend(neg_n_l)

            else:
                pos_keyword.extend([[]] * num_neg_total)

                num_neg = num_neg_intersection if len(labels_intersect[v]) >= num_neg_intersection else len(
                    labels_intersect[v])
                neg_samples = rn.sample(labels_intersect[v], num_neg)
                neg_samples.extend(
                    rn.sample([i_ for i_ in range(1837)
                               if i_ not in neg_samples + [v]], num_neg_total - len(neg_samples)))
                neg_w_l = [pred_word_level[s] for s in neg_samples]
                neg_w_l = ModelUtils.pad_seq(neg_w_l, p_w_len)

                neg_n_l = [[] for s in neg_samples]
                neg_n_l = ModelUtils.pad_seq(neg_n_l, k_len)

                x_ext.extend(x[k] * np.ones((num_neg_total, q_len)))
                y_ext.extend(y[k] * np.ones((num_neg_total,)))

                neg_pred_w_l.extend(neg_w_l)
                neg_keyword.extend(neg_n_l)

        pos_pred_w_l = ModelUtils.pad_seq(pos_pred_w_l, p_w_len)
        pos_keyword = ModelUtils.pad_seq(pos_keyword, k_len)

        neg_pred_w_l = np.array(neg_pred_w_l)
        neg_keyword = np.array(neg_keyword)

        x = np.array(x_ext)
        y = np.array(y_ext)

        return x, y, pos_pred_w_l, neg_pred_w_l, pos_keyword, neg_keyword
