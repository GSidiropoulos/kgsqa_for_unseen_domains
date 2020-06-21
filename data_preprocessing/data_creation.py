from collections import Counter

import numpy as np
import pandas as pd

from data_preprocessing.text_utils import TextUtils


class DataCreator(object):

    @staticmethod
    def create_dict(questions, eos=False, cut_frq=False, cut_under=5, additional=None):
        tokens = [token for sentence in questions for token in TextUtils.preprocess(sentence)]
        if additional is not None:
            tokens.extend(additional)
        if cut_frq:
            word_count = Counter(tokens)
            words_set = set()
            for k, v in word_count.items():
                if v > cut_under:
                    words_set.add(k)

            words = sorted(list(words_set))

        else:
            words = sorted(list(set(tokens)))

        data_size, vocab_size = len(tokens), len(words)

        print("Initialize dataset with {} characters, {} unique.".format(data_size, vocab_size))

        word_to_ix = {ch: i + 1 for i, ch in enumerate(words)}
        ix_to_word = {i + 1: ch for i, ch in enumerate(words)}

        word_to_ix["UNK"] = len(word_to_ix) + 1
        ix_to_word[len(ix_to_word) + 1] = "UNK"

        if eos:
            word_to_ix["EOS"] = len(word_to_ix) + 1
            ix_to_word[len(ix_to_word) + 1] = "EOS"

        return word_to_ix, ix_to_word

    @staticmethod
    def create_predicate_dict(predicates):
        predicates_ = []
        for p in predicates:
            predicates_.append(p.replace("www.freebase.com/", ""))

        predicates = sorted(list(set(predicates_)))
        pred2ix = {ch: i for i, ch in enumerate(predicates)}
        ix2pred = {i: ch for i, ch in enumerate(predicates)}

        print("Initialize dataset with {} unique predicates.".format(len(predicates)))

        label_words = []  # word level: football player
        label_relation = []  # relation level: football_player
        for pred in pred2ix.keys():
            pred_words = TextUtils.preprocess(pred.replace("_", " ").replace("/", " "))
            label_words.append(pred_words)

            pred_relation = TextUtils.preprocess(pred.replace("/", " "))
            label_relation.append(pred_relation)

        return pred2ix, ix2pred, label_words, label_relation

    @staticmethod
    def create_data(question_txt, word2ix, eos=False):
        """
        :param question_txt: List of questions as text
        :param word2ix: dictionary that matches word to id
        :param eos: Boolean variable for appending EOS char
        :return: list of questions where each questions a list of ids / of tokens
        """

        sentence_ids = []
        setence_words = []

        if eos:
            for q in question_txt:
                sentence_ids.append([word2ix[token]
                                     if token in word2ix else word2ix["UNK"] for token in
                                     TextUtils.preprocess(q)] + [word2ix["EOS"]])

                setence_words.append([token for token in TextUtils.preprocess(q)])
        else:
            for q in question_txt:
                sentence_ids.append([word2ix[token]
                                     if token in word2ix else word2ix["UNK"] for token in
                                     TextUtils.preprocess(q)])

                setence_words.append([token for token in TextUtils.preprocess(q)])

        return sentence_ids, setence_words

    @staticmethod
    def create_targets(predicates, pred2ix):
        pred_ids = []
        for p in predicates:
            pred_ids.append(pred2ix[p])

        pred_ids = np.array(pred_ids)

        return pred_ids

    @staticmethod
    def get_spo_question(annotated_fb_data_path="", file_name=""):
        """
        :param annotated_fb_data_path: path leading to the original SimpleQuestions dataset
        :param file_name: name of the data file e.g annotated_fb_data_test.txt
        :return: 4 lists -> subject, predicate, object and question(as text)
        """

        df = pd.read_csv(annotated_fb_data_path + file_name, sep="\t", usecols=[0, 1, 2, 3],
                         names=["sbj", "relation", "obj", "question"])
        sbj_mid = df["sbj"].str.replace("www.freebase.com/m/", "").to_list()
        predicate = df["relation"].str.replace("www.freebase.com/", "").to_list()
        obj_mid = df["obj"].str.replace("www.freebase.com/m/", "").to_list()
        questions = df["question"].to_list()

        return sbj_mid, predicate, obj_mid, questions

        # annotated_fb_data = DataSaverLoader.load_file(annotated_fb_data_path, file_name)
        # annotated_fb_data = TextUtils.string_to_list(annotated_fb_data, "\n")
        # annotated_fb_data = annotated_fb_data[:-1]
        #
        # sbj_mid = list()
        # predicate = list()
        # obj_mid = list()
        # questions = list()
        #
        # for line in annotated_fb_data:
        #     line_spl = TextUtils.string_to_list(line, "\t")
        #     s = line_spl[0]
        #     s = s.replace("www.freebase.com/m/", "")
        #     sbj_mid.append(s)
        #
        #     p = line_spl[1]
        #     p = p.replace("www.freebase.com/", "")
        #     predicate.append(p)
        #
        #     o = line_spl[2]
        #     o = o.replace("www.freebase.com/m/", "")
        #     obj_mid.append(o)
        #
        #     q = line_spl[3]
        #     questions.append(q)
        #
        # return sbj_mid, predicate, obj_mid, questions
