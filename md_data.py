import argparse

import numpy as np
import pandas as pd

from data_preprocessing.data_creation import DataCreator
from data_preprocessing.text_utils import TextUtils
from utils.data_io import DataSaverLoader


def get_data(q_tok_tag, sbj_label, none_sbj):
    wrong_num_of_ent = []
    wrong_num_of_ent_num = []
    annotations = []
    q_tok_tag_rm = []
    for i, _ in enumerate(q_tok_tag):

        question = " ".join([tok[0] for tok in q_tok_tag[i]])
        if i not in none_sbj:
            q_tok_tag_rm.append(q_tok_tag[i])
            most_similar_ngram = ""
            most_similar_ngram_v = -1

            n_grams = TextUtils.create_ngrams(question.split())
            fb_label = " ".join(TextUtils.preprocess(sbj_label[i]))
            for n_gram in n_grams:
                similarity = TextUtils.similar(fb_label, n_gram)
                if most_similar_ngram_v < similarity:
                    most_similar_ngram = n_gram
                    most_similar_ngram_v = similarity

            question_string = question
            replacement_ = ["1_"] * len(most_similar_ngram.split())
            question_repl = question_string.replace(most_similar_ngram, " ".join(replacement_))

            question_repl_split = question_repl.split()
            np_ = list()
            ce = []
            for token in question_repl_split:
                if token != "1_":
                    np_.append(0)
                    ce.append("C")
                else:
                    np_.append(1)
                    ce.append("E")

            ce_rep = np.array(np_)
            # check if only right number of tokens were annotated as entities
            if np.sum(ce_rep) != len(most_similar_ngram.split()):
                wrong_num_of_ent.append(i)
                wrong_num_of_ent_num.append(ce_rep)

            annotations.append(ce)
    print("Wrong: ", len(wrong_num_of_ent))

    return q_tok_tag_rm, annotations


def create_annotations(questions, sbj_label, none_sbj):
    wrong_num_of_ent = []
    wrong_num_of_ent_num = []
    annotations = []
    for i, question in enumerate(questions):

        if i not in none_sbj:
            # question = preprocess(question)
            # find the most probable part of the sentence that matches the true_label of the subject (sbj of the triple)
            most_similar_ngram = ""
            most_similar_ngram_v = -1

            n_grams = TextUtils.create_ngrams(question)
            fb_label = " ".join(TextUtils.preprocess(sbj_label[i]))
            for n_gram in n_grams:
                similarity = TextUtils.similar(fb_label, n_gram)
                if most_similar_ngram_v < similarity:
                    most_similar_ngram = n_gram
                    most_similar_ngram_v = similarity

            #
            question_string = " ".join(question)
            replacement_ = ["1_"] * len(most_similar_ngram.split())
            question_repl = question_string.replace(most_similar_ngram, " ".join(replacement_))

            question_repl_split = question_repl.split()
            np_ = list()
            for token in question_repl_split:
                if token != "1_":
                    np_.append(0)
                else:
                    np_.append(1)

            ce_rep = np.array(np_)

            # check if only the right number of tokens were annotated as entities
            if np.sum(ce_rep) != len(most_similar_ngram.split()):
                wrong_num_of_ent.append(i)
                wrong_num_of_ent_num.append(ce_rep)

            annotations.append(np_)

    return annotations, wrong_num_of_ent, wrong_num_of_ent_num


def create_vocab_dictionaries(args):
    """Takes as input the path for an annotated freebase data file, and returns
      dictionaries word2ix, ix2word w.r.t the questions found in the file"""

    sbj_mid, predicate, obj_mid, question = DataCreator.get_spo_question(args.path_load_sq,
                                                                         "annotated_fb_data_train.txt")
    # skip questions of target domain
    skip_ids = []
    for indx, p in enumerate(predicate):
        if p.split("/")[0] == args.target_domain:
            skip_ids.append(indx)
    question = np.delete(np.array(question), skip_ids, axis=0)

    return DataCreator.create_dict(question, eos=True)


def create_md_data(args, name, word2ix, mid2entity):
    """Takes as input the path for an annotated freebase data file, and returns
    subject mids, predicates, object mids, questions, data(questions as word ids),
    annotations(sequences of 0,1 representing context and entity),
    array indices, of where the respective sbj entity is not in mid2entity """

    sbj_mid, predicate, obj_mid, question = DataCreator.get_spo_question(args.path_load_sq,
                                                                         "annotated_fb_data_" + name + ".txt")

    sbj_label = [mid2entity.get(mid) if mid2entity.get(mid) is None else mid2entity.get(mid)[0] for mid in sbj_mid]
    #  obj_label = [mid2entity.get(mid) if mid2entity.get(mid) is None else mid2entity.get(mid)[0] for mid in obj_mid]

    none_sbj = []
    for i, s in enumerate(sbj_label):
        if s is None:
            none_sbj.append(i)
    print("Subject mids not in mid2ent: ", len(none_sbj))

    questions_initial = question
    data, question = DataCreator.create_data(question, word2ix, True)
    annotations, wrong_num_of_ent, wrong_num_of_ent_num = create_annotations(question, sbj_label, none_sbj)

    return sbj_mid, predicate, obj_mid, question, questions_initial, data, annotations, none_sbj


def main():
    """
    path_load_sq: path leading to the original SimpleQuestions dataset
    path_load_mid2ent: path leading to mid2ent file
    target_domain: available domains-> "all" or any of the 82 domains, e.g. "film", "book" etc.
    path_save: where to save the data
    :return: saves preprocessed data for the MD task; even for the target domain (those data will be removed later)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_load_sq", type=str, default=None, required=True)
    parser.add_argument("--path_load_mid2ent", type=str, default=None, required=True)
    parser.add_argument("--target_domain", type=str, default="all")
    parser.add_argument("--path_save", type=str, default=None, required=True)
    args = parser.parse_args()

    # load mid to entity dictionary
    mid2entity = DataSaverLoader.load_pickle(path=args.path_load_mid2ent, filename="mid2ent")

    # create dictionaries -> word to id and id to word
    word2ix, ix2word = create_vocab_dictionaries(args)

    for i in ["train", "test", "valid"]:
        sbj_mids, relations, obj_mids, q_text, q_init, data_ids, target_annotations, non_sbj = create_md_data(args, i,
                                                                                                              word2ix,
                                                                                                              mid2entity)

        sbj_mids = np.delete(np.array(sbj_mids), non_sbj, axis=0)
        relations = np.delete(np.array(relations), non_sbj, axis=0)
        obj_mids = np.delete(np.array(obj_mids), non_sbj, axis=0)
        q_text = np.delete(np.array(q_text), non_sbj, axis=0)
        q_init = np.delete(np.array(q_init), non_sbj, axis=0)
        data_ids = np.delete(np.array(data_ids), non_sbj, axis=0)

        df_out = pd.DataFrame({"subject": sbj_mids, "relation": relations, "object": obj_mids, "data": data_ids,
                               "annotation": target_annotations, "question": q_text, "question_initial": q_init})
        DataSaverLoader.save_csv(args.path_save + i + "/", "data.csv", df_out)

        DataSaverLoader.save_pickle(path=args.path_save + i + "/", name="none_sbj", python_object=non_sbj)

    DataSaverLoader.save_pickle(path=args.path_save, name="word2ix", python_object=word2ix)
    DataSaverLoader.save_pickle(path=args.path_save, name="ix2word", python_object=ix2word)


if __name__ == "__main__":
    main()
