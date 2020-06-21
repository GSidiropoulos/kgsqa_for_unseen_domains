import argparse
import ast

import numpy as np
import pandas as pd

from data_preprocessing.data_creation import DataCreator
from data_preprocessing.text_utils import TextUtils
from md_data import create_annotations
from utils.data_io import DataSaverLoader


def create_dict_pl(questions_list, tokens_label):
    """When we create vocabulary with placeholders, no need to preprocess
    sentences since they have already been preprocessed"""
    tokens = [token for sentence in questions_list for token in sentence]
    tokens.extend(list(tokens_label))

    words = sorted(list(set(tokens)))
    data_size, vocab_size = len(tokens), len(words)

    print("Initialize dataset with {} characters, {} unique.".format(data_size, vocab_size))

    word_to_ix = {ch: i + 1 for i, ch in enumerate(words)}
    ix_to_word = {i + 1: ch for i, ch in enumerate(words)}

    word_to_ix["UNK"] = len(word_to_ix) + 1
    ix_to_word[len(ix_to_word) + 1] = "UNK"

    return word_to_ix, ix_to_word


def create_dict_pl_separately(tokens):
    """This method  will be called twice, once for the tokens in questions
    and one for the tokens in predicate labels. Use this method if you do not
    want to share embeddings for questions and predicates"""

    words = sorted(list(set(tokens)))
    data_size, vocab_size = len(tokens), len(words)

    print("Initialize dataset with {} characters, {} unique.".format(data_size, vocab_size))

    word_to_ix = {ch: i + 1 for i, ch in enumerate(words)}
    ix_to_word = {i + 1: ch for i, ch in enumerate(words)}

    word_to_ix["UNK"] = len(word_to_ix) + 1
    ix_to_word[len(ix_to_word) + 1] = "UNK"

    return word_to_ix, ix_to_word


def create_predicate_dictionaries(path):
    train = pd.read_csv(path + "annotated_fb_data_train.txt", sep="\t", usecols=[1], names=["relation"])
    valid = pd.read_csv(path + "annotated_fb_data_valid.txt", sep="\t", usecols=[1], names=["relation"])
    test = pd.read_csv(path + "annotated_fb_data_test.txt", sep="\t", usecols=[1], names=["relation"])

    train_relations = train["relation"].to_list()
    valid_relations = valid["relation"].to_list()
    test_relations = test["relation"].to_list()

    train_relations.extend(valid_relations)
    train_relations.extend(test_relations)

    pred2ix, ix2pred, predicate_words, predicate_names = DataCreator.create_predicate_dict(train_relations)

    return pred2ix, ix2pred, predicate_words, predicate_names


def create_vocab_dictionaries(args, placeholder=False, pred_w=None, pred_n=None, annotations=None, none_sbj=None,
                              separately=False, keywords=None, indices=[]):

    """
    :param args: args
    :param placeholder: placeholders exist or not
    :param pred_w: predicate words
    :param pred_n: predicate names
    :param annotations: annotations
    :param none_sbj: samples for which we do not have problematic mid
    :param separately: use different embeddings for questions and predicates
    :param keywords: use keywords or not
    :param indices: target domain indices
    :return: dictionaries word2ix, ix2word
    """

    sbj_mid, predicate, obj_mid, question = DataCreator.get_spo_question(args.path_load_sq, "annotated_fb_data_train.txt")
    # init
    word2ix = None
    ix2word = None
    word2ix_predicates = None
    ix2word_predicates = None

    if (pred_w is None) and (not placeholder):
        # for the multi-class models
        word2ix, ix2word = DataCreator.create_dict(questions=question, eos=False, cut_frq=False, additional=keywords)

    elif (pred_w is not None) and (keywords is None) and (not placeholder):
        # for relatedness models without placeholders
        print("Pred_w: ", len(pred_w), " Placeholder: ", placeholder)
        tokens = [token for label_w in pred_w for token in label_w]
        word2ix, ix2word = DataCreator.create_dict(questions=question, eos=False, cut_frq=False, additional=tokens)

    elif (pred_w is not None) and (keywords is not None) and (not placeholder):
        # for relatedness models without placeholders
        # with word level predicate labels and keywords
        print("Pred_w: ", len(pred_w), " Placeholder:", placeholder, " Keywords:", len(keywords))
        tokens = [token for label_w in pred_w for token in label_w] + keywords
        word2ix, ix2word = DataCreator.create_dict(questions=question, eos=False, cut_frq=False, additional=tokens)

    elif (pred_w is not None) and (keywords is not None) and placeholder:
        # for relatedness models with placeholders
        # with word level predicate labels and keywords
        print("Pred_w: ", len(pred_w), " Placeholder:", placeholder, " Keywords:", len(keywords))
        question_words = []
        for q in question:
            question_words.append([token for token in TextUtils.preprocess(q)])
        question = replace_plchdr(np.delete(np.array(question_words), none_sbj, axis=0), annotations)
        # comment next line if there are no training indices to delete
        print("delete unseen domain questions")
        print(len(indices))
        question = list(np.delete(np.array(question), indices, axis=0))

        additional_predicate_txt = set([w for p_w in pred_w for w in p_w]).union(set(keywords))
        word2ix, ix2word = create_dict_pl(question, additional_predicate_txt)

    elif (pred_w is not None) and placeholder:
        # for relatedness models with placeholders instead of subject entities in question
        question_words = []
        for q in question:
            question_words.append([token for token in TextUtils.preprocess(q)])
        question = replace_plchdr(np.delete(np.array(question_words), none_sbj, axis=0), annotations)

        additional_predicate_txt = set([w for p_w in pred_w for w in p_w]) if pred_n is None else set(
            [w for p_w in pred_w for w in p_w]).union(set([n for p_n in pred_n for n in p_n]))

        if separately:
            # create different vocabs for question and predicate labels
            word2ix, ix2word = create_dict_pl_separately([token for sentence in question for token in sentence])
            word2ix_predicates, ix2word_predicates = create_dict_pl_separately(additional_predicate_txt)
        else:
            # same vocab for question and predicate labels
            # comment next line if there are no indices to delete
            print("delete unseen domain questions")
            print(len(indices))
            question = list(np.delete(np.array(question), indices, axis=0))
            word2ix, ix2word = create_dict_pl(question, additional_predicate_txt)

    print("Dictionary size: ", len(word2ix))

    return word2ix, ix2word, word2ix_predicates, ix2word_predicates


def remove_seq_duplicates(question):
    """ Removes consequent sbj placeholders"""
    i = 0
    while i < len(question) - 1:
        if question[i] == question[i + 1] and question[i] == "sbj":
            del question[i]
        else:
            i = i + 1

    return question


def replace_plchdr(questions, annotations, name="train"):
    questions_placeholder = []
    for i in range(len(questions)):
        tmp = questions[i]
        sbj_exists = np.where(np.array(annotations[i]) == 1)[0]

        if name == "test":
            if len(sbj_exists) > 0:
                q_new = [word if index not in sbj_exists else "sbj" for index, word in enumerate(tmp)]
                q_new = remove_seq_duplicates(q_new)
                questions_placeholder.append(q_new)
            else:
                questions_placeholder.append(tmp)
        else:
            tmp[sbj_exists[0]] = "sbj"

            if len(sbj_exists) > 1:
                tmp_ = np.delete(tmp, sbj_exists[1:])
                questions_placeholder.append(tmp_)
            else:
                questions_placeholder.append(tmp)

    return questions_placeholder


def create_data_pl(questions_list, word_to_ix):
    sentence_ids = []
    for q in questions_list:
        sentence_ids.append([word_to_ix[token]
                             if token in word_to_ix else word_to_ix["UNK"] for token in q])

    return sentence_ids


def create_data(path, word2ix, pred2ix):
    """
    :param path: path for an annotated freebase data file
    :param word2ix: word to id
    :param pred2ix: predicate to id
    :return: subject mids, predicates, object mids, data(questions as word ids), targets(predicate ids), questions
    """

    df_data = pd.read_csv(path, usecols=[0, 1, 2, 3, 4, 6])
    sbj_mid = df_data["subject"].to_list()
    obj_mid = df_data["object"].to_list()
    predicate = df_data["relation"].to_list()
    annotations = df_data["annotation"].apply(ast.literal_eval).to_list()
    question = df_data["question_initial"].to_list()

    data, question = DataCreator.create_data(question, word2ix, eos=False)
    targets = DataCreator.create_targets(predicate, pred2ix)

    return sbj_mid, predicate, obj_mid, data, targets, question


def create_data_placeholder(path, word2ix, pred2ix, name):
    """Takes as input the path for an annotated freebase data file, and returns
      subject mids, predicates, object mids, data(questions as word ids), targets(predicate ids), questions"""
    use_col = [0, 1, 2, 3, 4, 6, 7] if name == "test" else [0, 1, 2, 3, 4, 6]
    df_data = pd.read_csv(path, usecols=use_col)
    sbj_mid = df_data["subject"].to_list()
    obj_mid = df_data["object"].to_list()
    predicate = df_data["relation"].to_list()
    annotations = df_data["annotation"].apply(ast.literal_eval).to_list() if name != "test" else df_data[
        "prediction"].apply(ast.literal_eval).to_list()
    question = df_data["question_initial"].to_list()

    question_words = []
    for q in question:
        question_words.append([token for token in TextUtils.preprocess(q)])
    question = replace_plchdr(np.array(question_words), annotations, name)
    data = create_data_pl(question, word2ix)

    targets = DataCreator.create_targets(predicate, pred2ix)

    return sbj_mid, predicate, obj_mid, data, targets, question


def find_remove_domain_ids(domain, ix2pred):
    """ finds the relation types within the target domain """
    ids = []
    for key, value in ix2pred.items():
        value_ = value.split("/")[0]
        if value_ in [domain]:
            ids.append(key)
    return ids


def create_targets(predicates, pred_to_ix):
    train_targets = list()
    for p in predicates:
        train_targets.append(np.array(pred_to_ix[p.replace("\n", "").replace("www.freebase.com/", "")]).reshape(1))

    return train_targets


def preprocess_synthetic_questions(df, mid2entity, predicate_names):
    """ adds placeholders on the synthetic questions """

    add_questions = []
    additional_quest = []
    for i in range(len(df)):

        reference = [df["y_label_post_proc"][i].replace("_END_", "").replace("_START_", "").split()]
        candidate = df["y_post_proc"][i].replace("_END_", "").replace("_START_", "").split()

        new_cand = []
        if "_PLACEHOLDER_SUB_" not in candidate:
            tmp = mid2entity[df["sub"][i].replace("www.freebase.com/m/", "")]

            if len(tmp) != 0:
                annotations_padded, wrong_num_of_ent, wrong_num_of_ent_num = create_annotations([candidate], mid2entity[
                    df["sub"][i].replace("www.freebase.com/m/", "")], [])
                if 1 in annotations_padded[0]:
                    inds = [index for index, x in enumerate(annotations_padded[0]) if x == 1]

                    candidate = ["_PLACEHOLDER_SUB_" if index in inds else word for index, word in
                                 enumerate(candidate)]

        for word in candidate:
            if word == "_PLACEHOLDER_SUB_" and "sbj" not in new_cand:
                new_cand.append("sbj")
            elif word != "_PLACEHOLDER_SUB_":
                new_cand.append(word)
        new_cand = TextUtils.preprocess(" ".join(new_cand))

        add_questions.extend(new_cand)
        additional_quest.append(new_cand)
    for pp in predicate_names:
        add_questions.extend(pp)

    return add_questions, additional_quest


def check_args(args, parser):
    if args.use_synthetic_questions and (args.path_load_synthetic is None):
        parser.error("when --use_synthetic_questions, you must provide --path_load_synthetic")

    if args.use_keywords and (args.path_load_keywords is None):
        parser.error("when --use_keywords, you must provide --path_load_keywords")

    if args.use_relation_words_only and (args.use_keywords or args.use_synthetic_questions):
        parser.error("when --use_relation_words_only,  --use_synthetic_questions | --use_keywords"
                     "should not be given as arguments")


def main():
    """
    path_load_sq: path of the original SimpleQuestions dataset
    path_load_md_data: path of the folder generated after running MD
    path_load_synthetic: path of the csv file generated by QG
    path_load_mid2ent: path of the folder in which mid2ent exists in
    path_load_keywords: path of the folder in which pred2key exists in (this file is created when extracting keywords)
    path_save: where to save the RP data
    target_domain: the target domain (e.g. book, film, astronomy etc)
    placeholders: placeholders in questions instead of subject names or questions in the original format
    use_relation_words_only: use only words from original questions not to be used with keywords or synthetic questions
    use_keywords: if keywords w.r.t relation will be provided
    use_synthetic_questions: if synthetic questions of the target domain will be provided
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_load_sq", type=str, default=None, required=True)
    parser.add_argument("--path_load_md_data", type=str, default=None)
    parser.add_argument("--path_load_synthetic", type=str, default=None)
    parser.add_argument("--path_load_mid2ent", type=str, default=None, required=True)
    parser.add_argument("--path_load_keywords", type=str, default=None)
    parser.add_argument("--path_save", type=str, default=None, required=True)
    parser.add_argument("--target_domain", type=str, default=None)
    parser.add_argument("--placeholders", action="store_true")
    parser.add_argument("--use_relation_words_only", action="store_true")
    parser.add_argument("--use_keywords", action="store_true")
    parser.add_argument("--use_synthetic_questions", action="store_true")
    args = parser.parse_args()

    # check if args are provided correctly
    check_args(args, parser)

    save_path = args.path_save

    # load mid to entity dictionary
    mid2entity = DataSaverLoader.load_pickle(path=args.path_load_mid2ent, filename="mid2ent")

    # if args.placeholders:
    #     # load train annotations only for the placeholder case (plc in questions)
    df_train = pd.read_csv(args.path_load_md_data + "/train/" + "/data.csv", usecols=[4])
    train_annotations = df_train["annotation"].apply(ast.literal_eval).to_list()
    train_none_sbj = DataSaverLoader.load_pickle(path=args.path_load_md_data + "/train/", filename="none_sbj")

    # create predicate label to id, id to predicate dictionaries
    # and word level predicate labels and name level predicate labels list
    pred2ix, ix2pred, predicate_words, predicate_names = create_predicate_dictionaries(args.path_load_sq)

    if args.use_keywords:
        # load keywords so they can be included in the vocab
        predicate2keywords = DataSaverLoader.load_pickle(path=args.path_load_keywords, filename="pred2key")
        keywords_total = []
        for keywords in predicate2keywords.values():
            keywords_total.extend(keywords)

    # indices to delete, if there is no domain given it will remain empty
    indices = []
    if args.target_domain is not None:
        # find the training samples of the target domain which appear in the initial training set
        # those samples need to be removed for the domain adaptation scenario, otherwise we have
        # information leakage

        train = pd.read_csv(args.path_load_sq + "annotated_fb_data_train.txt", sep="\t", usecols=[1], names=["relation"])

        labels_train = create_targets(train["relation"].to_list(), pred2ix)

        # find the relations types of which are part of the target domain
        rem = find_remove_domain_ids(args.target_domain, ix2pred)

        for indx, v in enumerate(labels_train):
            if v[0] in rem:
                indices.append(indx)
        indices_to_delete = np.zeros(len(labels_train))
        indices_to_delete[indices] = 1
        indices_to_delete = np.delete(indices_to_delete, train_none_sbj, 0)
        indices = np.where(indices_to_delete == 1)[0]

    if args.use_synthetic_questions and args.placeholders:
        # the text of the target domain synthetic questions should be part of the final vocabulary
        path_noisy_q = args.path_load_synthetic
        new_q = pd.read_csv(path_noisy_q)
        add_questions, additional_quest = preprocess_synthetic_questions(new_q, mid2entity, predicate_names)

    elif args.use_synthetic_questions and not args.placeholders:
        path_noisy_q = args.path_load_synthetic
        new_q = pd.read_csv(path_noisy_q)

        add_questions = []
        additional_quest = []
        for q in new_q["y_post_proc"]:
            q = TextUtils.preprocess(q)
            add_questions.extend(q)
            additional_quest.append(q)
        for pp in predicate_names:
            add_questions.extend(pp)

    # create vocabulary
    if args.use_keywords:
        print(1)
        word2ix, ix2word, _, _ = create_vocab_dictionaries(args=args, placeholder=args.placeholders,
                                                           pred_w=predicate_words, keywords=keywords_total,
                                                           annotations=train_annotations,
                                                           none_sbj=train_none_sbj, indices=indices)
    elif args.use_relation_words_only:
        print(2)
        word2ix, ix2word, _, _ = create_vocab_dictionaries(args=args, placeholder=args.placeholders,
                                                           pred_w=predicate_words, annotations=train_annotations,
                                                           none_sbj=train_none_sbj, indices=indices)
    elif args.use_synthetic_questions:
        print(3)
        word2ix, ix2word, _, _ = create_vocab_dictionaries(args=args, placeholder=args.placeholders,
                                                           pred_w=predicate_words, keywords=add_questions,
                                                           annotations=train_annotations,
                                                           none_sbj=train_none_sbj, indices=indices)
    else:
        print(4)
        word2ix, ix2word, _, _ = create_vocab_dictionaries(args=args, placeholder=False,
                                                           pred_w=predicate_words, indices=indices)

    for i in ["train", "valid", "test"]:
        print("----", i, "----")

        path_tmp = args.path_load_md_data + i + "/data_new.csv" if i != "train" else args.path_load_md_data + i + "/data.csv"
        sbj_mid, predicate, obj_mid, data, targets, questions = create_data_placeholder(path_tmp, word2ix, pred2ix, i)

        df_out = pd.DataFrame({"subject": sbj_mid, "relation": predicate, "object": obj_mid, "data": data,
                               "targets": targets, "question": questions})
        DataSaverLoader.save_csv(save_path + i + "/", "data.csv", df_out)

        print("Number of samples: ", len(data))

    DataSaverLoader.save_pickle(path=save_path, name="word2ix", python_object=word2ix)
    DataSaverLoader.save_pickle(path=save_path, name="ix2word", python_object=ix2word)

    DataSaverLoader.save_pickle(path=save_path, name="pred2ix", python_object=pred2ix)
    DataSaverLoader.save_pickle(path=save_path, name="ix2pred", python_object=ix2pred)
    DataSaverLoader.save_pickle(path=save_path, name="pred_names", python_object=predicate_names)
    DataSaverLoader.save_pickle(path=save_path, name="pred_words", python_object=predicate_words)

    if args.use_synthetic_questions:
        y_noisy = []
        x_noisy = []
        for noisy_predicate in new_q["pred"]:
            y_noisy.append(pred2ix[noisy_predicate.replace("www.freebase.com/", "")])

        for noisy_q in additional_quest:
            x_noisy.append([word2ix[w] for w in noisy_q])
        DataSaverLoader.save_pickle(path=save_path + "train/", name="y_noisy", python_object=y_noisy)
        DataSaverLoader.save_pickle(path=save_path + "train/", name="x_noisy", python_object=x_noisy)


if __name__ == "__main__":
    main()
