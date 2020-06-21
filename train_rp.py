import argparse
import random as rn

import numpy as np
import tensorflow as tf

np.random.seed(42)
rn.seed(12345)

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph())  # , config=session_conf)
K.set_session(sess)

from rp_models.classification.bigru import BiGRU
from rp_models.classification.dan import DAN
from rp_models.similarity.lstm_names import LSTMWordsNames
from rp_models.similarity.lstm_words import LSTMWords
from rp_models.similarity.avg_pool_words import AvgPoolingWords
from utils.data_io import DataSaverLoader
from utils.rp_data_helper import DataHelper
from layers.custom_layer import AverageWords, WordDropout


def mid2pred(path, pred_to_ix):
    mid2predicate = dict()
    with open(file=path, mode="r") as handle:
        for line in handle.readlines():
            spo = line.split("\t")
            s = spo[0].replace("www.freebase.com/m/", "")
            p = spo[1].replace("www.freebase.com/", "")

            if p in pred_to_ix:
                if s in mid2predicate:
                    mid2predicate[s] = mid2predicate[s] + [pred_to_ix[p]]
                else:
                    mid2predicate[s] = [pred_to_ix[p]]

    return mid2predicate


def predicates_per_quest(candidates, mid2pred):
    predicates_per_question = []
    for cand in candidates:
        predicates_per_cand = [mid2pred[i] for i in cand if i in mid2pred]
        predicates_per_cand_flat = [item for sublist in predicates_per_cand for item in sublist]

        predicates_per_question.append(predicates_per_cand_flat)

    return predicates_per_question


def main():
    """
    job_id: job id given by SLURM. In case you do not use a machine with SLURM scheduler, use a random value.
    path_load: path to RP data folder, generated after running rp_data.py
    path_test_candidates: path leading to test entity candidates generated after running train_md.py
    path_save: where to save the generated files
    model_type: "lstm_words" ranking model with lstm encoders which takes into account the word level of the relations,
    "lstm_names" ranking model with lstm encoders which takes into account the word level and name level of the relations,
    "avg_pool_words"ranking model with average pooling encoders which takes into account the word level of the relations,
    "bigru" and "dan" are multi-class classification models
    target_domain: "film", "book", "fictional_universe" etc or "all" in case you want to use all the available domains
    total_negatives: number of total negatives in training set
    negatives_intersection: how many of the negatives should come from the intersection
    intersection: "domain" or "lexical"; domain creates negatives from the same domain while lexical only takes into
    account the lexical overlap of relations
    use_synthetic_questions: Boolean indicating the usage of synthetic questions in training set
    layers: number of layers
    units: number of units per layer e.g if you indicate 2 layers then you need to provide 2 units input like 400 400
    dropout: dropout
    rec_dropout: recurrent dropout
    word_dropout: dropping whole words, to be used in the case of DAN
    train_emb: Boolean indicating if embeddings are trainable or not
    lr: learning rate
    batch_size: batch size
    max_epochs: epochs
    save_model: Boolean indicating to save the trained model
    load_model: path to a trained model, to continue the training phase or proceed to inference (if epochs 0)
    :return: saves a RP trained model (if specified) and generates results file for RP (rp_test_results)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default=None, required=True)
    parser.add_argument("--path_load", type=str, default=None, required=True)
    parser.add_argument("--path_test_candidates", type=str, default=None, required=True)
    parser.add_argument("--path_save", type=str, default=None, required=True)
    parser.add_argument("--model_type", type=str, default="lstm_words", required=True)
    parser.add_argument("--target_domain", type=str, default="all")
    parser.add_argument("--total_negatives", type=int, default=None)
    parser.add_argument("--negatives_intersection", type=int, default=None)
    parser.add_argument("--intersection", type=str, default=None)
    parser.add_argument("--use_synthetic_questions", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--units", nargs="+", type=int, default=None)
    parser.add_argument("--dropout", type=float, nargs="+", default=None)
    parser.add_argument("--rec_dropout", nargs="+", type=float, default=None)
    parser.add_argument("--word_dropout", type=float, default=0)
    parser.add_argument("--train_emb", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", type=str, default=None)

    args = parser.parse_args()

    print(args)
    check_args(args, parser)

    if args.load_model is None:
        config = dict()
        config["job_id"] = args.job_id
        config["path_load"] = args.path_load
        config["path_test_candidates"] = args.path_test_candidates
        config["path_save"] = args.path_save
        config["model_type"] = args.model_type
        config["target_domain"] = args.target_domain
        if args.model_type == "lstm_words" or args.model_type == "lstm_names" or args.model_type == "avg_pool_words":
            config["total_negatives"] = args.total_negatives
            config["negatives_intersection"] = args.negatives_intersection
            config["intersection"] = args.intersection
        config["use_synthetic_questions"] = args.use_synthetic_questions
        config["layers"] = args.layers
        config["units"] = args.units
        config["dropout"] = args.dropout
        config["rec_dropout"] = args.rec_dropout
        if args.model_type == "dan":
            config["word_dropout"] = args.word_dropout
        config["train_emb"] = args.train_emb
        config["lr"] = args.lr
        config["batch_size"] = args.batch_size
        config["max_epochs"] = args.max_epochs
        config["save_model"] = args.save_model

        # save to config file
        DataSaverLoader.save_json(config, args.path_save + args.job_id + "/", "config")

    path_data = args.path_load
    max_q_len = 36
    embed_dim = 300
    num_cls = 1837
    max_words_len = 17
    max_names_len = 5
    create_negatives = False if args.model_type == "bigru" or args.model_type == "dan" else True
    tot_neg = {"train": args.total_negatives, "test": 1, "valid": 1}  # number of total negative samples
    intersect_neg = {"train": args.negatives_intersection, "test": 1, "valid": 1}  # neg samples from intersection
    domain_intersection = True if args.intersection == "domain" else False
    names_included = True if args.model_type == "lstm_names" else False
    rmv_dom = [args.target_domain]
    onego = False
    noisy_questions = args.use_synthetic_questions

    data_helper = DataHelper(path_data=path_data, max_seq_len=max_q_len, embed_dim=embed_dim, num_classes=num_cls,
                             max_label_words_len=max_words_len, max_label_names_len=max_names_len,
                             create_negatives=create_negatives, total_negatives=tot_neg,
                             negatives_intersection=intersect_neg, domain_intersection=domain_intersection,
                             incl_names=names_included, remove_domains=rmv_dom,
                             one_go=onego, noisy_questions=noisy_questions)

    if args.model_type == "bigru":
        rc = BiGRU(ix2word=data_helper.ix2word)

        if args.load_model is None:
            rc.create_model(learning_rate=args.lr, units=args.units, dropout=args.dropout,
                            rnn_drop=args.rec_dropout, train_emb=args.train_emb, layers=args.layers)
        else:
            rc.load_trained_model(args.load_model, name="model")
        rc.train(data_helper.x_train, data_helper.y_train, data_helper.x_valid, data_helper.y_valid,
                 epochs=args.max_epochs, batch_size=args.batch_size)

    if args.model_type == "dan":
        rc = DAN(ix2word=data_helper.ix2word)

        if args.load_model is None:
            rc.create_model(lr=args.lr, units=args.units, dropout=args.dropout,
                            word_dropout=args.word_dropout, train_emb=args.train_emb, layers=args.layers)
        else:
            rc.load_trained_model(args.load_model, name="model",
                                  custom_obj={"WordDropout": WordDropout, "AverageWords": AverageWords})
        rc.train(data_helper.x_train, data_helper.y_train, data_helper.x_valid, data_helper.y_valid,
                 epochs=args.max_epochs, batch_size=args.batch_size)

    if args.model_type == "lstm_names":
        rc = LSTMWordsNames(max_label_names_len=5, ix2word=data_helper.ix2word,
                            pred_words_ids=data_helper.pred_words_ids,
                            pred_names_ids=data_helper.pred_names_ids)
        rc.create_model(lr=args.lr, units_l=args.units[0], units_q=args.units[0], train_emb=args.train_emb)

        if args.load_model is not None:
            print("Loading model...")
            rc.model.load_weights(args.load_model + "model.h5")
            print("Done!")

        rc.train(x_train=data_helper.x_train, y_train=data_helper.y_train,
                 pos_train=data_helper.pos_word_level_train, neg_train=data_helper.neg_word_level_train,
                 pos_name_level_train=data_helper.pos_name_level_train,
                 neg_name_level_train=data_helper.neg_name_level_train,
                 x_valid=data_helper.x_valid, y_valid=data_helper.y_valid,
                 pos_valid=data_helper.pos_word_level_valid, neg_valid=data_helper.neg_word_level_valid,
                 pos_name_level_valid=data_helper.pos_name_level_valid,
                 neg_name_level_valid=data_helper.neg_name_level_valid,
                 num_epochs=args.max_epochs, batch_size=args.batch_size)

    if args.model_type == "avg_pool_words":
        rc = AvgPoolingWords(ix2word=data_helper.ix2word, pred_words_ids=data_helper.pred_words_ids)
        rc.create_model(lr=args.lr)
        if args.load_model is not None:
            print("Loading model...")
            rc.model.load_weights(args.load_model + "model.h5")
            print("Done!")

        rc.train(x_train=data_helper.x_train,
                 y_train=data_helper.y_train,
                 pos_train=data_helper.pos_word_level_train,
                 neg_train=data_helper.neg_word_level_train,
                 x_valid=data_helper.x_valid,
                 y_valid=data_helper.y_valid,
                 pos_valid=data_helper.pos_word_level_valid,
                 neg_valid=data_helper.neg_word_level_valid,
                 num_epochs=args.max_epochs, batch_size=args.batch_size)

    if args.model_type == "lstm_words":
        rc = LSTMWords(ix2word=data_helper.ix2word, pred_words_ids=data_helper.pred_words_ids)
        rc.create_model(lr=args.lr, units_l=args.units[0], units_q=args.units[0], train_emb=args.train_emb)
        if args.load_model is not None:
            print("Loading model...")
            rc.model.load_weights(args.load_model + "model.h5")
            print("Done!")
        rc.train(x_train=data_helper.x_train,
                 y_train=data_helper.y_train,
                 pos_train=data_helper.pos_word_level_train,
                 neg_train=data_helper.neg_word_level_train,
                 x_valid=data_helper.x_valid,
                 y_valid=data_helper.y_valid,
                 pos_valid=data_helper.pos_word_level_valid,
                 neg_valid=data_helper.neg_word_level_valid,
                 num_epochs=args.max_epochs, batch_size=args.batch_size)

    if args.save_model:
        if args.model_type == "lstm_words" or args.model_type == "avg_pool_words" or args.model_type == "lstm_names":

            rc.model.save_weights(args.path_save + args.job_id + "/" + "model")
        else:
            rc.save_model_(args.path_save + args.job_id + "/", "model")

    mid2pred_ = mid2pred("data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt",
                         DataSaverLoader.load_pickle(args.path_load, "pred2ix"))

    test_candidates = DataSaverLoader.load_pickle(args.path_test_candidates, "candidates")
    predicates_per_question_test = predicates_per_quest(candidates=test_candidates, mid2pred=mid2pred_)

    res = rc.evaluate_wrt_sbj_entity(data_helper.x_test, data_helper.y_test, data_helper.pos_word_level_test,
                                     data_helper.neg_word_level_test, np.array(predicates_per_question_test))

    DataSaverLoader.save_pickle(args.path_save, "rp_test_results", res)


def check_args(args, parser):
    if args.target_domain == "all" and args.use_synthetic_questions:
        parser.error("When using all available domains, --use_synthetic_questions must not appear")

    if args.model_type == "bigru" and (len(args.units) <= len(args.rec_dropout)):
        parser.error("For the BiGRU case: len(args.rec_dropout) must be len(args.units)-1")

    if args.model_type == "bigru" and (len(args.dropout) > 1):
        parser.error("For the BiGRU case: args.dropout should have one entry")

    if (args.model_type == "bigru" or args.model_type == "dan") and ((args.total_negatives is not None) or
                                                                     (args.negatives_intersection is not None) or
                                                                     (args.intersection is not None)):
        parser.error("For the classification (BiGRU/DAN) models no --total_negatives, --negatives_intersection, "
                     "--intersection")

    if (args.model_type == "bigru" or args.model_type == "dan") and ((args.total_negatives is not None) or
                                                                     (args.negatives_intersection is not None) or
                                                                     (args.intersection is not None)):
        parser.error("For the classification (BiGRU/DAN) models no --total_negatives, --negatives_intersection, "
                     "--intersection")

    if (args.model_type == "lstm_words" or args.model_type == "avg_pool_words" or args.model_type == "lstm_names") \
            and ((args.layers != 1) or
                 (args.rec_dropout is not None) or
                 (args.dropout is not None)):
        parser.error("For the similarity based models (LSTM words/names & AVG Pool words)"
                     " models no --layers, --rec_dropout, --dropout")


if __name__ == "__main__":
    main()
