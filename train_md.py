import argparse
import ast
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf

from md_models.md_bilstm import MDBiLSTM
from md_models.md_bilstm_crf import MDBiLSTMCRF
from md_models.md_rbilstm import MDResidualBiLSTM
from utils.data_io import DataSaverLoader
from utils.md_data_helper import DataHelperMD

np.random.seed(42)
rn.seed(12345)

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph())  # , config=session_conf)
K.set_session(sess)


def inference(md, X, y):
    pred = md.evaluate_on_test_set(X, y)
    return pred


def init_model(type, data_helper):
    if type == "bilstm_crf":
        model = MDBiLSTMCRF(ix2word=data_helper.ix2word)
    elif type == "bilstm":
        model = MDBiLSTM(ix2word=data_helper.ix2word)
    elif type == "rbilstm":
        model = MDResidualBiLSTM(ix2word=data_helper.ix2word)

    return model


def to_results_file(args, helper, name, predictions):
    df_data = pd.read_csv(args.path_load + name + "/data.csv",
                          skiprows=helper.skip_rows[name]) if args.target_domain != "all" else pd.read_csv(
        args.path_load + name + "/data.csv")

    gold_annotation = df_data["annotation"].apply(ast.literal_eval).to_list()
    predictions = [annot[:len(gold_annotation[index])].tolist() for index, annot in enumerate(predictions)]
    df_data = df_data.assign(prediction=pd.Series(predictions))
    DataSaverLoader.save_csv(args.path_load + name + "/", "data_new.csv", df_data)


def main():
    """
    job_id: job id given by SLURM. In case you do not use a machine with SLURM scheduler, use a random value.
    path_load: from where to load data.
    path_save: where to save models and config files.
    model_types: available types -> rbilstm, bilstm, bilstm_crf.
    target_domains: available domains-> "all" or any of the 82 domains, e.g. "film", "book" etc.
    layers: number of layers.
    units: a list of units (one value per layer).
    dropout: a list of rnn dropout values (one value per layer).
    rec_dropout: a list of recurrent dropout (one value per layer).
    train_emb: trainable or frozen word embeddings; if given then True else False.
    lr: learning rate.
    batch_size: batch size.
    max_epochs: the maximum number of epochs for training the model.
    save_model: save the trained model in an h5 file. If given then True else False.
    If you provide a model (load_model) you do not have to provide most of
    the arguments (for the rbilstm & bilstm).Mandatory to include: path_load,
    path_save, model_type, max_epochs, target_domain;
    If the model you load is of bilstm_crf_type, please provide all the arguments.

    :return: for validation and test save new data files where the prediction of MD is included (only for target)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default=None, required=True)
    parser.add_argument("--path_load", type=str, default=None, required=True)
    parser.add_argument("--path_save", type=str, default=None, required=True)
    parser.add_argument("--model_type", type=str, default="rbilstm", required=True)
    parser.add_argument("--target_domain", type=str, default="all", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--units", nargs="+", type=int, default=None)
    parser.add_argument("--dropout", nargs="+", type=float, default=None)
    parser.add_argument("--rec_dropout", nargs="+", type=float, default=None)
    parser.add_argument("--train_emb", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
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
        config["path_save"] = args.path_save
        config["model_type"] = args.model_type
        config["target_domain"] = args.target_domain
        config["layers"] = args.layers
        config["units"] = args.units
        config["dropout"] = args.dropout
        config["rec_dropout"] = args.rec_dropout
        config["train_emb"] = args.train_emb
        config["lr"] = args.lr
        config["batch_size"] = args.batch_size
        config["max_epochs"] = args.max_epochs
        config["save_model"] = args.save_model

        # save to config file
        DataSaverLoader.save_json(config, args.path_save + args.job_id + "/", "config")

    data_helper = DataHelperMD(path_data=args.path_load, target_domain=args.target_domain)

    # init model
    md_model = init_model(args.model_type, data_helper)

    if args.load_model is not None:

        if args.model_type == "bilstm_crf":
            md_model.load_pretrained_model(path_model=args.load_model, layers=args.layers, rec_dropout=args.rec_dropout,
                                           rnn_dropout=args.dropout, units=args.units, lr=args.lr,
                                           train_emb=args.train_emb)
        else:
            print("Loading model...")
            md_model.load_trained_model(args.load_model)
            print("Done!")
    else:
        md_model.create_model(layers=args.layers, rec_dropout=args.rec_dropout, rnn_dropout=args.dropout,
                              units=args.units, lr=args.lr, train_emb=args.train_emb)

    md_model.train(x_train=data_helper.x_train, y_train=data_helper.y_train,
                   x_valid=data_helper.x_valid, y_valid=data_helper.y_valid,
                   batch_size=args.batch_size, epochs=args.max_epochs)

    # new csv file which includes the prediction for the validation and test
    pred = inference(md_model, data_helper.x_test, data_helper.y_test)
    to_results_file(args, data_helper, "test", pred)

    pred = inference(md_model, data_helper.x_valid, data_helper.y_valid)
    to_results_file(args, data_helper, "valid", pred)

    if args.save_model:
        md_model.save_model_h5(path=args.path_save + args.job_id + "/",
                               name="model_" + args.target_domain) if args.model_type != "bilstm_crf" else \
            md_model.save_crf_model(args.path_save + args.job_id + "/", "model_weights_" + args.target_domain)


def check_args(args, parser):
    if args.model_type == "bilstm_crf" and (
            args.layers is None or args.units is None or args.dropout is None or args.rec_dropout is None or args.lr is None):
        parser.error("when you use bilstm_crf, you must provide all the arguments")


if __name__ == "__main__":
    main()
