import argparse
import random as rn
from collections import defaultdict

import pandas as pd

from utils.data_io import DataSaverLoader


def get_mid2pred(path, pred2ix):
    """
    :param path: path to freebase
    :param pred2ix: path to dictionary of relation->id
    :return: dictionary mid -> relations
    """
    mid2predicate = dict()
    with open(file=path, mode="r") as handle:
        for line in handle.readlines():
            spo = line.split("\t")
            s = spo[0].replace("www.freebase.com/m/", "")
            p = spo[1].replace("www.freebase.com/", "")

            if p in pred2ix:
                if s in mid2predicate:
                    mid2predicate[s] = mid2predicate[s] + [pred2ix[p]]
                else:
                    mid2predicate[s] = [pred2ix[p]]

    return mid2predicate


def sp2o_dictionary(path_fb,pred2ix):
    """
    :param path_fb: path to freebase
    :return: dictionary subject_relation -> object
    """
    sp2o = defaultdict(list)
    with open(path_fb) as f:
        for line in f.readlines():
            spo = line.split("\t")
            s_ = spo[0].replace("www.freebase.com/m/", "")
            o_ = spo[2].replace("www.freebase.com/m/", "")
            for o in o_.split():
                if o.endswith("\n"):
                    o = o.replace("\n", "")
                key = "_".join([s_, spo[1].replace("www.freebase.com/", "")])
                sp2o[key].append(o)

    return sp2o


def get_incoming_edges(pred2ix, path_fb):
    """
    :param pred2ix: path to dictionary of relation->id
    :param path_fb: path to freebase
    :return: entity popularity
    """
    obj2popularity = defaultdict(list)
    with open(path_fb) as f:
        for line in f.readlines():
            spo = line.split("\t")
            s_ = spo[0].replace("www.freebase.com/m/", "")
            o_ = spo[2].replace("www.freebase.com/m/", "")
            for o in o_.split():
                if o.endswith("\n"):
                    o = o.replace("\n", "")
                obj2popularity[o].append(
                    pred2ix[spo[1].replace("www.freebase.com/", "")] if spo[1].replace("www.freebase.com/",
                                                                                       "") in pred2ix else -1)
                obj2popularity[s_].append(
                    pred2ix[spo[1].replace("www.freebase.com/", "")] if spo[1].replace("www.freebase.com/",
                                                                                       "") in pred2ix else -1)

    return obj2popularity


def answer_prediction(predictions, candidates, obj_mid, sbj_mid, relation, mid2pred, ix2pred, random, mid2ent, sp2o,
                      obj2popularity):
    """
    :param predictions: rp predictions
    :param candidates:  candidate generation
    :param obj_mid: golden object
    :param sbj_mid: golden subject
    :param relation: golden relation
    :param mid2pred: mid->relation
    :param ix2pred: id->relation
    :param random: Bool to make a random subject entity prediction not taking into account popularity
    :param mid2ent: mid->entity
    :param sp2o: subject_relation -> object
    :param obj2popularity: entity -> list of relations
    :return: Accuracy w.r.t predicted object and subject relation pair
    """
    predicted_answer = []
    predicted_pair = []
    pred_sbj = []
    for i in range(len(predictions)):

        if random:

            final_cand = []
            for indx, candidate in enumerate(candidates[i]):
                if candidate in mid2pred:
                    if 0 < mid2pred[candidate].count(predictions[i]):
                        final_cand.append(candidate)

            predicted_sbj_mid = rn.sample(final_cand, 1)[-1] if len(final_cand) >= 1 else rn.sample(candidates[i], 1)[
                -1]
            predicted_answer.append(
                sp2o[predicted_sbj_mid + "_" + ix2pred[predictions[i]].replace("www.freebase.com/", "")])

        else:

            max_val = -1
            max_index = -1
            for indx, candidate in enumerate(candidates[i]):

                if candidate in mid2pred:
                    if mid2pred[candidate].count(predictions[i]) > 0:

                        if max_val < obj2popularity[candidate].count(predictions[i]):
                            max_val = obj2popularity[candidate].count(predictions[i])
                            max_index = indx

            predicted_sbj_mid = candidates[i][max_index]
            pred_sbj.append(predicted_sbj_mid)

            predicted_answer.append(
                sp2o[predicted_sbj_mid + "_" + ix2pred[predictions[i]].replace("www.freebase.com/", "")])
            predicted_pair.append(predicted_sbj_mid + "_" + ix2pred[predictions[i]].replace("www.freebase.com/", ""))

    # count w.r.t obj
    c = 0
    ccc = 0
    cccc = 0
    for i, answ in enumerate(predicted_answer):
        if obj_mid[i] in answ:
            c += 1
            if sbj_mid[i] + "_" + ix2pred[relation[i]] != predicted_pair[i]:
                if relation[i] == predictions[i]:
                    ccc += 1
        if pred_sbj[i] == sbj_mid[i]:
            cccc += 1

    # counting w.r.t pair
    c_pair = 0
    c_pair_2 = 0
    for i_, answ_ in enumerate(predicted_pair):
        if answ_ == (sbj_mid[i_] + "_" + ix2pred[relation[i_]]).replace("www.freebase.com/", ""):
            c_pair += 1
            c_pair_2 += 1
        elif (obj_mid[i_] in predicted_answer[i_]) and (
                mid2ent[sbj_mid[i_]] == mid2ent[pred_sbj[i_]] and relation[i_] == predictions[i_]):
            c_pair_2 += 1

    print("Object: ", c/len(sbj_mid))

    print("Subject, Relation (strict): ", c_pair/len(sbj_mid))

    print("Subject Relation:", c_pair_2/len(sbj_mid))


def main():
    """
    path_load_gold: path to data.csv file generated after running rp_data.py
    path_cg: path to entity candidate generation file
    path_rp_predictions: path to RP predictions file
    path_fb: path to freebase
    path_mid2entity: path to mid->entity dictionary
    path_pred2ix: path to relation->id dictionary
    path_ix2pred: path to id->relation dictionary
    :return: Accuracy w.r.t predicted (subject,relation) and w.r.t predicted object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_load_gold", type=str, default=None, required=True)
    parser.add_argument("--path_cg", type=str, default=None, required=True)
    parser.add_argument("--path_rp_predictions", type=str, default=None, required=True)
    parser.add_argument("--path_fb", type=str, default=None, required=True)
    parser.add_argument("--path_mid2entity", type=str, default=None, required=True)
    parser.add_argument("--path_pred2ix", type=str, default=None, required=True)
    parser.add_argument("--path_ix2pred", type=str, default=None, required=True)

    args = parser.parse_args()

    print(args)

    df_data = pd.read_csv(args.path_load_gold)
    gold_sbj = df_data["subject"].to_list()
    gold_obj = df_data["object"].to_list()
    gold_rel = df_data["targets"].to_list()

    entity_candidates = DataSaverLoader.load_pickle("/".join(args.path_cg.split("/")[:-1]) + "/"
                                                    , args.path_cg.split("/")[-1].replace(".pkl", ""))

    rp_predictions = DataSaverLoader.load_pickle("/".join(args.path_rp_predictions.split("/")[:-1]) + "/"
                                                 , args.path_rp_predictions.split("/")[-1].replace(".pkl", ""))

    pred2ix = DataSaverLoader.load_pickle("/".join(args.path_pred2ix.split("/")[:-1]) + "/"
                                          , args.path_pred2ix.split("/")[-1].replace(".pkl", ""))

    ix2pred = DataSaverLoader.load_pickle("/".join(args.path_ix2pred.split("/")[:-1]) + "/"
                                          , args.path_ix2pred.split("/")[-1].replace(".pkl", ""))

    mid2ent = DataSaverLoader.load_pickle("/".join(args.path_mid2entity.split("/")[:-1]) + "/"
                                          , args.path_mid2entity.split("/")[-1].replace(".pkl", ""))

    mid2pred = get_mid2pred(args.path_fb, pred2ix)
    sp2o = sp2o_dictionary(args.path_fb,pred2ix)

    obj2popularity = get_incoming_edges(pred2ix, args.path_fb)

    answer_prediction(predictions=rp_predictions, candidates=entity_candidates,
                      sbj_mid=gold_sbj, relation=gold_rel, obj_mid=gold_obj, mid2pred=mid2pred,
                      ix2pred=ix2pred, random=False, mid2ent=mid2ent, sp2o=sp2o, obj2popularity=obj2popularity)


if __name__ == "__main__":
    main()
