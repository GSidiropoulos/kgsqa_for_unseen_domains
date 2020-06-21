import argparse
import ast
import time

import numpy as np
import pandas as pd

from data_preprocessing.text_utils import TextUtils
from utils.data_io import DataSaverLoader


def get_alias(pred, questions):
    """
    :param pred: annotations from MD
    :param questions: list of tokens
    :return: text span of the question representing the entity
    """
    aliases = []
    for i in range(len(pred)):
        alias = []
        for ind in np.where(np.array(pred[i]) == 1)[0]:
            #if ind < len(questions[i]):
            alias.append(questions[i][ind])
        alias_string = " ".join(alias)

        aliases.append(alias_string)

    return aliases


def preprocess_mid2ent(mid2entity):
    """
    :return: dictionary where key is an MID and value is the entity"s name
    """
    mid2alias = dict()
    for m, l in mid2entity.items():
        tokens = TextUtils.preprocess(l[0])
        mid2alias[m] = " ".join(token.lower() for token in tokens)

    return mid2alias


def get_candidate_entities(question, aliases, mid2alias, version, print_every=10, inv_indx=None):
    if version == 1:

        c = 0
        candidates = []
        count_1 = 0
        for indx, alias in enumerate(aliases):

            if indx % print_every == 0:
                print("First " + str(indx) + " done")

            if alias != "":
                cand = [mid for mid, al in mid2alias.items() if al == alias]
                if len(cand) == 0:
                    cand = [mid for mid, al in mid2alias.items() if alias in al]

            else:
                cand = []

            if len(cand) == 0:
                # if no entity prediction was made from the MD
                # use the whole question as candidate
                count_1 += 1
                mids_ = 0
                mids = []
                q = question[indx]
                n_grams = TextUtils.create_ngrams(q)
                for n_gr in reversed(n_grams):
                    mids_tmp = [mid for mid, al in mid2alias.items() if al == n_gr]

                    if len(mids_tmp) == 0:
                        mids_tmp = [mid for mid, al in mid2alias.items() if n_gr in al]

                    if mids_ > 0:
                        break
                    if len(mids_tmp) >= 1:
                        mids_ += 1
                        mids.extend(mids_tmp)
                cand = mids

            candidates.append(cand)

    elif version == 2:

        candidates = []
        count_1 = 0
        for indx, alias in enumerate(aliases):
            print(indx, alias)

            alias_ngrams = reversed(TextUtils.create_ngrams(alias))

            for ngram in alias_ngrams:
                if ngram != "" and ngram in inv_indx:
                    cand = [triple[0] for triple in inv_indx[ngram]]

                if len(cand) > 0:
                    break

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_load_md_data", type=str, default=None, required=True)
    parser.add_argument("--path_load_mid2ent", type=str, default=None, required=True)
    parser.add_argument("--path_inverted_index", type=str, default=None, required=True)
    parser.add_argument("--path_save", type=str, default=None, required=True)

    args = parser.parse_args()

    print(args)

    # load mid to entity dictionary
    mid2entity = DataSaverLoader.load_pickle(path=args.path_load_mid2ent, filename="mid2ent")

    inv_indx = DataSaverLoader.load_pickle(path=args.path_inverted_index, filename="inverted_index")

    # DataFrame with MD predictions included (only for the test set)
    df_md = pd.read_csv(args.path_load_md_data + "/data_new.csv", usecols=[5, 7])

    # get the questions and predictions
    questions_text = df_md["question"].apply(ast.literal_eval).to_list()
    predicted_annotations = df_md["prediction"].apply(ast.literal_eval).to_list()

    alias = get_alias(predicted_annotations, questions_text)
    #print(alias)

    mid2name = preprocess_mid2ent(mid2entity)

    start = time.time()
    print("Retrieving candidates")
    print("This can take a while...")
    candidates_per_q = get_candidate_entities(questions_text, alias, mid2name, 1, 10, inv_indx)
    end = time.time()
    print("Time: ", end - start)

    DataSaverLoader.save_pickle(args.path_save, "candidates", candidates_per_q)


if __name__ == "__main__":
    main()
