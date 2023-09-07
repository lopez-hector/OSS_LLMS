import json
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score


def encode_classes(intents, dacts, LB=None, MLB=None):
    if LB is None:
        LB = LabelBinarizer()
        LB.fit(intents)
    intents_transformed = LB.transform(intents)
    print(f'LB classes: {LB.classes_}')

    if MLB is None:
        MLB = MultiLabelBinarizer()
        MLB = MLB.fit(dacts)

    dacts_transformed = MLB.transform(dacts)
    print(f'MLB classes: {MLB.classes_}')

    return intents_transformed, dacts_transformed, LB, MLB


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    return acc, recall, precision, f1


def analyze_outputs(
        predictions_file: str,
        model_type: str = 'causal'
):
    # load predictions from file
    predictions = json.load(open(predictions_file, 'r'))
    predictions_sentences = predictions['predictions']

    def extract_info(data):
        """
        Script that will process causal output to compare to targets

            :returns
                extracted_result: Text that was outputed by model (i.e without special tokens)
                extracted_result_idx: indeces of samples with valid outputs (i.e. some may not have valid outputs)
        """
        extracted_results = None
        extracted_result_indices = None
        return extracted_results, extracted_result_indices

    print('hello')

    extracted_results, extracted_result_indices = extract_info(predictions_sentences)

    targets = np.array(predictions['targets'])[extracted_result_indices]


    # Print the extracted information
    """script to evaluate metrics"""
    """save to disk"""

def to_df(recall, precision, f1, label_counts, pred_counts, binarizer):
    import pandas as pd  # have to import here to avoid package missing in modal error
    df = pd.DataFrame(data=zip(recall, precision, f1, label_counts, pred_counts),
                      columns=['recall', 'precision', 'F1', 'total_in_set', 'preds_counts(TP+FP)'])
    df.index = binarizer.classes_
    return df
