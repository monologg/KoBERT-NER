import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, BertTokenizer
from tokenization_kobert import KoBertTokenizer

from model import BertClassifier, DistilBertClassifier

MODEL_CLASSES = {
    'kobert': (BertConfig, BertClassifier, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertClassifier, KoBertTokenizer),
    'bert': (BertConfig, BertClassifier, BertTokenizer),
    'kobert-lm': (BertConfig, BertClassifier, KoBertTokenizer)
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm'
}


def get_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return f1_pre_rec(preds, labels)


def f1_pre_rec(preds, labels):
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }
