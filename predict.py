# -*- coding: utf-8 -*-

import pickle
import argparse

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.predictors import SentenceTaggerPredictor

from model import BiLSTMCRF


def main(args):
    # load a trained model
    loaded_vocab = Vocabulary.from_files(f'{args.checkpoint}.vocab')
    with open(f'{args.checkpoint}.args', 'rb') as f:
        loaded_args = pickle.load(f)
    model = BiLSTMCRF(loaded_vocab, loaded_args)

    with open(f'{args.checkpoint}.th', 'rb') as f:
        model.load_state_dict(torch.load(f))
    if torch.cuda.is_available():
        cuda_device = 0
        model.cuda(cuda_device)
    
    
    # set a predicator 
    reader = SequenceTaggingDatasetReader(
        word_tag_delimiter='###',
        token_delimiter=' ',
    )
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    # prediction
    with open(args.test) as f:
        for line in f:
            print(' '.join(predictor.predict(line.rstrip())['tags']))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-checkpoint', default='./checkpoint/bilstm-crf',
        help='file path without extention')
    parser.add_argument('--test', '-test', default='./data/sample_valid.txt',
        help='file path of test dataset')
    args = parser.parse_args()
    main(args)
