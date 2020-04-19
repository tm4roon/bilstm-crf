# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from model import BiLSTMCRF


def main(args):

    # set a DatasetReader
    reader = SequenceTaggingDatasetReader(
        word_tag_delimiter='###',
        token_delimiter=' ',
    )

    train_dataset = reader.read(args.train)
    valid_dataset = reader.read(args.valid)
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    # set a model
    model = BiLSTMCRF(vocab, args)

    # set a trainer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    iterator = BucketIterator(
        batch_size=args.batch_size, 
        sorting_keys=[("tokens", "num_tokens")]
    )
    iterator.index_with(vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    trainer = Trainer(model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        patience=args.patience,
        num_epochs=args.max_epoch,
        grad_norm=args.grad_norm,
        cuda_device=cuda_device
    )
    trainer.train()


    # save a trained model
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with open(f'{args.save_dir}/{args.name}.th', 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(f'{args.save_dir}/{args.name}.vocab')

    with open(f'{args.save_dir}/{args.name}.args', 'wb') as f:
        pickle.dump(args, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    '''
    Implementation of BiLSTM-CRF for Sequence labeling tasks.
    ''')

    parser.add_argument('--train', '-train', default='./data/sample_train.txt',
        help='path or url of training dataset')
    parser.add_argument('--valid', '-valid', default='./data/sample_valid.txt',
        help='path or url of validation dataset')

    parser.add_argument('--lr', '-lr', type=float, default=0.1,
        help='learning rate')
    parser.add_argument('--batch-size', '-batch-size', type=int, default=32,
        help='batch size')
    parser.add_argument('--max-epoch', '-max-epoch', type=int, default=30,
        help='maximum number of epoch')
    parser.add_argument('--patience', '-patience', type=int, default=5,
        help='Number of epochs to be patient before early stopping')
    parser.add_argument('--grad-norm', '-grad-norm', type=float, default=1.0,
        help='Gradient Norm')
    parser.add_argument('--save-dir', '-ssave-dir', default='./tmp',
        help='path to directory to save a trained model')
    parser.add_argument('--name', '-name', default='bistm-crf',
        help='prefix name of model and vocabulary file')

    parser.add_argument('--embed-dim', '-embed-dim', type=int, default=256,
        help='dimension of embedding layer')
    parser.add_argument('--hidden-dim', '-hidden-dim', type=int, default=256,
        help='dimension of BiLSTM layer')
    parser.add_argument('--dropout', '-do', type=float, default=0.2,
        help='dropout rate')

    args = parser.parse_args()
    main(args)
