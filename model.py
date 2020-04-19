# -*- coding: utf-8 -*-

import torch.nn as nn

from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.models import CrfTagger


@Model.register("bilstm-crf")
class BiLSTMCRF(CrfTagger):
    def __init__(self, vocab, args):
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=args.embed_dim,
        )

        word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

        bilstm = nn.LSTM(
            args.embed_dim,
            args.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        encoder = PytorchSeq2SeqWrapper(bilstm)
        super().__init__(
            vocab=vocab,
            text_field_embedder=word_embeddings,
            encoder=encoder,
            dropout=args.dropout,
        )
