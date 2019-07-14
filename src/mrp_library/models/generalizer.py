import logging
from typing import Dict, Optional

import numpy

import torch
import torch.nn.functional as F
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import (PytorchSeq2SeqWrapper,
                                               Seq2SeqEncoder)
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


# @Model.register("generalizer")
class Generalizer(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            word_embedder: TextFieldEmbedder,
            pos_embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(Generalizer, self).__init__(vocab, regularizer)
        self.word_embedder = word_embedder
        logger.debug(('word_embedder', word_embedder))
        self.pos_embedder = pos_embedder
        logger.debug(('pos_embedder', pos_embedder))
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(
            self,
            parse_label: Dict[str, torch.LongTensor],
            parse_lemma: Dict[str, torch.LongTensor],
            parse_upos: Dict[str, torch.LongTensor],
            parse_xpos: Dict[str, torch.LongTensor],
            label: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:

        features = [
            parse_label,
            parse_lemma,
            parse_upos,
            parse_xpos,
        ]

        embedded_parse_label = self.word_embedder(parse_label)
        embedded_parse_lemma = self.word_embedder(parse_lemma)
        embedded_parse_upos = self.pos_embedder(parse_upos)
        embedded_parse_xpos = self.pos_embedder(parse_xpos)

        embedded_features = [
            embedded_parse_label,
            embedded_parse_lemma,
            embedded_parse_upos,
            embedded_parse_xpos,
        ]

        encoded_features = []
        for feature, embedded_feature in zip(features, embedded_features):
            feature_mask = util.get_text_field_mask(feature)
            encoded_feature = self.encoder(embedded_feature, feature_mask)
            encoded_features.append(encoded_feature)

        logits = self.classifier_feedforward(
            torch.cat(encoded_features, dim=-1))

        output_dict = {'logits': logits}

        if label is not None:
            logger.debug(('label', label))
            loss = self.loss(logits, label)
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [
            self.vocab.get_token_from_index(x, namespace="labels")
            for x in argmax_indices
        ]
        output_dict['label'] = labels
        return output_dict
