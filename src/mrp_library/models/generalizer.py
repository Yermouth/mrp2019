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
logger.setLevel(logging.INFO)

# logger.setLevel(logging.DEBUG)


# @Model.register("generalizer")
class ActionGeneralizer(Model):
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
        super(ActionGeneralizer, self).__init__(vocab, regularizer)
        self.word_embedder = word_embedder
        logger.debug(('word_embedder', word_embedder))
        self.pos_embedder = pos_embedder
        logger.debug(('pos_embedder', pos_embedder))
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    @overrides
    def forward(
            self,
            curr_parse_node_label: Dict[str, torch.LongTensor],
            curr_parse_node_lemma: Dict[str, torch.LongTensor],
            curr_parse_node_upos: Dict[str, torch.LongTensor],
            curr_parse_node_xpos: Dict[str, torch.LongTensor],
            prev_parse_node_labels: Dict[str, torch.LongTensor],
            prev_parse_node_lemmas: Dict[str, torch.LongTensor],
            prev_parse_node_uposs: Dict[str, torch.LongTensor],
            prev_parse_node_xposs: Dict[str, torch.LongTensor],
            next_parse_node_labels: Dict[str, torch.LongTensor],
            next_parse_node_lemmas: Dict[str, torch.LongTensor],
            next_parse_node_uposs: Dict[str, torch.LongTensor],
            next_parse_node_xposs: Dict[str, torch.LongTensor],
            action_type: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:

        features = [
            curr_parse_node_label,
            curr_parse_node_lemma,
            curr_parse_node_upos,
            curr_parse_node_xpos,
            prev_parse_node_labels,
            prev_parse_node_lemmas,
            prev_parse_node_uposs,
            prev_parse_node_xposs,
            next_parse_node_labels,
            next_parse_node_lemmas,
            next_parse_node_uposs,
            next_parse_node_xposs,
        ]

        parse_features = [
            ('curr_parse_node_label', curr_parse_node_label, 'word'),
            ('curr_parse_node_lemma', curr_parse_node_lemma, 'word'),
            ('curr_parse_node_upos', curr_parse_node_upos, 'pos'),
            ('curr_parse_node_xpos', curr_parse_node_xpos, 'pos'),
            ('prev_parse_node_labels', prev_parse_node_labels, 'word'),
            ('prev_parse_node_lemmas', prev_parse_node_lemmas, 'word'),
            ('prev_parse_node_uposs', prev_parse_node_uposs, 'pos'),
            ('prev_parse_node_xposs', prev_parse_node_xposs, 'pos'),
            ('next_parse_node_labels', next_parse_node_labels, 'word'),
            ('next_parse_node_lemmas', next_parse_node_lemmas, 'word'),
            ('next_parse_node_uposs', next_parse_node_uposs, 'pos'),
            ('next_parse_node_xposs', next_parse_node_xposs, 'pos'),
        ]

        embedded_features = []
        for field_name, field, field_type in parse_features:
            if field_type == 'word':
                embedded_features.append(self.word_embedder(field))
            elif field_type == 'pos':
                embedded_features.append(self.pos_embedder(field))

        encoded_features = []
        for feature, embedded_feature in zip(features, embedded_features):
            feature_mask = util.get_text_field_mask(feature)
            encoded_feature = self.encoder(embedded_feature, feature_mask)
            encoded_features.append(encoded_feature)

        logits = self.classifier_feedforward(
            torch.cat(encoded_features, dim=-1))

        output_dict = {'logits': logits}

        if action_type is not None:
            logger.debug(('logits', logits))
            logger.debug(('label', action_type))
            loss = self.loss(logits, action_type)
            for metric in self.metrics.values():
                metric(logits, action_type)
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

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }


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
