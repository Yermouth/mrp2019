import logging
import pprint
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

#logger.setLevel(logging.DEBUG)


# @Model.register("generalizer")
class ActionGeneralizer(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            field_type2embedder: Dict[str, TextFieldEmbedder],
            field_type2seq2vec_encoder: Dict[str, Seq2VecEncoder],
            field_type2seq2seq_encoder: Dict[str, Seq2SeqEncoder],
            classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            framework2field_type2embedder: Dict[str, Dict[
                str, TextFieldEmbedder]] = None,
    ) -> None:
        super(ActionGeneralizer, self).__init__(vocab, regularizer)
        self.field_types = [
            'word', 'pos', 'resolved', 'token_node_label',
            'token_node_prev_action'
        ]
        self.field_type2embedder = torch.nn.ModuleDict(field_type2embedder)
        assert all(field_type in self.field_type2embedder
                   for field_type in self.field_types)
        self.framework2field_type2embedder = torch.nn.ModuleDict(
            framework2field_type2embedder)
        self.field_type2seq2vec_encoder = torch.nn.ModuleDict(
            field_type2seq2vec_encoder)
        self.classifier_feedforward = classifier_feedforward
        self.action_type_loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    @overrides
    def forward(
            self,
            framework,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type: torch.LongTensor,
            action_params,
            token_state,
            token_node_resolveds: Dict[str, torch.LongTensor],
            token_node_labels: Dict[str, torch.LongTensor],
            token_node_prev_actions: Dict[str, torch.LongTensor],
            token_node_ids,
            token_node_childs,
    ) -> Dict[str, torch.Tensor]:

        logger.debug(('curr_node_id', curr_node_id[0]))
        logger.debug(('action_types', action_type[0]))
        logger.debug(('action_paramss', action_params[0]))
        logger.debug(('token_states', token_state[0]))
        logger.debug(('token_node_resolveds', token_node_resolveds))
        logger.debug(('token_node_labels', token_node_labels))
        logger.debug(('token_node_prev_actions', token_node_prev_actions))

        seq2vec_fields = [
            # (parse_node_labels, 'word'),
            # (parse_node_lemmas, 'word'),
            # (parse_node_uposs, 'pos'),
            # (parse_node_xposs, 'pos'),
            (token_node_resolveds, 'resolved'),
            (token_node_labels, 'token_node_label'),
            (token_node_prev_actions, 'token_node_prev_action'),
        ]

        embedded_fields = []
        for field, field_type in seq2vec_fields:
            embedded_fields.append(self.field_type2embedder[field_type](field))

        logger.debug(('embedded_fields', embedded_fields[0].shape))

        encoded_fields = []
        for seq2vec_field, embedded_field in zip(seq2vec_fields,
                                                 embedded_fields):
            field, field_type = seq2vec_field
            field_mask = util.get_text_field_mask(field)
            logger.debug(('field_mask', field_mask.shape))
            encoded_field = self.field_type2seq2vec_encoder[field_type](
                embedded_field, field_mask)
            encoded_fields.append(encoded_field)

        logits = self.classifier_feedforward(torch.cat(encoded_fields, dim=-1))

        output_dict = {'logits': logits}
        logger.debug(('output_dict', output_dict))

        if action_type is not None:
            logger.debug(('logits', logits))
            logger.debug(('label', action_type))
            action_type_loss = self.action_type_loss(logits, action_type)
            for metric in self.metrics.values():
                metric(logits, action_type)
            output_dict['loss'] = action_type_loss

        # raise NotImplementedError
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


# # @Model.register("generalizer")
# class ActionGeneralizerOld(Model):
#     def __init__(
#             self,
#             vocab: Vocabulary,
#             word_embedder: TextFieldEmbedder,
#             pos_embedder: TextFieldEmbedder,
#             seq2vec_encoder: Seq2VecEncoder,
#             classifier_feedforward: FeedForward,
#             initializer: InitializerApplicator = InitializerApplicator(),
#             regularizer: Optional[RegularizerApplicator] = None,
#     ) -> None:
#         super(ActionGeneralizer, self).__init__(vocab, regularizer)
#         self.word_embedder = word_embedder
#         logger.debug(('word_embedder', word_embedder))
#         self.pos_embedder = pos_embedder
#         logger.debug(('pos_embedder', pos_embedder))
#         self.seq2vec_encoder = seq2vec_encoder
#         self.classifier_feedforward = classifier_feedforward
#         self.loss = torch.nn.CrossEntropyLoss()
#         self.metrics = {
#             "accuracy": CategoricalAccuracy(),
#         }
#         initializer(self)

#     @overrides
#     def forward(
#             self,
#             curr_parse_node_label: Dict[str, torch.LongTensor],
#             curr_parse_node_lemma: Dict[str, torch.LongTensor],
#             curr_parse_node_upos: Dict[str, torch.LongTensor],
#             curr_parse_node_xpos: Dict[str, torch.LongTensor],
#             prev_parse_node_labels: Dict[str, torch.LongTensor],
#             prev_parse_node_lemmas: Dict[str, torch.LongTensor],
#             prev_parse_node_uposs: Dict[str, torch.LongTensor],
#             prev_parse_node_xposs: Dict[str, torch.LongTensor],
#             next_parse_node_labels: Dict[str, torch.LongTensor],
#             next_parse_node_lemmas: Dict[str, torch.LongTensor],
#             next_parse_node_uposs: Dict[str, torch.LongTensor],
#             next_parse_node_xposs: Dict[str, torch.LongTensor],
#             action_type: torch.LongTensor,
#     ) -> Dict[str, torch.Tensor]:

#         features = [
#             curr_parse_node_label,
#             curr_parse_node_lemma,
#             curr_parse_node_upos,
#             curr_parse_node_xpos,
#             prev_parse_node_labels,
#             prev_parse_node_lemmas,
#             prev_parse_node_uposs,
#             prev_parse_node_xposs,
#             next_parse_node_labels,
#             next_parse_node_lemmas,
#             next_parse_node_uposs,
#             next_parse_node_xposs,
#         ]

#         parse_features = [
#             ('curr_parse_node_label', curr_parse_node_label, 'word'),
#             ('curr_parse_node_lemma', curr_parse_node_lemma, 'word'),
#             ('curr_parse_node_upos', curr_parse_node_upos, 'pos'),
#             ('curr_parse_node_xpos', curr_parse_node_xpos, 'pos'),
#             ('prev_parse_node_labels', prev_parse_node_labels, 'word'),
#             ('prev_parse_node_lemmas', prev_parse_node_lemmas, 'word'),
#             ('prev_parse_node_uposs', prev_parse_node_uposs, 'pos'),
#             ('prev_parse_node_xposs', prev_parse_node_xposs, 'pos'),
#             ('next_parse_node_labels', next_parse_node_labels, 'word'),
#             ('next_parse_node_lemmas', next_parse_node_lemmas, 'word'),
#             ('next_parse_node_uposs', next_parse_node_uposs, 'pos'),
#             ('next_parse_node_xposs', next_parse_node_xposs, 'pos'),
#         ]

#         embedded_features = []
#         for field_name, field, field_type in parse_features:
#             if field_type == 'word':
#                 embedded_features.append(self.word_embedder(field))
#             elif field_type == 'pos':
#                 embedded_features.append(self.pos_embedder(field))

#         encoded_features = []
#         for feature, embedded_feature in zip(features, embedded_features):
#             feature_mask = util.get_text_field_mask(feature)
#             encoded_feature = self.seq2vec_encoder(embedded_feature,
#                                                    feature_mask)
#             encoded_features.append(encoded_feature)

#         logits = self.classifier_feedforward(
#             torch.cat(encoded_features, dim=-1))

#         output_dict = {'logits': logits}

#         if action_type is not None:
#             logger.debug(('logits', logits))
#             logger.debug(('label', action_type))
#             loss = self.loss(logits, action_type)
#             for metric in self.metrics.values():
#                 metric(logits, action_type)
#             output_dict['loss'] = loss

#         return output_dict

#     @overrides
#     def decode(self, output_dict: Dict[str, torch.Tensor]
#                ) -> Dict[str, torch.Tensor]:
#         """
#         Does a simple argmax over the class probabilities, converts indices to string labels, and
#         adds a ``"label"`` key to the dictionary with the result.
#         """
#         class_probabilities = F.softmax(output_dict['logits'], dim=-1)
#         output_dict['class_probabilities'] = class_probabilities

#         predictions = class_probabilities.cpu().data.numpy()
#         argmax_indices = numpy.argmax(predictions, axis=-1)
#         labels = [
#             self.vocab.get_token_from_index(x, namespace="labels")
#             for x in argmax_indices
#         ]
#         output_dict['label'] = labels
#         return output_dict

#     @overrides
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         return {
#             metric_name: metric.get_metric(reset)
#             for metric_name, metric in self.metrics.items()
#         }


# @Model.register("generalizer")
class Generalizer(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            word_embedder: TextFieldEmbedder,
            pos_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(Generalizer, self).__init__(vocab, regularizer)
        self.word_embedder = word_embedder
        logger.debug(('word_embedder', word_embedder))
        self.pos_embedder = pos_embedder
        logger.debug(('pos_embedder', pos_embedder))
        self.seq2vec_encoder = seq2vec_encoder
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
            encoded_feature = self.seq2vec_encoder(embedded_feature,
                                                   feature_mask)
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
