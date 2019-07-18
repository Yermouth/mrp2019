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
            cuda_device,
            vocab: Vocabulary,
            field_type2embedder: Dict[str, TextFieldEmbedder],
            field_type2seq2vec_encoder: Dict[str, Seq2VecEncoder],
            field_type2seq2seq_encoder: Dict[str, Seq2SeqEncoder],
            action_classifier_feedforward: FeedForward,
            action_num_pop_classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            framework2field_type2embedder: Dict[
                str, Dict[str, TextFieldEmbedder]] = None,
    ) -> None:
        super(ActionGeneralizer, self).__init__(vocab, regularizer)
        self.cuda_device = cuda_device
        self.vocab = vocab
        self.action_type2action_type_name = {
            -1: 'ERROR',
            0: 'APPEND',
            1: 'RESOLVE',
            2: 'IGNORE',
        }
        self.resolve_tensor = torch.tensor(vocab.get_token_index('RESOLVE'))
        if cuda_device != -1:
            self.resolve_tensor = self.resolve_tensor.cuda(cuda_device)

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
        self.field_type2seq2seq_encoder = torch.nn.ModuleDict(
            field_type2seq2seq_encoder)

        self.action_classifier_feedforward = action_classifier_feedforward
        self.action_num_pop_classifier_feedforward = action_num_pop_classifier_feedforward
        self.action_type_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.action_num_pop_loss = torch.nn.CrossEntropyLoss()
        self.action_type_metrics = {
            "action_type_accuracy": CategoricalAccuracy(),
        }
        self.action_num_pop_metrics = {
            "action_num_pop_accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    @overrides
    def forward(self, instance_type, framework, *args, **kwargs):
        if instance_type[0] == 'action' or instance_type[0] == 'resolve':
            return self._forward_action(instance_type, framework, *args,
                                        **kwargs)
        elif instance_type[0] == 'resolve':
            return self._forward_resolve_framework(instance_type, framework,
                                                   *args, **kwargs)
        else:
            raise NotImplementedError

    def _forward_resolve_framework(self, instance_type, framework, *args,
                                   **kwargs):
        if framework[0] == 'dm':
            return self._forward_resolve_dm(instance_type, framework, *args,
                                            **kwargs)
        elif framework[0] == 'ucca':
            return self._forward_resolve_ucca(instance_type, framework, *args,
                                              **kwargs)
        else:
            raise NotImplementedError

    def _forward_action(
            self,
            instance_type,
            framework,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type: torch.LongTensor,
            action_type_name,
            action_params,
            action_num_pop,
            action_num_pop_mask,
            stack_len,
            token_state,
            token_node_resolveds: Dict[str, torch.LongTensor],
            token_node_labels: Dict[str, torch.LongTensor],
            token_node_prev_actions: Dict[str, torch.LongTensor],
            token_node_ids,
            token_node_childs,
            *args,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        assert all(fr == framework[0] for fr in framework)
        # assert all(name == action_type_name[0] for name in action_type_name)

        logger.debug(('instance_type', instance_type[0]))
        logger.debug(('framework', framework[0]))
        logger.debug(('curr_node_id', curr_node_id[0]))
        logger.debug(('action_types', action_type[0]))
        logger.debug(('action_paramss', action_params[0]))
        logger.debug(('token_states', token_state[0]))
        logger.debug(('token_node_resolveds', token_node_resolveds))
        logger.debug(('token_node_labels', token_node_labels))
        logger.debug(('token_node_prev_actions', token_node_prev_actions))

        # seq2vec features
        seq2vec_fields = [
            # (parse_node_labels, 'word'),
            # (parse_node_lemmas, 'word'),
            # (parse_node_uposs, 'pos'),
            # (parse_node_xposs, 'pos'),
            (token_node_resolveds, 'resolved'),
            (token_node_labels, 'token_node_label'),
            (token_node_prev_actions, 'token_node_prev_action'),
        ]

        embedded_seq2vec_fields = []
        for field, field_type in seq2vec_fields:
            embedded_seq2vec_fields.append(
                self.field_type2embedder[field_type](field))

        logger.debug(
            ('embedded_seq2vec_fields', embedded_seq2vec_fields[0].shape))

        encoded_seq2vec_fields = []
        for seq2vec_field, embedded_field in zip(seq2vec_fields,
                                                 embedded_seq2vec_fields):
            field, field_type = seq2vec_field
            field_mask = util.get_text_field_mask(field)
            logger.debug(('field_mask', field_mask.shape))
            encoded_field = self.field_type2seq2vec_encoder[field_type](
                embedded_field, field_mask)
            encoded_seq2vec_fields.append(encoded_field)

        # seq2seq features
        seq2seq_fields = [
            (token_node_resolveds, 'resolved'),
            (token_node_labels, 'token_node_label'),
        ]

        embedded_seq2seq_fields = []
        for field, field_type in seq2seq_fields:
            embedded_seq2seq_fields.append(
                self.field_type2embedder[field_type](field))

        logger.debug(
            ('embedded_seq2seq_fields', embedded_seq2seq_fields[0].shape))

        encoded_seq2seq_fields = []
        for seq2seq_field, embedded_field in zip(seq2seq_fields,
                                                 embedded_seq2seq_fields):
            field, field_type = seq2seq_field
            field_mask = util.get_text_field_mask(field)
            logger.debug(('field_mask', field_mask.shape))
            encoded_field = self.field_type2seq2seq_encoder[field_type](
                embedded_field, field_mask)
            logger.debug(('seq2seq encoded_field', encoded_field.shape))
            encoded_seq2seq_fields.append(encoded_field)

        encoded_seq2vec_features = torch.cat(encoded_seq2vec_fields, dim=-1)
        encoded_seq2seq_features = torch.cat(encoded_seq2seq_fields, dim=-1)

        # Predict action type
        action_type_logits = self.action_classifier_feedforward(
            encoded_seq2vec_features)
        action_probs, action_preds = action_type_logits.max(1)
        action_resolve_preds = action_preds.eq(self.resolve_tensor)

        if stack_len[0] >= 2:
            logger.debug(('token_node_resolveds', token_node_resolveds))
            # Predict root position
            action_num_pop = action_num_pop.squeeze(-1)
            action_num_pop_logits = self.action_num_pop_classifier_feedforward(
                encoded_seq2seq_features).squeeze(-1)
            action_num_pop_mask = action_num_pop_mask.float().expand_as(
                action_num_pop_logits)
            # action_num_pop_mask = action_num_pop.gt(-1).float().unsqueeze(-1).expand_as(action_num_pop_logits)
            masked_action_num_pop_logits = action_num_pop_logits * action_num_pop_mask

            logger.debug(('masked_action_num_pop_logits',
                          masked_action_num_pop_logits.shape))

        output_dict = {
            'instance_type': instance_type[0],
            'framework': framework[0],
            'action_type_logits': action_type_logits,
            'action_probs': action_probs,
            'action_preds': action_preds,
        }
        logger.debug(('output_dict', output_dict))

        if action_type is not None:
            logger.debug(('action_type_logits', action_type_logits))
            logger.debug(('action_type', action_type))
            action_type_loss = self.action_type_loss(action_type_logits,
                                                     action_type)
            for metric in self.action_type_metrics.values():
                metric(action_type_logits, action_type)
            output_dict['loss'] = action_type_loss

        if action_num_pop is not None and stack_len[0] >= 2:
            logger.debug(('action_num_pop_logits', action_num_pop_logits))
            logger.debug(
                ('masked_action_num_pop_logits', masked_action_num_pop_logits))
            logger.debug(('action_num_pop', action_num_pop))

            action_num_pop_loss = self.action_num_pop_loss(
                masked_action_num_pop_logits, action_num_pop)
            for metric in self.action_num_pop_metrics.values():
                metric(masked_action_num_pop_logits, action_num_pop)
            output_dict['loss'] += action_num_pop_loss

        # raise NotImplementedError
        return output_dict

    def _forward_resolve_dm(
            self,
            instance_type,
            framework,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type: torch.LongTensor,
            action_type_name,
            action_params,
            token_state,
            token_node_resolveds: Dict[str, torch.LongTensor],
            token_node_labels: Dict[str, torch.LongTensor],
            token_node_prev_actions: Dict[str, torch.LongTensor],
            token_node_ids,
            token_node_childs,
    ):
        output_dict = {'loss': None}
        return output_dict

    def _forward_resolve_ucca(
            self,
            instance_type,
            framework,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type: torch.LongTensor,
            action_type_name,
            action_params,
            token_state,
            token_node_resolveds: Dict[str, torch.LongTensor],
            token_node_labels: Dict[str, torch.LongTensor],
            token_node_prev_actions: Dict[str, torch.LongTensor],
            token_node_ids,
            token_node_childs,
    ):
        output_dict = {'loss': None}
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['action_logits'], dim=-1)
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
            for metric_name, metric in list(self.action_type_metrics.items()) +
            list(self.action_num_pop_metrics.items())
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
