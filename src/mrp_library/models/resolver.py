import logging
import pprint
import random
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
from allennlp.training.metrics import CategoricalAccuracy, SequenceAccuracy
from overrides import overrides

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)


# @Model.register("resolver")
class Resolver(Model):
    def __init__(
            self,
            cuda_device,
            vocab: Vocabulary,
            field_type2embedder: Dict[str, TextFieldEmbedder],
            field_type2seq2vec_encoder: Dict[str, Seq2VecEncoder],
            field_type2seq2seq_encoder: Dict[str, Seq2SeqEncoder],
            action_classifier_feedforward: FeedForward,
            action_num_pop_classifier_feedforward: FeedForward,
            framework2field_type2feedforward: Dict[str, FeedForward],
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            framework2field_type2embedder: Dict[
                str, Dict[str, TextFieldEmbedder]] = None,
    ) -> None:
        super(Resolver, self).__init__(vocab, regularizer)
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
        self.framework2field_type2feedforward = framework2field_type2feedforward
        self.action_type_loss = torch.nn.CrossEntropyLoss()
        self.action_num_pop_loss = torch.nn.CrossEntropyLoss()
        self.child_edges_type_loss = torch.nn.CrossEntropyLoss()
        self.root_label_type_loss = torch.nn.CrossEntropyLoss()
        self.action_type_metrics = {
            "action_type_accuracy": CategoricalAccuracy(),
        }
        self.action_num_pop_metrics = {
            "action_num_pop_accuracy": CategoricalAccuracy(),
        }
        self.child_edges_type_metrics = {
            "child_edges_type_accuracy": CategoricalAccuracy(),
        }
        self.root_label_type_metrics = {
            "root_label_type_accuracy": CategoricalAccuracy(),
        }

        initializer(self)

    @overrides
    def forward(self, instance_type, framework, *args, **kwargs):
        logger.debug(('forward', instance_type[0], framework[0]))
        if instance_type[0] == 'action':  # or instance_type[0] == 'resolve':
            return self._forward_action(instance_type, framework, *args,
                                        **kwargs)
        elif instance_type[0] == 'resolve':
            return self._forward_resolve_framework(instance_type, framework,
                                                   *args, **kwargs)
        else:
            raise NotImplementedError

    def _forward_resolve_framework(self, instance_type, framework, *args,
                                   **kwargs):
        if framework[0] in {'dm', 'ucca', 'psd', 'eds', 'amr'}:
            return self._forward_resolve_default(instance_type, framework,
                                                 *args, **kwargs)
        elif framework[0] == 'dm':
            return self._forward_resolve_dm(instance_type, framework, *args,
                                            **kwargs)
        elif framework[0] == 'ucca':
            return self._forward_resolve_ucca(instance_type, framework, *args,
                                              **kwargs)
        elif framework[0] in {'psd', 'eds', 'amr'}:
            return self._forward_resolve_default(instance_type, framework,
                                                 *args, **kwargs)
        else:
            raise NotImplementedError

    def _forward_action(
            self,
            instance_type,
            framework,
            mode,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            parse_node_curr_labels: Dict[str, torch.LongTensor],
            parse_node_curr_lemmas: Dict[str, torch.LongTensor],
            parse_node_curr_uposs: Dict[str, torch.LongTensor],
            parse_node_curr_xposs: Dict[str, torch.LongTensor],
            parse_node_prev_labels: Dict[str, torch.LongTensor],
            parse_node_prev_lemmas: Dict[str, torch.LongTensor],
            parse_node_prev_uposs: Dict[str, torch.LongTensor],
            parse_node_prev_xposs: Dict[str, torch.LongTensor],
            parse_node_next_labels: Dict[str, torch.LongTensor],
            parse_node_next_lemmas: Dict[str, torch.LongTensor],
            parse_node_next_uposs: Dict[str, torch.LongTensor],
            parse_node_next_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type_name,
            action_params,
            stack_len,
            token_state,
            token_node_ids,
            token_node_childs,
            action_type: torch.LongTensor=None,
            action_num_pop=None,
            action_num_pop_mask=None,
            token_node_resolveds: Dict[str, torch.LongTensor]=None,
            token_node_labels: Dict[str, torch.LongTensor]=None,
            token_node_prev_actions: Dict[str, torch.LongTensor]=None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        assert all(fr == framework[0] for fr in framework)
        assert all(slen == stack_len[0] for slen in stack_len)
        # assert all(name == action_type_name[0] for name in action_type_name)
        logger.debug(('stack_len', stack_len[0]))
        logger.debug(('**kwargs', list(kwargs.keys())))
        logger.debug(('instance_type', instance_type[0]))
        logger.debug(('framework', framework[0]))
        logger.debug(('curr_node_id', curr_node_id[0]))
        if action_type is not None:
            logger.debug(('action_types', action_type[0]))
            logger.debug(('action_paramss', action_params[0]))
        logger.debug(('token_states', token_state[0]))
        logger.debug(('token_node_resolveds', token_node_resolveds))
        logger.debug(('token_node_labels', token_node_labels))
        logger.debug(('token_node_prev_actions', token_node_prev_actions))

        # Embedding field
        action_type_embedding_fields = [
            (parse_node_curr_labels, 'parse_curr_word'),
            (parse_node_curr_lemmas, 'parse_curr_word'),
            (parse_node_curr_uposs, 'parse_curr_pos'),
            (parse_node_curr_xposs, 'parse_curr_pos'),
        ]
        action_type_embedded_embedding_fields = []
        for field, field_type in action_type_embedding_fields:
            action_type_embedded_embedding_fields.append(
                self.field_type2embedder[field_type](field))

        action_num_pop_embedding_fields = [
            (token_node_resolveds, 'resolved'),
            (token_node_labels, 'token_node_label'),
        ]

        action_num_pop_embedded_embedding_fields = []
        for field, field_type in action_num_pop_embedding_fields:
            action_num_pop_embedded_embedding_fields.append(
                self.field_type2embedder[field_type](field))

        # seq2vec features
        seq2vec_fields = [
            (parse_node_curr_labels, 'parse_curr_word'),
            (parse_node_curr_lemmas, 'parse_curr_word'),
            (parse_node_curr_uposs, 'parse_curr_pos'),
            (parse_node_curr_xposs, 'parse_curr_pos'),
            (parse_node_prev_labels, 'parse_prev_word'),
            (parse_node_prev_lemmas, 'parse_prev_word'),
            (parse_node_prev_uposs, 'parse_prev_pos'),
            (parse_node_prev_xposs, 'parse_prev_pos'),
            (parse_node_next_labels, 'parse_next_word'),
            (parse_node_next_lemmas, 'parse_next_word'),
            (parse_node_next_uposs, 'parse_next_pos'),
            (parse_node_next_xposs, 'parse_next_pos'),
            (token_node_resolveds, 'token_node_resolved'),
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

        action_type_embedded_embedding_features = torch.cat(
            action_type_embedded_embedding_fields, dim=-1).squeeze(1)
        action_num_pop_embedded_embedding_features = torch.cat(
            action_num_pop_embedded_embedding_fields, dim=-1)

        encoded_seq2vec_features = torch.cat(encoded_seq2vec_fields, dim=-1)
        encoded_seq2seq_features = torch.cat(encoded_seq2seq_fields, dim=-1)
        logger.debug((
            ('action_type_embedded_embedding_features',
             action_type_embedded_embedding_features.shape),
            ('encoded_seq2vec_features', encoded_seq2vec_features.shape),
            ('encoded_seq2seq_features', encoded_seq2seq_features.shape),
        ))

        action_type_features = torch.cat([
            action_type_embedded_embedding_features, encoded_seq2vec_features
        ],
                                         dim=-1)

        # Predict action type
        action_type_logits = self.action_classifier_feedforward(
            action_type_features)
        action_type_probs, action_type_preds = action_type_logits.max(1)
        action_resolve_preds = action_type_preds.eq(self.resolve_tensor)

        output_dict = {
            'instance_type': instance_type[0],
            'framework': framework[0],
            'action_type_logits': action_type_logits,
            'action_type_probs': action_type_probs,
            'action_type_preds': action_type_preds,
        }

        # Predict action num pop stack length >= 2
        if stack_len[0] >= 2:
            logger.debug(('token_node_resolveds', token_node_resolveds))
            # Predict number of pops
            action_num_pop_features = torch.cat([
                action_num_pop_embedded_embedding_features,
                encoded_seq2seq_features
            ],
                                                dim=-1)
            logger.debug(
                ('action_num_pop_features', action_num_pop_features.shape))
            action_num_pop_logits = self.action_num_pop_classifier_feedforward(
                action_num_pop_features)
            logger.debug(
                ('action_num_pop_logits', action_num_pop_logits.shape))
            action_num_pop_logits = action_num_pop_logits.squeeze(-1)

            if mode[0] == 'predict':
                masked_action_num_pop_logits = action_num_pop_logits
            else:
                action_num_pop_mask = action_num_pop_mask.float().expand_as(
                    action_num_pop_logits)
                # action_num_pop_mask = action_num_pop.gt(-1).float().unsqueeze(-1).expand_as(action_num_pop_logits)
                masked_action_num_pop_logits = action_num_pop_logits * action_num_pop_mask
            masked_action_num_pop_probs, masked_action_num_pop_preds = masked_action_num_pop_logits.max(
                1)
            logger.debug(('masked_action_num_pop_logits',
                          masked_action_num_pop_logits.shape))
            output_dict[
                'masked_action_num_pop_logits'] = masked_action_num_pop_logits
            output_dict[
                'masked_action_num_pop_probs'] = masked_action_num_pop_probs
            output_dict[
                'masked_action_num_pop_preds'] = masked_action_num_pop_preds

        logger.debug(('output_dict', output_dict))

        rand_int = random.randint(0, 100)
        if not rand_int:
            logger.debug(('seq2vec fields', seq2vec_fields))
        if action_type is not None and framework[0] != 'ucca':
            logger.debug(('action_type_logits', action_type_logits))
            logger.debug(('action_type_preds', action_type_preds))
            logger.debug(('action_type', action_type))
            logger.debug(('action_type_logits', action_type_logits))
            logger.debug(('action_type_logits', action_type_logits.shape))
            logger.debug(('action_type', action_type))
            logger.debug(('action_type', action_type.shape))
            action_type_loss = self.action_type_loss(action_type_logits,
                                                     action_type)
            for metric in self.action_type_metrics.values():
                metric(action_type_logits, action_type)
            output_dict['loss'] = action_type_loss

        if action_num_pop is not None and stack_len[0] >= 2:
            action_num_pop = action_num_pop.squeeze(-1)
            if not rand_int:
                logger.debug(
                    ('action_num_pop_logits', action_num_pop_logits))
                logger.debug(('masked_action_num_pop_logits',
                                masked_action_num_pop_logits))
                logger.debug(('masked_action_num_pop_preds',
                                masked_action_num_pop_preds))
                logger.debug(('action_num_pop', action_num_pop))
            logger.debug(('action_num_pop_logits', action_num_pop_logits))
            logger.debug(
                ('masked_action_num_pop_logits', masked_action_num_pop_logits))
            logger.debug(('action_num_pop', action_num_pop))

            action_num_pop_loss = self.action_num_pop_loss(
                masked_action_num_pop_logits, action_num_pop)
            for metric in self.action_num_pop_metrics.values():
                metric(masked_action_num_pop_logits, action_num_pop)
            if 'loss' not in output_dict:
                output_dict['loss'] = action_num_pop_loss
            else:
                output_dict['loss'] += action_num_pop_loss

        # raise NotImplementedError
        return output_dict

    def _forward_resolve_default(
            self,
            instance_type,
            framework,
            mode,
            stack_len,
            parse_node_labels: Dict[str, torch.LongTensor],
            parse_node_lemmas: Dict[str, torch.LongTensor],
            parse_node_uposs: Dict[str, torch.LongTensor],
            parse_node_xposs: Dict[str, torch.LongTensor],
            curr_node_id,
            action_type_name,
            action_params,
            token_state,
            token_node_resolveds: Dict[str, torch.LongTensor],
            token_node_labels: Dict[str, torch.LongTensor],
            token_node_prev_actions: Dict[str, torch.LongTensor],
            token_node_ids,
            token_node_childs,
            resolved_root_parse_node_label: Dict[str, torch.LongTensor],
            resolved_root_parse_node_lemma: Dict[str, torch.LongTensor],
            resolved_root_parse_node_upos: Dict[str, torch.LongTensor],
            resolved_root_parse_node_xpos: Dict[str, torch.LongTensor],
            action_type: torch.LongTensor = None,
            resolved_child_parse_node_labels: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_lemmas: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_uposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_child_parse_node_xposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_label_root_label=None,
            resolved_label_edge_labels=None,
            resolved_child_parse_node_ids=None,
            **kwargs,
    ):
        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))
        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))

        resolved_root_fields = [
            (resolved_root_parse_node_label,
             'resolve_{}_word'.format(framework[0])),
            (resolved_root_parse_node_lemma,
             'resolve_{}_word'.format(framework[0])),
            (resolved_root_parse_node_upos,
             'resolve_{}_pos'.format(framework[0])),
            (resolved_root_parse_node_xpos,
             'resolve_{}_pos'.format(framework[0])),
        ]

        embedded_resolved_root_fields = []
        for field, field_type in resolved_root_fields:
            embedded_resolved_root_fields.append(
                self.field_type2embedder[field_type](field))
            logger.debug(('embedded_resolved_root_fields',
                          embedded_resolved_root_fields[0].shape))

        embedded_resolved_root_features = torch.cat(
            embedded_resolved_root_fields, dim=-1)
        root_node_label_feedforward = self.framework2field_type2feedforward[
            framework[0]]['root_label']
        root_node_label_logits = root_node_label_feedforward(
            embedded_resolved_root_features).squeeze(1)

        # Single node resolve (Predict propagate node)
        is_single_node_resolve = stack_len[0] == 1
        logger.warning(('stack_len', stack_len))

        if is_single_node_resolve:
            output_dict = {
                'instance_type': instance_type[0],
                'framework': framework[0],
                'resolve_type': 'single_node',
            }
            logger.debug(
                ('root_node_label_logits', root_node_label_logits))
            logger.debug(
                ('resolved_label_root_label', resolved_label_root_label))

            if mode[0] == 'train':
                root_label_type_loss = self.root_label_type_loss(
                    root_node_label_logits, resolved_label_root_label)
                for metric in self.root_label_type_metrics.values():
                    metric(root_node_label_logits, resolved_label_root_label)
                output_dict['loss'] = root_label_type_loss

            output_dict['root_node_label_logits'] = root_node_label_logits
            return output_dict

        # seq2seq features
        seq2seq_fields = [
            (resolved_child_parse_node_labels,
             'resolve_{}_word'.format(framework[0])),
            (resolved_child_parse_node_lemmas,
             'resolve_{}_word'.format(framework[0])),
            (resolved_child_parse_node_uposs,
             'resolve_{}_pos'.format(framework[0])),
            (resolved_child_parse_node_xposs,
             'resolve_{}_pos'.format(framework[0])),
        ]

        logger.warning(('seq2seq_fields', seq2seq_fields))
        embedded_seq2seq_fields = []
        for field, field_type in seq2seq_fields:
            embedded_seq2seq_fields.append(
                self.field_type2embedder[field_type](field))

        logger.warning(
            ('embedded_seq2seq_fields', embedded_seq2seq_fields[0].shape))

        encoded_seq2seq_fields = []
        for seq2seq_field, embedded_field in zip(seq2seq_fields,
                                                 embedded_seq2seq_fields):
            field, field_type = seq2seq_field
            field_mask = util.get_text_field_mask(field)
            logger.debug(('field_mask', field_mask.shape))
            encoded_field = self.field_type2seq2seq_encoder[field_type](
                embedded_field, field_mask)
            # print(field_type, field, embedded_field, field_mask)
            # print(embedded_field.shape, field_mask.shape)
            ##  print(resolved_label_edge_labels)
            encoded_seq2seq_fields.append(encoded_field)

        encoded_seq2seq_features = torch.cat(encoded_seq2seq_fields, dim=-1)
        logger.warning(('encoded_seq2seq_features', encoded_seq2seq_features.shape))
        logger.warning(('framework[0]', framework[0]))
        child_edges_feedforward = self.framework2field_type2feedforward[
            framework[0]]['child_edges']

        child_edges_type_logits = child_edges_feedforward(
            encoded_seq2seq_features)
        # child_edges_type_logits = child_edges_feedforward(
        #     encoded_seq2seq_features).transpose(1, 2)
        logger.warning(('child_edges_type_logits', child_edges_type_logits))
        child_edges_probs, child_edges_preds = child_edges_type_logits.max(1)
        logger.debug((child_edges_type_logits.shape))

        output_dict = {
            'instance_type': instance_type[0],
            'framework': framework[0],
            'child_edges_type_logits': child_edges_type_logits,
            'child_edges_probs': child_edges_probs,
            'child_edges_preds': child_edges_preds,
        }
        logger.debug(('output_dict', output_dict))

        if resolved_label_edge_labels is not None:
            logger.debug(('child_edges_type_logits', child_edges_type_logits))
            logger.debug(
                ('child_edges_type_logits', child_edges_type_logits.shape))
            logger.debug(
                ('resolved_label_edge_labels', resolved_label_edge_labels))
            child_edges_type_loss = self.child_edges_type_loss(
                child_edges_type_logits, resolved_label_edge_labels)

            class_size = self.vocab.get_vocab_size(
                '{}_resolved_label_edge_labels'.format(framework[0]))
            for metric in self.child_edges_type_metrics.values():
                # metric(child_edges_type_logits, resolved_label_edge_labels)
                logger.info(('child_edges_preds', child_edges_preds.shape))
                logger.info(
                    ('child_edges_type_logits', child_edges_type_logits.shape))
                logger.info(('resolved_label_edge_labels',
                             resolved_label_edge_labels.shape))
                metric(
                    child_edges_type_logits.contiguous().view(-1, class_size),
                    resolved_label_edge_labels.view(-1))
            output_dict['loss'] = child_edges_type_loss

        logger.debug(kwargs.keys())
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
            resolved_root_parse_node_label: Dict[str, torch.LongTensor],
            resolved_root_parse_node_lemma: Dict[str, torch.LongTensor],
            resolved_root_parse_node_upos: Dict[str, torch.LongTensor],
            resolved_root_parse_node_xpos: Dict[str, torch.LongTensor],
            resolved_child_parse_node_labels: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_lemmas: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_uposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_child_parse_node_xposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_label_root_label=None,
            resolved_label_edge_labels=None,
            resolved_child_parse_node_ids=None,
            **kwargs,
    ):
        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))
        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))

        resolved_root_fields = [
            (resolved_root_parse_node_label, 'resolve_dm_word'),
            (resolved_root_parse_node_lemma, 'resolve_dm_word'),
            (resolved_root_parse_node_upos, 'resolve_dm_pos'),
            (resolved_root_parse_node_xpos, 'resolve_dm_pos'),
        ]

        embedded_resolved_root_fields = []
        for field, field_type in resolved_root_fields:
            embedded_resolved_root_fields.append(
                self.field_type2embedder[field_type](field))
            logger.debug(('embedded_resolved_root_fields',
                          embedded_resolved_root_fields[0].shape))

        embedded_resolved_root_features = torch.cat(
            embedded_resolved_root_fields, dim=-1)
        propagate_node_label_feedforward = self.framework2field_type2feedforward[
            'dm']['root_label']
        propagate_node_label_logits = propagate_node_label_feedforward(
            embedded_resolved_root_features).squeeze(1)

        # Single node resolve (Predict propagate node)
        is_single_node_resolve = any(
            dim == 0 for dim in resolved_label_edge_labels.size())

        if is_single_node_resolve:
            output_dict = {
                'instance_type': instance_type[0],
                'framework': framework[0],
                'resolve_type': 'single_node',
            }
            logger.debug(
                ('propagate_node_label_logits', propagate_node_label_logits))
            logger.debug(
                ('resolved_label_root_label', resolved_label_root_label))
            root_label_type_loss = self.root_label_type_loss(
                propagate_node_label_logits, resolved_label_root_label)
            for metric in self.root_label_type_metrics.values():
                metric(propagate_node_label_logits, resolved_label_root_label)
            output_dict[
                'propagate_node_label_logits'] = propagate_node_label_logits
            output_dict['loss'] = root_label_type_loss
            return output_dict

        # seq2seq features
        seq2seq_fields = [
            (resolved_child_parse_node_labels, 'resolve_dm_word'),
            (resolved_child_parse_node_lemmas, 'resolve_dm_word'),
            (resolved_child_parse_node_uposs, 'resolve_dm_pos'),
            (resolved_child_parse_node_xposs, 'resolve_dm_pos'),
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
            # print(field_type, field, embedded_field, field_mask)
            # print(embedded_field.shape, field_mask.shape)
            ##  print(resolved_label_edge_labels)
            encoded_seq2seq_fields.append(encoded_field)

        encoded_seq2seq_features = torch.cat(encoded_seq2seq_fields, dim=-1)
        child_edges_feedforward = self.framework2field_type2feedforward['dm'][
            'child_edges']
        child_edges_type_logits = child_edges_feedforward(
            encoded_seq2seq_features).transpose(1, 2)
        child_edges_probs, child_edges_preds = child_edges_type_logits.max(1)
        logger.debug((child_edges_type_logits.shape))

        output_dict = {
            'instance_type': instance_type[0],
            'framework': framework[0],
            'child_edges_type_logits': child_edges_type_logits.shape,
            'child_edges_probs': child_edges_probs.shape,
            'child_edges_preds': child_edges_preds.shape,
            'resolved_label_edge_labels': resolved_label_edge_labels.shape,
        }
        logger.debug(('output_dict', output_dict))

        if resolved_label_edge_labels is not None:
            logger.debug(('child_edges_type_logits', child_edges_type_logits))
            logger.debug(
                ('child_edges_type_logits', child_edges_type_logits.shape))
            logger.debug(
                ('resolved_label_edge_labels', resolved_label_edge_labels))
            child_edges_type_loss = self.child_edges_type_loss(
                child_edges_type_logits, resolved_label_edge_labels)

            class_size = self.vocab.get_vocab_size(
                'dm_resolved_label_edge_labels')
            for metric in self.child_edges_type_metrics.values():
                # metric(child_edges_type_logits, resolved_label_edge_labels)
                logger.info(('child_edges_preds', child_edges_preds.shape))
                logger.info(
                    ('child_edges_type_logits', child_edges_type_logits.shape))
                logger.info(('resolved_label_edge_labels',
                             resolved_label_edge_labels.shape))
                metric(
                    child_edges_type_logits.contiguous().view(-1, class_size),
                    resolved_label_edge_labels.view(-1))
            output_dict['loss'] = child_edges_type_loss

        logger.debug(kwargs.keys())
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
            resolved_root_parse_node_label: Dict[str, torch.LongTensor],
            resolved_root_parse_node_lemma: Dict[str, torch.LongTensor],
            resolved_root_parse_node_upos: Dict[str, torch.LongTensor],
            resolved_root_parse_node_xpos: Dict[str, torch.LongTensor],
            resolved_child_parse_node_labels: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_lemmas: Dict[str, torch.
                                                   LongTensor] = None,
            resolved_child_parse_node_uposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_child_parse_node_xposs: Dict[str, torch.
                                                  LongTensor] = None,
            resolved_label_root_label=None,
            resolved_label_edge_labels=None,
            resolved_child_parse_node_ids=None,
            **kwargs,
    ):

        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))
        logger.debug(
            ('resolved_label_edge_labels', resolved_label_edge_labels))

        resolved_root_fields = [
            (resolved_root_parse_node_label, 'resolve_ucca_word'),
            (resolved_root_parse_node_lemma, 'resolve_ucca_word'),
            (resolved_root_parse_node_upos, 'resolve_ucca_pos'),
            (resolved_root_parse_node_xpos, 'resolve_ucca_pos'),
        ]

        embedded_resolved_root_fields = []
        for field, field_type in resolved_root_fields:
            embedded_resolved_root_fields.append(
                self.field_type2embedder[field_type](field))
            logger.debug(('embedded_resolved_root_fields',
                          embedded_resolved_root_fields[0].shape))

        embedded_resolved_root_features = torch.cat(
            embedded_resolved_root_fields, dim=-1)
        propagate_node_label_feedforward = self.framework2field_type2feedforward[
            'ucca']['root_label']
        propagate_node_label_logits = propagate_node_label_feedforward(
            embedded_resolved_root_features).squeeze(1)

        # Single node resolve (Predict propagate node)
        is_single_node_resolve = any(
            dim == 0 for dim in resolved_label_edge_labels.size())

        if is_single_node_resolve:
            output_dict = {
                'instance_type': instance_type[0],
                'framework': framework[0],
                'resolve_type': 'single_node',
            }
            logger.debug(
                ('propagate_node_label_logits', propagate_node_label_logits))
            logger.debug(('propagate_node_label_logits',
                          propagate_node_label_logits.shape))
            logger.debug(
                ('resolved_label_root_label', resolved_label_root_label))
            logger.debug(
                ('resolved_label_root_label', resolved_label_root_label.shape))
            root_label_type_loss = self.root_label_type_loss(
                propagate_node_label_logits, resolved_label_root_label)
            for metric in self.root_label_type_metrics.values():
                metric(propagate_node_label_logits, resolved_label_root_label)
            output_dict[
                'propagate_node_label_logits'] = propagate_node_label_logits
            output_dict['loss'] = root_label_type_loss
            return output_dict

        # seq2seq features
        seq2seq_fields = [
            (resolved_child_parse_node_labels, 'resolve_ucca_word'),
            (resolved_child_parse_node_lemmas, 'resolve_ucca_word'),
            (resolved_child_parse_node_uposs, 'resolve_ucca_pos'),
            (resolved_child_parse_node_xposs, 'resolve_ucca_pos'),
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
            # print(field_type, field, embedded_field, field_mask)
            # print(embedded_field.shape, field_mask.shape)
            ##  print(resolved_label_edge_labels)
            encoded_seq2seq_fields.append(encoded_field)

        encoded_seq2seq_features = torch.cat(encoded_seq2seq_fields, dim=-1)
        child_edges_feedforward = self.framework2field_type2feedforward[
            'ucca']['child_edges']
        child_edges_type_logits = child_edges_feedforward(
            encoded_seq2seq_features).transpose(1, 2)
        child_edges_probs, child_edges_preds = child_edges_type_logits.max(1)
        logger.debug((child_edges_type_logits.shape))

        output_dict = {
            'instance_type': instance_type[0],
            'framework': framework[0],
            'child_edges_type_logits': child_edges_type_logits.shape,
            'child_edges_probs': child_edges_probs.shape,
            'child_edges_preds': child_edges_preds.shape,
            'resolved_label_edge_labels': resolved_label_edge_labels.shape,
        }
        logger.debug(('output_dict', output_dict))

        if resolved_label_edge_labels is not None:
            logger.debug(('child_edges_type_logits', child_edges_type_logits))
            logger.debug(
                ('child_edges_type_logits', child_edges_type_logits.shape))
            logger.debug(
                ('resolved_label_edge_labels', resolved_label_edge_labels))
            logger.debug(('resolved_label_edge_labels',
                          resolved_label_edge_labels.shape))

            child_edges_type_loss = self.child_edges_type_loss(
                child_edges_type_logits, resolved_label_edge_labels)
            # for metric in self.child_edges_type_metrics.values():
            #     metric(child_edges_type_logits.transpose(1, 2), resolved_label_edge_labels)
            output_dict['loss'] = child_edges_type_loss

        logger.debug(kwargs.keys())
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        instance_type = output_dict.get('instance_type')
        framework = output_dict.get('framework')
        if instance_type == 'action':
            class_probabilities = F.softmax(output_dict['action_type_logits'],
                                            dim=-1)
            output_dict['class_probabilities'] = class_probabilities

            predictions = class_probabilities.cpu().data.numpy()
            argmax_indices = numpy.argmax(predictions, axis=-1)
            labels = [
                self.vocab.get_token_from_index(x,
                                                namespace="action_type_label")
                for x in argmax_indices
            ]
            output_dict['label'] = labels
        elif instance_type == 'resolve':
            pass
        else:
            raise NotImplementedError

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in list(self.action_type_metrics.items()) +
            list(self.action_num_pop_metrics.items()) +
            list(self.root_label_type_metrics.items()) +
            list(self.child_edges_type_metrics.items())
        }
