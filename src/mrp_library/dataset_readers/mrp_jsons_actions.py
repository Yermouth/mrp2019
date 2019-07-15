import json
import logging
import random
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import IndexField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

UNKNOWN_POS = '<UNKOWN-POS>'
START_PADDING_POS = '<START-POS>'
END_PADDING_POS = '<END-POS>'

UNKNOWN_LABEL = '<UNKOWN-LABEL>'
START_PADDING_LABEL = '<START-LABEL>'
END_PADDING_LABEL = '<END-LABEL>'

# Actions
ERROR = -1
APPEND = 0  # APPEND_TO_CURRENT_GROUP
RESOLVE = 1  # RESOLVE_GROUP
IGNORE = 2  # IGNORE_CURRENT_TOKEN


# @DatasetReader.register("mrp_json_reader")
class MRPDatasetActionReader(DatasetReader):
    def __init__(
            self,
            lazy: bool = False,
            #             tokenizer: Tokenizer = None,
            word_indexers: Dict[str, TokenIndexer] = None,
            pos_indexers: Dict[str, TokenIndexer] = None,
            prev_parse_feature_window_size: int = 5,
            next_parse_feature_window_size: int = 5,
    ) -> None:
        super().__init__(lazy)
        #         self._tokenizer = tokenizer or WordTokenizer()
        self._word_indexers = word_indexers or {
            "word": SingleIdTokenIndexer(namespace='word'),
        }
        self._pos_indexers = pos_indexers or {
            'pos': SingleIdTokenIndexer(namespace='pos'),
        }
        self.prev_parse_feature_window_size = prev_parse_feature_window_size
        self.next_parse_feature_window_size = next_parse_feature_window_size

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s",
                        file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue

                data_instance = json.loads(line)
                yield from self.process_mrp_graph(data_instance)

    def process_mrp_graph(self, data_instance):
        mrp_json = data_instance.get('mrp_json', {})
        parse_json = data_instance.get('parse_json', {})
        mrp_parser_states = data_instance.get('mrp_parser_states', {})
        mrp_meta_data = data_instance.get('mrp_meta_data', {})
        companion_parser_states = data_instance.get('companion_parser_states',
                                                    {})
        companion_meta_data = data_instance.get('companion_meta_data', {})

        # Generate features
        framework = mrp_json.get('framework', '')
        parse_nodes = parse_json.get('nodes', [])
        parse_node_labels = []
        parse_node_lemmas = []
        parse_node_uposs = []
        parse_node_xposs = []

        for node in parse_nodes:
            parse_node_labels.append(Token(node.get('label', UNKNOWN_LABEL)))
            for prop, value in zip(
                    node.get('properties', []), node.get('values', [])):
                if prop == 'lemma':
                    parse_node_lemmas.append(Token(value))
                if prop == 'upos':
                    parse_node_uposs.append(Token(value))
                if prop == 'xpos':
                    parse_node_xposs.append(Token(value))

        if mrp_meta_data:
            *_, curr_node_ids, token_states, actions = mrp_meta_data
        else:
            return

        for curr_node_id, action, token_state in zip(curr_node_ids, actions,
                                                     [[]] + token_states):
            # Current node information
            curr_parse_node_label = parse_node_labels[curr_node_id]
            curr_parse_node_lemma = parse_node_lemmas[curr_node_id]
            curr_parse_node_upos = parse_node_uposs[curr_node_id]
            curr_parse_node_xpos = parse_node_xposs[curr_node_id]

            # Previous node information
            prev_low_bound = max(
                curr_node_id - self.prev_parse_feature_window_size, 0)
            prev_padding = max(
                self.prev_parse_feature_window_size - curr_node_id, 0)
            prev_parse_node_labels = [
                Token(START_PADDING_LABEL)
            ] * prev_padding + parse_node_labels[prev_low_bound:curr_node_id]
            prev_parse_node_lemmas = [
                Token(START_PADDING_LABEL)
            ] * prev_padding + parse_node_lemmas[prev_low_bound:curr_node_id]
            prev_parse_node_uposs = [
                Token(START_PADDING_POS)
            ] * prev_padding + parse_node_uposs[prev_low_bound:curr_node_id]
            prev_parse_node_xposs = [
                Token(START_PADDING_POS)
            ] * prev_padding + parse_node_xposs[prev_low_bound:curr_node_id]

            # Next node information
            next_low_bound = max(
                curr_node_id - self.next_parse_feature_window_size, 0)
            next_padding = max(
                self.next_parse_feature_window_size - curr_node_id, 0)
            next_parse_node_labels = parse_node_labels[next_low_bound:
                                                       curr_node_id] + [
                                                           Token(
                                                               END_PADDING_LABEL
                                                           )
                                                       ] * next_padding
            next_parse_node_lemmas = parse_node_lemmas[next_low_bound:
                                                       curr_node_id] + [
                                                           Token(
                                                               END_PADDING_LABEL
                                                           )
                                                       ] * next_padding
            next_parse_node_uposs = parse_node_uposs[next_low_bound:
                                                     curr_node_id] + [
                                                         Token(END_PADDING_POS)
                                                     ] * next_padding
            next_parse_node_xposs = parse_node_xposs[next_low_bound:
                                                     curr_node_id] + [
                                                         Token(END_PADDING_POS)
                                                     ] * next_padding

            # Action information
            action_type, params = action
            action_type = str(action_type)

            if action_type == RESOLVE:
                (num_pop, root_position, resolved_node,
                 resolved_edges) = params
                yield self.text_to_instance(
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
                    action_type,
                    num_pop=num_pop,
                    root_position=root_position,
                )
            else:
                yield self.text_to_instance(
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
                    action_type,
                )

    @overrides
    def text_to_instance(
            self,
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
            action_type,
            num_pop=None,
            root_position=None,
    ) -> Instance:  # type: ignore

        parse_features = [
            ('curr_parse_node_label', [curr_parse_node_label], 'word'),
            ('curr_parse_node_lemma', [curr_parse_node_lemma], 'word'),
            ('curr_parse_node_upos', [curr_parse_node_upos], 'pos'),
            ('curr_parse_node_xpos', [curr_parse_node_xpos], 'pos'),
            ('prev_parse_node_labels', prev_parse_node_labels, 'word'),
            ('prev_parse_node_lemmas', prev_parse_node_lemmas, 'word'),
            ('prev_parse_node_uposs', prev_parse_node_uposs, 'pos'),
            ('prev_parse_node_xposs', prev_parse_node_xposs, 'pos'),
            ('next_parse_node_labels', next_parse_node_labels, 'word'),
            ('next_parse_node_lemmas', next_parse_node_lemmas, 'word'),
            ('next_parse_node_uposs', next_parse_node_uposs, 'pos'),
            ('next_parse_node_xposs', next_parse_node_xposs, 'pos'),
        ]

        instance_fields = {}
        for field_name, field, field_type in parse_features:
            # logger.debug(('field', field_name))
            if field_type == 'word':
                instance_fields[field_name] = TextField(
                    field, self._word_indexers)
            elif field_type == 'pos':
                instance_fields[field_name] = TextField(
                    field, self._pos_indexers)

        if action_type:
            instance_fields['action_type'] = LabelField(action_type)
        return Instance(instance_fields)
