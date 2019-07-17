import json
import logging
import random
from collections import deque
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (IndexField, LabelField, MetadataField,
                                  TextField)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

UNKNOWN_WORD = '<UNKOWN-WORD>'
START_PADDING_WORD = '<START-WORD>'
END_PADDING_WORD = '<END-WORD>'

UNKNOWN_POS = '<UNKOWN-POS>'
START_PADDING_POS = '<START-POS>'
END_PADDING_POS = '<END-POS>'

EMPTY_TOKEN_STACK = '<EMPTY-TOKEN-STACK>'
INITIAL_ACTION = '<INITIAL-ACTION>'

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
            field_type2indexers: Dict[str, Dict[str, TokenIndexer]] = None,
            framework_indexers: Dict[str, Dict[str, Dict[
                str, TokenIndexer]]] = None,
            frameworks=['ucca', 'psd', 'eds', 'dm', 'amr'],
            prev_parse_feature_window_size: int = 5,
            next_parse_feature_window_size: int = 5,
            prev_action_window_size: int = 5,
    ) -> None:
        super().__init__(lazy)
        #         self._tokenizer = tokenizer or WordTokenizer()
        self.field_types = [
            'word', 'pos', 'resolved', 'token_node_label',
            'token_node_prev_action'
        ]
        self.field_type2indexers = field_type2indexers or {
            field_type: {
                field_type: SingleIdTokenIndexer(namespace=field_type)
            }
            for field_type in self.field_types
        }
        assert all(field_type in self.field_type2indexers
                   for field_type in self.field_types)

        self.frameworks = frameworks or ['ucca', 'psd', 'eds', 'dm', 'amr']
        if framework_indexers is not None:
            self._framework_indexers = framework_indexers
        else:
            self._framework_indexers = {}
            for framework in self.frameworks:
                framework_indexer = {}
                framework_indexer['token_node_label'] = {
                    'token_node_label':
                    SingleIdTokenIndexer(
                        namespace='{}_token_node_label'.format(framework))
                }
                self._framework_indexers[framework] = framework_indexer
        # logger.debug(('framework_indexers', self._framework_indexers))

        self.prev_parse_feature_window_size = prev_parse_feature_window_size
        self.next_parse_feature_window_size = next_parse_feature_window_size
        self.prev_action_window_size = prev_action_window_size
        self.parse_node_field_name2field_type = {
            'label': 'word',
            'lemma': 'word',
            'upos': 'pos',
            'xpos': 'pos',
        }
        self.token_node_field_name2field_type = {
            'label': 'token_node_label',
            'resolved': 'resolved',
            'prev_action': 'token_node_prev_action',
        }

        self.field_type2PADDINGS = {
            'word': (START_PADDING_WORD, END_PADDING_WORD),
            'pos': (START_PADDING_POS, END_PADDING_POS),
        }
        self.action_type2action_type_name = {
            -1: 'ERROR',
            0: 'APPEND',
            1: 'RESOLVE',
            2: 'IGNORE',
        }

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

        parse_node_field_name2features = {}
        field_name2features = {}

        # Add start padding
        for field_name, field_type in self.parse_node_field_name2field_type.items(
        ):
            START_PADDING = self.field_type2PADDINGS[field_type][0]
            parse_node_field_name2features[field_name] = [
                Token(START_PADDING)
            ] * self.prev_parse_feature_window_size

        # Add features
        for node in parse_nodes:
            tokenized_label = Token(node.get('label', UNKNOWN_WORD))
            parse_node_field_name2features['label'].append(tokenized_label)

            properties = node.get('properties', [])
            values = node.get('values', [])
            for prop, value in zip(properties, values):
                parse_node_field_name2features[prop].append(Token(value))

        # Add end padding
        for field_name, field_type in self.parse_node_field_name2field_type.items(
        ):
            END_PADDING = self.field_type2PADDINGS[field_type][-1]
            parse_node_field_name2features[field_name].extend(
                [Token(END_PADDING)] * self.next_parse_feature_window_size)

        if mrp_meta_data:
            *_, curr_node_ids, token_states, actions = mrp_meta_data
        else:
            return

        action_type_name_deque = deque([Token(INITIAL_ACTION)])
        for action_id, (curr_node_id, action, token_state) in enumerate(
                zip(curr_node_ids, actions, [[]] + token_states)):
            # Action information
            action_type, action_params = action
            action_type_name = self.action_type2action_type_name[action_type]

            token_node_field_name2features = {
                'resolved': [
                    Token('RESOLVED') if token_node[1] else Token('UNRESOLVED')
                    for token_node in token_state
                ] or [Token(EMPTY_TOKEN_STACK)],
                'label': [Token(token_node[2]) for token_node in token_state]
                or [Token(EMPTY_TOKEN_STACK)],
                'prev_action':
                list(action_type_name_deque),
            }
            logger.debug(('token_node_field_name2features',
                          token_node_field_name2features))
            logger.debug(('action', action_type))

            action_type_name_deque.append(Token(action_type_name))
            if len(action_type_name_deque) > self.prev_action_window_size:
                action_type_name_deque.popleft()

            instance_type = 'action'
            yield self.text_to_instance(
                instance_type,
                framework,
                parse_node_field_name2features,
                token_node_field_name2features,
                curr_node_id,
                action_type_name,
                action_params,
                token_state,
            )

            if action_type_name == 'RESOLVE':
                instance_type = 'resolve'
                yield self.text_to_instance(
                    instance_type,
                    framework,
                    parse_node_field_name2features,
                    token_node_field_name2features,
                    curr_node_id,
                    action_type_name,
                    action_params,
                    token_state,
                )

    @overrides
    def text_to_instance(
            self,
            instance_type,
            framework,
            parse_node_field_name2features,
            token_node_field_name2features,
            curr_node_id,
            action_type_name,
            action_params,
            token_state,
    ) -> Instance:  # type: ignore

        field_name2field = {
            'instance_type': MetadataField(instance_type),
            'framework': MetadataField(framework),
        }
        for parse_node_field_name, features in parse_node_field_name2features.items(
        ):
            full_name = 'parse_node_{}s'.format(parse_node_field_name)
            field_type = self.parse_node_field_name2field_type[
                parse_node_field_name]
            if field_type in self.field_type2indexers:
                text_field = TextField(features,
                                       self.field_type2indexers[field_type])
            else:
                raise NotImplementedError
            field_name2field[full_name] = text_field

        if action_type_name:
            field_name2field['action_type'] = LabelField(action_type_name)

        field_name2field['curr_node_id'] = MetadataField(curr_node_id)
        field_name2field['action_params'] = MetadataField(action_params)
        field_name2field['action_type_name'] = MetadataField(action_type_name)
        field_name2field['token_state'] = MetadataField(token_state)

        for token_node_field_name, features in token_node_field_name2features.items(
        ):
            full_name = 'token_node_{}s'.format(token_node_field_name)
            field_type = self.token_node_field_name2field_type[
                token_node_field_name]
            if field_type in self.field_type2indexers:
                text_field = TextField(features,
                                       self.field_type2indexers[field_type])
            else:
                raise NotImplementedError
            field_name2field[full_name] = text_field

        # field_name2field['framework_token_node_labels'] = TextField(
        #     token_node_labels,
        #     self._framework_indexers[framework]['token_node_label'])

        token_node_ids = [token_node[0] for token_node in token_state]
        token_node_childs = [token_node[3] for token_node in token_state]
        field_name2field['token_node_ids'] = MetadataField(token_node_ids)
        field_name2field['token_node_childs'] = MetadataField(
            token_node_childs)

        return Instance(field_name2field)
