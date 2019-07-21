import json
import logging
import pprint
import random
from collections import deque
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (IndexField, LabelField, MetadataField,
                                  SequenceLabelField, TextField)
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

NO_MORE_PARSE_NODE = '<NO-MORE-PARSE-NODE>'

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
        self.parse_node_basic_field_names = list(
            self.parse_node_field_name2field_type.keys())

        # Resolving edge node labels requires parse_node information
        self.resolved_field_name2field_type = {
            'root_label': 'LabelField',
            'edge_labels': 'SequenceLabelField',
        }
        for parse_node_field_name, field_type in self.parse_node_field_name2field_type.items(
        ):
            full_name = 'parse_node_{}'.format(parse_node_field_name)
            self.resolved_field_name2field_type[full_name] = field_type

        # Add state feature for parse_nodes
        self.state_templates = ['curr_{}', 'prev_{}', 'next_{}']
        self.parse_node_state_field_name2field_type = {}
        for state_template in self.state_templates:
            for parse_node_field_name, field_type in self.parse_node_field_name2field_type.items(
            ):
                state_field_name = state_template.format(parse_node_field_name)
                self.parse_node_state_field_name2field_type[
                    state_field_name] = field_type

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
        self.error_count = 0

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

    def _generate_parse_node_field_name2features(self, parse_nodes):

        parse_node_field_name2features = {}
        for field_name, field_type in self.parse_node_field_name2field_type.items(
        ):
            parse_node_field_name2features[field_name] = []

        # Add parse_node features
        for node in parse_nodes:
            tokenized_label = Token(node.get('label', UNKNOWN_WORD).lower())
            parse_node_field_name2features['label'].append(tokenized_label)

            properties = node.get('properties', [])
            values = node.get('values', [])
            for prop, value in zip(properties, values):
                parse_node_field_name2features[prop].append(Token(value))
        return parse_node_field_name2features

    def _generate_parse_node_state_field_name2features(
            self,
            curr_node_id,
            parse_nodes_size,
            parse_node_field_name2features,
    ):
        parse_node_state_field_name2features = {}
        for state_template in self.state_templates:
            for field_name in self.parse_node_basic_field_names:
                field_type = self.parse_node_field_name2field_type[field_name]
                full_name = state_template.format(field_name)
                parse_node_features = parse_node_field_name2features[
                    field_name]
                if state_template == 'curr_{}':
                    if curr_node_id >= parse_nodes_size:
                        features = [Token(NO_MORE_PARSE_NODE)]
                    else:
                        features = [parse_node_features[curr_node_id]]
                elif state_template == 'prev_{}':
                    #prev_padding_size = self.prev_parse_feature_window_size - curr_node_id - 1
                    features = parse_node_features[:
                                                   curr_node_id][:self.
                                                                 prev_parse_feature_window_size]
                    if not features:
                        features = [
                            Token(self.field_type2PADDINGS[field_type][0])
                        ]
                elif state_template == 'next_{}':
                    features = parse_node_features[
                        curr_node_id +
                        1:][:self.next_parse_feature_window_size]
                    if not features:
                        features = [
                            Token(self.field_type2PADDINGS[field_type][1])
                        ]
                else:
                    raise NotImplementedError
                logger.debug(('features', state_template, field_type,
                              parse_node_features, features))
                parse_node_state_field_name2features[full_name] = features
        return parse_node_state_field_name2features

    def _generate_token_node_field_name2features(self, token_state,
                                                 action_type_name_deque):
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
        return token_node_field_name2features

    def _generate_resolved_field_name2features(self,
                                               token_state,
                                               parse_nodes_size,
                                               parse_node_field_name2features,
                                               framework,
                                               action_num_pop,
                                               resolved_node_position=None,
                                               resolved_node=None,
                                               resolved_edgess=None,
                                               predict=False):

        # initialization
        resolved_child_field_name2features = {}
        resolved_root_field_name2features = {}
        for parse_node_field_name, features in parse_node_field_name2features.items(
        ):
            resolved_child_field_name2features['parse_node_' +
                                               parse_node_field_name] = []
            resolved_root_field_name2features['parse_node_' +
                                              parse_node_field_name] = []

        resolved_field_name2label = {'edge_labels': []}
        resolved_field_name2metadata = {}

        # Find parse node id
        resolved_child_parse_node_ids = []
        resolved_root_parse_node_id = token_state[-1][0]

        logger.warning(('token_state', token_state))
        if action_num_pop >= 2:
            for token_node_id, resolved, *_ in token_state[-action_num_pop:]:
                logger.warning(('token_node_id', token_node_id, resolved))
                if resolved:
                    resolved_child_parse_node_ids.append(token_node_id)
                else:
                    resolved_root_parse_node_id = token_node_id
        logger.warning(('action_num_pop', action_num_pop))
        logger.warning(('resolved_child_parse_node_ids',
                        resolved_child_parse_node_ids))
        logger.warning(('resolved_root_parse_node_id',
                        resolved_root_parse_node_id))

        resolved_field_name2metadata[
            'resolved_child_parse_node_ids'] = resolved_child_parse_node_ids
        resolved_field_name2metadata[
            'resolved_root_parse_node_ids'] = resolved_root_parse_node_id

        # Resolved root features
        if resolved_root_parse_node_id >= parse_nodes_size:
            for parse_node_field_name, features in parse_node_field_name2features.items(
            ):
                resolved_root_field_name2features[
                    'parse_node_' + parse_node_field_name].append(
                        Token('UNKWOWN'))
        else:
            for parse_node_field_name, features in parse_node_field_name2features.items(
            ):
                resolved_root_field_name2features[
                    'parse_node_' + parse_node_field_name].append(
                        features[resolved_root_parse_node_id])

        # Resolved child features
        for node_id in resolved_child_parse_node_ids:
            # Abstract node
            if node_id >= parse_nodes_size:
                for parse_node_field_name, features in parse_node_field_name2features.items(
                ):
                    resolved_child_field_name2features[
                        'parse_node_' + parse_node_field_name].append(
                            Token('UNKWOWN'))
            # Token node
            else:
                for parse_node_field_name, features in parse_node_field_name2features.items(
                ):
                    resolved_child_field_name2features[
                        'parse_node_' + parse_node_field_name].append(
                            features[node_id])

        resolved_feature_size = min(
            len(resolved_child_field_name2features['parse_node_{}'.format(
                parse_node_field_name)])
            for parse_node_field_name in parse_node_field_name2features)

        logger.warning(('resolved_feature_size', resolved_feature_size))
        logger.warning(('resolved_root_field_name2features',
                        resolved_root_field_name2features))

        logger.warning(('resolved_child_field_name2features',
                        resolved_child_field_name2features))
        if predict:
            resolved_edge_label_size = 0
        else:
            if framework != 'ucca':
                root_label = resolved_node.get('label')
                resolved_field_name2label['root_label'] = root_label
            else:
                root_label = resolved_node.get('propagate_label')
                resolved_field_name2label['root_label'] = root_label

            # One root child pair can have more than one edge
            resolved_field_name2label['edge_labels'] = []
            if action_num_pop >= 2:
                for position, resolved_edges in enumerate(resolved_edgess):
                    if position == resolved_node_position and framework != 'ucca':
                        continue
                    sorted_edge_labels = sorted(
                        [edge.get('label') for edge in resolved_edges])
                    edge_label = '_'.join(sorted_edge_labels)
                    resolved_field_name2label['edge_labels'].append(edge_label)

            resolved_edge_label_size = len(
                resolved_field_name2label['edge_labels'])

            if resolved_feature_size != resolved_edge_label_size:
                pass
            # logger.warning((resolved_feature_size,
            #                 resolved_edge_label_size))
            # pprint.pprint(
            #     (action_num_pop, resolved_node_position,
            #      resolved_edgess, resolved_child_field_name2features,
            #      resolved_root_field_name2features,
            #      resolved_field_name2label))

        return (
            resolved_child_field_name2features,
            resolved_root_field_name2features,
            resolved_field_name2metadata,
            resolved_field_name2label,
            resolved_feature_size,
            resolved_edge_label_size,
        )

    def process_mrp_graph(self, data_instance):
        mode = 'train'
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
        parse_nodes_size = len(parse_nodes)

        field_name2features = {}
        parse_node_field_name2features = self._generate_parse_node_field_name2features(
            parse_nodes, )
        # for field_name, field_type in self.parse_node_field_name2field_type.items(
        # ):
        #     START_PADDING = self.field_type2PADDINGS[field_type][0]
        #     parse_node_field_name2features[field_name].extend(
        #         [Token(START_PADDING)] * self.prev_parse_feature_window_size)

        # # Add end padding
        # for field_name, field_type in self.parse_node_field_name2field_type.items(
        # ):
        #     END_PADDING = self.field_type2PADDINGS[field_type][-1]
        #     parse_node_field_name2features[field_name].extend(
        #         [Token(END_PADDING)] * self.next_parse_feature_window_size)

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

            # Add parse_node state features
            parse_node_state_field_name2features = self._generate_parse_node_state_field_name2features(
                curr_node_id, parse_nodes_size, parse_node_field_name2features)

            token_node_field_name2features = self._generate_token_node_field_name2features(
                token_state, action_type_name_deque)
            logger.debug(('token_node_field_name2features',
                          token_node_field_name2features))
            logger.debug(('action', action_type))

            action_type_name_deque.append(Token(action_type_name))
            if len(action_type_name_deque) > self.prev_action_window_size:
                action_type_name_deque.popleft()

            action_num_pop = 1
            action_num_pop_mask = 0

            resolved_child_field_name2features = {}
            resolved_root_field_name2features = {}
            resolved_field_name2metadata = {}
            resolved_field_name2label = {}
            if action_type_name == 'RESOLVE':
                instance_type = 'resolve'
                (action_num_pop, resolved_node_position, resolved_node,
                 resolved_edgess) = action_params

                (
                    resolved_child_field_name2features,
                    resolved_root_field_name2features,
                    resolved_field_name2metadata,
                    resolved_field_name2label,
                    resolved_feature_size,
                    resolved_edge_label_size,
                ) = self._generate_resolved_field_name2features(
                    token_state, parse_nodes_size,
                    parse_node_field_name2features, framework, action_num_pop,
                    resolved_node_position, resolved_node, resolved_edgess)

                if resolved_feature_size != resolved_edge_label_size:
                    # logger.warning((resolved_feature_size,
                    #                 resolved_edge_label_size))
                    # pprint.pprint(
                    #     (action_num_pop, resolved_node_position,
                    #      resolved_edgess, resolved_child_field_name2features,
                    #      resolved_root_field_name2features,
                    #      resolved_field_name2label))
                    self.error_count += 1
                    logger.warning(('error count', self.error_count))
                else:
                    action_num_pop_mask = 1
                    yield self.text_to_instance(
                        instance_type,
                        framework,
                        mode,
                        curr_node_id,
                        parse_node_field_name2features,
                        parse_node_state_field_name2features,
                        token_node_field_name2features,
                        resolved_child_field_name2features,
                        resolved_root_field_name2features,
                        resolved_field_name2label,
                        resolved_field_name2metadata,
                        action_num_pop_mask,
                        action_type_name,
                        action_params,
                        action_num_pop - 1,
                        token_state,
                    )

            if framework == 'ucca' and action_num_pop == 1:
                continue

            instance_type = 'action'
            yield self.text_to_instance(
                instance_type,
                framework,
                mode,
                curr_node_id,
                parse_node_field_name2features,
                parse_node_state_field_name2features,
                token_node_field_name2features,
                resolved_child_field_name2features,
                resolved_root_field_name2features,
                resolved_field_name2label,
                resolved_field_name2metadata,
                action_num_pop_mask,
                action_type_name,
                action_params,
                action_num_pop - 1,
                token_state,
            )

    @overrides
    def text_to_instance(
            self,
            instance_type,
            framework,
            mode,
            curr_node_id,
            parse_node_field_name2features,
            parse_node_state_field_name2features,
            token_node_field_name2features,
            resolved_child_field_name2features={},
            resolved_root_field_name2features={},
            resolved_field_name2label={},
            resolved_field_name2metadata={},
            action_num_pop_mask=None,
            action_type_name=None,
            action_params=None,
            action_num_pop=None,
            token_state=None,
    ) -> Instance:
        field_name2field = {
            'instance_type': MetadataField(instance_type),
            'framework': MetadataField(framework),
            'mode': MetadataField(mode),
        }

        # Parse node fields
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

        for parse_node_state_field_name, features in parse_node_state_field_name2features.items(
        ):
            full_name = 'parse_node_{}s'.format(parse_node_state_field_name)
            field_type = self.parse_node_state_field_name2field_type[
                parse_node_state_field_name]
            if field_type in self.field_type2indexers:
                text_field = TextField(features,
                                       self.field_type2indexers[field_type])
            else:
                raise NotImplementedError
            field_name2field[full_name] = text_field

        # Token node fields
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

        # Action label fields
        if action_num_pop is not None and 'token_node_resolveds' in field_name2field:
            field_name2field['action_num_pop'] = IndexField(
                action_num_pop, field_name2field['token_node_resolveds'])
            field_name2field['action_num_pop_mask'] = IndexField(
                action_num_pop_mask, field_name2field['token_node_resolveds'])

        if action_type_name:
            field_name2field['action_type'] = LabelField(
                action_type_name, label_namespace='action_type_labels')

        # Metadata fields
        field_name2field['curr_node_id'] = MetadataField(curr_node_id)
        field_name2field['action_params'] = MetadataField(action_params)
        field_name2field['action_type_name'] = MetadataField(action_type_name)
        field_name2field['token_state'] = MetadataField(token_state)

        if instance_type == 'action':
            field_name2field['stack_len'] = MetadataField(len(token_state))
        else:
            field_name2field['stack_len'] = MetadataField(action_num_pop)

        token_node_ids = [token_node[0] for token_node in token_state]
        token_node_childs = [token_node[3] for token_node in token_state]
        field_name2field['token_node_ids'] = MetadataField(token_node_ids)
        field_name2field['token_node_childs'] = MetadataField(
            token_node_childs)

        # Action specific fields
        if instance_type == 'action':
            pass
        # Resolve specific fields
        elif instance_type == 'resolve':
            # Resolve root features
            for resolved_root_field_name, features in resolved_root_field_name2features.items(
            ):
                field_type = self.resolved_field_name2field_type[
                    resolved_root_field_name]
                full_name = 'resolved_root_{}'.format(resolved_root_field_name)
                if field_type in self.field_type2indexers:
                    text_field = TextField(
                        features, self.field_type2indexers[field_type])
                else:
                    raise NotImplementedError
                field_name2field[full_name] = text_field

            # Resolve child features
            for resolved_child_field_name, features in resolved_child_field_name2features.items(
            ):
                field_type = self.resolved_field_name2field_type[
                    resolved_child_field_name]
                full_name = 'resolved_child_{}s'.format(
                    resolved_child_field_name)
                if field_type in self.field_type2indexers:
                    text_field = TextField(
                        features, self.field_type2indexers[field_type])
                else:
                    raise NotImplementedError
                field_name2field[full_name] = text_field

            logger.warning(field_name2field.keys())
            logger.debug(field_name2field['resolved_child_parse_node_labels'])
            logger.debug(
                resolved_child_field_name2features['parse_node_label'])
            assert 'resolved_child_parse_node_labels' in field_name2field

            # Resolve labels
            if mode == 'train':
                for resolved_field_name, label in resolved_field_name2label.items(
                ):
                    field_type = self.resolved_field_name2field_type[
                        resolved_field_name]
                    full_name = 'resolved_label_{}'.format(resolved_field_name)
                    if field_type == 'LabelField':
                        field_name2field[full_name] = LabelField(
                            label, label_namespace=framework + '_' + full_name)
                    elif field_type == 'SequenceLabelField':
                        field_name2field[full_name] = SequenceLabelField(
                            label,
                            field_name2field[
                                'resolved_child_parse_node_labels'],
                            label_namespace=framework + '_' + full_name)
                    else:
                        raise NotImplementedError

            for resolved_field_name, metadata in resolved_field_name2metadata.items(
            ):
                field_name2field[resolved_field_name] = MetadataField(metadata)
        else:
            raise NotImplementedError

        return Instance(field_name2field)
