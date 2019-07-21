import copy
import logging
import pprint
from collections import deque

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (IndexField, LabelField, MetadataField,
                                  SequenceLabelField, TextField)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from mrp_library.dataset_readers.mrp_jsons_actions import (APPEND,
                                                           EMPTY_TOKEN_STACK,
                                                           END_PADDING_POS,
                                                           END_PADDING_WORD,
                                                           ERROR, IGNORE,
                                                           INITIAL_ACTION,
                                                           NO_MORE_PARSE_NODE,
                                                           RESOLVE,
                                                           START_PADDING_POS,
                                                           START_PADDING_WORD,
                                                           UNKNOWN_POS,
                                                           UNKNOWN_WORD)
from overrides import overrides

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)


#@Predictor.register('mrp-predictor')
class MrpPredictor(Predictor):
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        cid = json_dict['cid']
        doc = json_dict['doc']
        framework = json_dict['framework']
        mrp_json = json_dict['mrp_json']
        parse_json = json_dict['parse_json']
        tokenized_parse_nodes = parse_json['nodes']
        companion_parser_states = json_dict['companion_parser_states']
        companion_meta_data = json_dict['companion_meta_data']

        # Generate parse node features
        parse_nodes = parse_json.get('nodes', [])
        parse_nodes_size = len(parse_nodes)
        parse_node_field_name2features = self._dataset_reader._generate_parse_node_field_name2features(
            parse_nodes)

        action_type_name_deque = deque([Token(INITIAL_ACTION)])

        # Build graph (similar to _simulate_actions in action_state.py)
        # initialization
        token_states = []
        curr_node_ids = []
        parse_token_stack = copy.deepcopy(tokenized_parse_nodes)
        parse_token_stack.reverse()
        token_stack = []
        curr_node_id = 0
        curr_edge_id = 0
        abstract_node_id = len(parse_token_stack)
        num_node_unresolved = 0
        parse_node_state_field_name2features = self._dataset_reader._generate_parse_node_state_field_name2features(
            curr_node_id, parse_nodes_size, parse_node_field_name2features)

        curr_state = {
            'mode':
            'predict',
            'framework':
            framework,
            'curr_node_id':
            curr_node_id,
            'parse_node_field_name2features':
            parse_node_field_name2features,
            'parse_node_state_field_name2features':
            parse_node_state_field_name2features,
            'token_node_field_name2features': {},
            'token_state':
            token_stack,
        }
        node_id2mrp_node = {}
        edge_id2mrp_edge = {}
        next_action = APPEND

        # Build graph until all nodes resolved and no parse token left
        while num_node_unresolved > 1 or len(parse_token_stack) > 0 or len(
                token_stack) > 1:
            logger.warning(('num_node_unresolved', num_node_unresolved))
            logger.warning(('len(parse_token_stack)', len(parse_token_stack)))
            logger.warning(('token_stack', len(token_stack)))
            # Generate current state features
            if next_action is None or next_action == RESOLVE:
                curr_state['instance_type'] = 'action'
                action_instance = self._json_to_instance(curr_state)
                action_output_dict = self.predict_instance(action_instance)
                action_type_pred = action_output_dict.get('action_type_preds')
                action_type = action_type_pred

            if next_action is not None:
                action_type = next_action
                next_action = None

            if action_type == APPEND:
                token = parse_token_stack.pop()
                token_stack.append((curr_node_id, False, token.get('label'),
                                    []))
                curr_node_id += 1
                num_node_unresolved += 1
                parse_node_state_field_name2features = self._dataset_reader._generate_parse_node_state_field_name2features(
                    curr_node_id, parse_nodes_size,
                    parse_node_field_name2features)

            elif action_type == RESOLVE:
                curr_state['instance_type'] = 'resolve'
                masked_action_num_pop_pred = action_output_dict.get(
                    'masked_action_num_pop_preds')

                not_resolved_exist = False
                resolved_end_found = False
                action_num_pop = 0
                while all([
                        action_num_pop < len(token_stack),
                        not resolved_end_found,
                    (not not_resolved_exist
                     or action_num_pop < masked_action_num_pop_pred),
                ]):
                    token_node = token_stack[::-1][action_num_pop]
                    (resolved_node_id, is_resolved, resolved_node_label,
                     resolved_node_childs) = token_node
                    logger.debug(('token_node', token_node))
                    if not is_resolved:
                        if not_resolved_exist:
                            resolved_end_found = True
                        else:
                            not_resolved_exist = True
                    if not resolved_end_found:
                        action_num_pop += 1
                    logger.debug(('token_node', action_num_pop, is_resolved,
                                  resolved_end_found))

                num_node_unresolved -= (action_num_pop - 1)
                logger.debug(('action_num_pop', action_num_pop))

                # Generate resolved features
                (
                    resolved_child_field_name2features,
                    resolved_root_field_name2features,
                    resolved_field_name2metadata,
                    resolved_field_name2label,
                    resolved_feature_size,
                    resolved_edge_label_size,
                ) = self._dataset_reader._generate_resolved_field_name2features(
                    token_stack,
                    parse_nodes_size,
                    parse_node_field_name2features,
                    framework,
                    action_num_pop,
                    predict=True,
                )

                curr_state[
                    'resolved_child_field_name2features'] = resolved_child_field_name2features
                curr_state[
                    'resolved_root_field_name2features'] = resolved_root_field_name2features
                curr_state[
                    'resolved_field_name2metadata'] = resolved_field_name2metadata
                curr_state[
                    'resolved_field_name2label'] = resolved_field_name2label
                curr_state['action_num_pop'] = action_num_pop
                curr_state['action_num_pop_mask'] = 1
                # Resolve node and edges details
                resolved_instance = self._json_to_instance(curr_state)
                resolved_output_dict = self.predict_instance(resolved_instance)

                if action_num_pop == 1:
                    if token_stack and not token_stack[-1][1]:
                        token_node = token_stack.pop()
                        (resolved_node_id, is_resolved, resolved_node_label,
                         resolved_node_childs) = token_node
                        token_stack.append(
                            (resolved_node_id, True, resolved_node_label,
                             resolved_node_childs))

                        resolved_node = {
                            'id': resolved_node_id,
                            'label': resolved_node_label,
                        }
                        # Add to mrp nodes
                        node_id2mrp_node[resolved_node_id] = resolved_node
                    else:
                        logger.info('Warning: resolving complete node')
                        logger.info('switching to APPEND')
                else:
                    # Pop children
                    resolved_node_childs = []
                    num_pop = action_num_pop
                    resolved_node_id = token_stack[-1][0]
                    resolved_token_node = token_stack[-1]

                    while num_pop > 0 and token_stack:
                        token_node = token_stack.pop()
                        (node_id, is_resolved, node_label,
                         node_childs) = token_node
                        if not is_resolved:
                            resolved_node_id = node_id
                            resolved_token_node = token_node
                        resolved_node_childs.append(token_node)
                        num_pop -= 1
                    resolved_node_childs.reverse()

                    resolved_node = {'id': resolved_node_id}
                    node_id2mrp_node[resolved_node_id] = resolved_token_node

                    resolved_edges = resolved_output_dict.get(
                        'resolved_edges', [])
                    for edge in resolved_edges:
                        edge_id2mrp_edge[curr_edge_id] = edge
                        curr_edge_id += 1

                    # Append complete node to token_stack
                    token_stack.append(
                        (resolved_node_id, True, resolved_node_label,
                         resolved_node_childs))

            elif action_type == IGNORE:
                token = parse_token_stack.pop()
                curr_node_id += 1
                parse_node_state_field_name2features = self._dataset_reader._generate_parse_node_state_field_name2features(
                    curr_node_id, parse_nodes_size,
                    parse_node_field_name2features)

            else:
                raise NotImplementedError

            action_type_name = self._dataset_reader.action_type2action_type_name[
                action_type]
            action_type_name_deque.append(Token(action_type_name))
            if len(action_type_name_deque
                   ) > self._dataset_reader.prev_action_window_size:
                action_type_name_deque.popleft()

            token_node_field_name2features = self._dataset_reader._generate_token_node_field_name2features(
                token_stack, action_type_name_deque)

            curr_state = {
                'mode':
                'predict',
                'curr_node_id':
                curr_node_id,
                'framework':
                framework,
                'token_state':
                token_stack,
                'parse_node_field_name2features':
                parse_node_field_name2features,
                'parse_node_state_field_name2features':
                parse_node_state_field_name2features,
                'token_node_field_name2features':
                token_node_field_name2features,
                'resolved_child_field_name2features': {},
                'resolved_root_field_name2features': {},
            }

            if len(parse_token_stack) == 0:
                next_action = RESOLVE
            logger.info(('curr_state', curr_state))
            logger.info(pprint.pformat(('token_stack', token_stack)))
        # # label_dict will be like {0: "ACL", 1: "AI", ...}
        # label_dict = self._model.vocab.get_index_to_token_vocabulary(
        #     'labels')
        # # Convert it to list ["ACL", "AI", ...]
        # all_labels = [label_dict[i] for i in range(len(label_dict))]

        # Convert final state to mrp output format
        mrp_output_dict = {
            'nodes': node_id2mrp_node,
            'edges': edge_id2mrp_edge,
        }
        return mrp_output_dict

    @overrides
    def _json_to_instance(self, curr_state: JsonDict) -> Instance:
        logger.info(('_json_to_instance: curr_state', curr_state))
        return self._dataset_reader.text_to_instance(**curr_state)

    # def text_to_instance(
    #         self,
    #         instance_type,
    #         framework,
    #         parse_node_field_name2features,
    #         parse_node_state_field_name2features,
    #         token_node_field_name2features,
    #         resolved_child_field_name2features,
    #         resolved_root_field_name2features,
    #         resolved_field_name2label,
    #         resolved_field_name2metadata,
    #         curr_node_id,
    #         action_type_name,
    #         action_params,
    #         action_num_pop,
    #         action_num_pop_mask,
    #         token_stack,
    # ) -> Instance:
