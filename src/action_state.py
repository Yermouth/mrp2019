import copy
import logging
import pprint
import re
from collections import Counter, defaultdict, deque

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


def mrp_json2parser_states(
        mrp_json,
        mrp_doc='',
        tokenized_parse_nodes=[],
        alignment={},
):
    framework = mrp_json.get('framework', '')
    nodes = mrp_json.get('nodes', [])
    num_nodes = len(nodes)
    edges = mrp_json.get('edges', [])
    top_node_ids = mrp_json.get('tops', [])
    doc = mrp_doc or mrp_json.get('input', '')
    if not doc:
        raise ValueError

    node_id2node = {node.get('id', -1): node for node in nodes}

    parent_id2indegree = Counter()
    parent_id2child_id_set = defaultdict(set)
    child_id2parent_id_set = defaultdict(set)
    parent_id2edge_id_set = defaultdict(set)
    child_id2edge_id_set = defaultdict(set)
    parent_child_id2edge_id_set = defaultdict(set)
    node_id2edge_id_set = defaultdict(set)
    node_id2neig_id_set = defaultdict(set)
    node_id2node_id2direction = [[0] * num_nodes for _ in range(num_nodes)]
    top_oriented_edge_id_set = set()

    # parent_to_leaf_framework_set = {}
    # leaf_to_parent_framework_set = {'eds', 'dm', 'psd', 'ucca', 'amr'}

    # By observation,
    # edges of **PSD, UCCA and AMR** are from parent to leaf
    # edges of **EDS and DM** are from leaf to parent

    # We define in here:
    # source as parent and
    # target as child

    for edge_id, edge in enumerate(edges):
        edges[edge_id]['id'] = edge_id
        is_remote_edge = any([
            'properties' in edge and 'remote' in edge['properties'],
            # edge.get('label', '') == 'APPS.member',
            edge.get('label', '') == 'BV' and framework == 'eds',
        ])

        if is_remote_edge:
            logger.debug(('remote 1', edge_id))
            continue

        parent_id = edge.get('source')
        child_id = edge.get('target')
        node_id2edge_id_set[child_id].add(edge_id)
        node_id2edge_id_set[parent_id].add(edge_id)
        top_oriented_edge_id_set.add(edge_id)

    edge_id2edge = {edge_id: edge for edge_id, edge in enumerate(edges)}

    top_oriented_edges = _top_oriented_edges(edges, edge_id2edge,
                                             top_oriented_edge_id_set,
                                             node_id2edge_id_set, top_node_ids)

    for edge in top_oriented_edges:
        edge_id = edge.get('id', -1)
        is_remote_edge = any([
            'properties' in edge and 'remote' in edge['properties'],
            edge.get('label', '') == 'APPS.member',
            edge.get('label', '').startswith('_') and framework == 'dm',
            edge.get('label', '') == 'BV' and framework == 'eds',
        ])

        if is_remote_edge:
            logger.debug(('remote 2', edge_id))
            continue
        # if framework in parent_to_leaf_framework_set:
        #     parent_id = edge.get('parent')
        #     child_id = edge.get('child')
        # else:
        #     parent_id = edge.get('child')
        #     child_id = edge.get('parent')

        # parent_id = edge.get('source')
        # child_id = edge.get('target')

        parent_id = edge.get('parent')
        child_id = edge.get('child')

        parent_id2child_id_set[parent_id].add(child_id)
        child_id2parent_id_set[child_id].add(parent_id)
        child_id2edge_id_set[child_id].add(edge_id)
        parent_id2edge_id_set[parent_id].add(edge_id)
        parent_child_id2edge_id_set[(parent_id, child_id)].add(edge_id)

    for parent_id, child_id_set in parent_id2child_id_set.items():
        parent_id2indegree[parent_id] = len(child_id_set)

    (token_nodes, abstract_node_id_set) = _resolve_dependencys(
        nodes,
        # edges,
        top_oriented_edges,
        parent_id2indegree,
        child_id2edge_id_set,
        framework,
        alignment,
    )

    if tokenized_parse_nodes:
        token_pos = 0
        parse_nodes_anchors = []
        char_pos2tokenized_node_id = []
        for node_id, node in enumerate(tokenized_parse_nodes):
            label = node.get('label')
            label_size = len(label)
            while doc[token_pos] == ' ':
                token_pos += 1
                char_pos2tokenized_node_id.append(node_id)
            parse_nodes_anchors.append((token_pos, token_pos + label_size))
            char_pos2tokenized_node_id.extend([node_id] * (label_size))
            token_pos += label_size
    else:
        raise NotImplementedError

    # prev_anchor_from = 0
    # anchor2token_id = {}
    # anchors = []
    # for token_id, token in enumerate(sentence_spliter(doc)):
    #     anchor_to = prev_anchor_from + len(token)
    #     anchors.append((prev_anchor_from, anchor_to))
    #     anchor2token_id[prev_anchor_from] = token_id
    #     anchor2token_id[anchor_to] = token_id
    #     prev_anchor_from = anchor_to + 1
    # logger.debug(pprint.pformat(('anchor2token_id', anchor2token_id)))

    parser_states, actions = _generate_parser_action_states(
        framework,
        doc,
        nodes,
        # edges,
        top_oriented_edges,
        node_id2node,
        token_nodes,
        abstract_node_id_set,
        child_id2edge_id_set,
        parent_id2indegree,
        parent_id2child_id_set,
        child_id2parent_id_set,
        parent_child_id2edge_id_set,
        parse_nodes_anchors,
        char_pos2tokenized_node_id,
        tokenized_parse_nodes,
    )
    # parser_states, actions = [], []

    meta_data = (
        doc,
        nodes,
        node_id2node,
        edge_id2edge,
        top_oriented_edges,
        token_nodes,
        # abstract_node_id_set,
        parent_id2indegree,
        # parent_id2child_id_set,
        # child_id2parent_id_set,
        # child_id2edge_id_set,
        # parent_id2edge_id_set,
        # parent_child_id2edge_id_set,
        parse_nodes_anchors,
        char_pos2tokenized_node_id,
        actions,
    )
    return parser_states, meta_data


def _top_oriented_edges(edges, edge_id2edge, top_oriented_edge_id_set,
                        node_id2edge_id_set, top_node_ids):
    seen_node_id_set = set()
    seen_edge_id_set = set()

    if not top_node_ids:
        logger.warning('No top node provided')
        return edges

    elif len(top_node_ids) == 1:
        top_node_id = top_node_ids[0]

    # Can only happen in **PSD**, where root is a punctuation
    # If more then one top node find their common ancestor instead
    else:
        top_node_id = _common_ancestor(top_node_ids)

    top_oriented_edges = copy.deepcopy(edges)
    curr_node_id_layer = set([top_node_id])
    while curr_node_id_layer:
        logger.debug(curr_node_id_layer)
        next_node_id_layer = set()
        for curr_node_id in curr_node_id_layer:
            if curr_node_id not in seen_node_id_set:
                seen_node_id_set.add(curr_node_id)
                for edge_id in node_id2edge_id_set[curr_node_id]:
                    if edge_id in seen_edge_id_set:
                        continue
                    if edge_id in top_oriented_edge_id_set:
                        seen_edge_id_set.add(edge_id)
                        edge = edge_id2edge[edge_id]
                        node_id_1 = edge.get('source')
                        node_id_2 = edge.get('target')
                        if node_id_1 == curr_node_id:
                            parent_id = node_id_1
                            child_id = node_id_2
                        else:
                            parent_id = node_id_2
                            child_id = node_id_1

                        if child_id not in seen_node_id_set:
                            top_oriented_edges[edge_id]['parent'] = parent_id
                            top_oriented_edges[edge_id]['child'] = child_id
                            next_node_id_layer.add(child_id)
        curr_node_id_layer = next_node_id_layer
    return top_oriented_edges


def _common_ancestor(top_nodes):
    return top_nodes[0]


def _resolve_dependencys(nodes,
                         edges,
                         parent_id2indegree,
                         child_id2edge_id_set,
                         framework,
                         alignment={}):
    """Split nodes into two sets: token and abstract

    Anchored nodes: node that anchor on one and only one token
    Abstract nodes: node that anchor on more than one token or zero token
    """

    # For **eds**, nodes connected only with one 'BV' edge
    # is considered abstract
    if framework == 'eds':
        # Split nodes to abstract and token set
        abstract_node_id_set = set()
        # token_nodes = []
        # for node in nodes:
        #     node_id = node.get('id', -1)
        #     node_label = node.get('label', '')
        #     if 'anchors' in node:
        #         if node_label.startswith(
        #                 '_') or 'properties' in node or node_label == 'pron':
        #             token_nodes.append(node)
        #         else:
        #             abstract_node_id_set.add(node_id)
        #     else:
        #         abstract_node_id_set.add(node_id)
        token_nodes = nodes

        assert all(len(node.get('anchors', [])) == 1 for node in token_nodes)
        assert all('from' in node.get('anchors')[0] for node in token_nodes)
        assert all('to' in node.get('anchors')[0] for node in token_nodes)

        # Sort nodes according to anchor range
        token_nodes.sort(
            key=
            lambda node: (node['anchors'][0]['to'], -node['anchors'][0]['from'])
        )
        # pprint.pprint(token_nodes)

    # For **ucca**, nodes without anchors
    # is considered abstract
    elif framework == 'ucca':
        abstract_node_id_set = set()
        token_nodes = []

        for node in nodes:
            if 'anchors' in node:
                token_nodes.append(node)
            else:
                abstract_node_id_set.add(node.get('id', -1))
        assert all(len(node.get('anchors', [])) >= 1 for node in token_nodes)
        assert all(
            all('from' in anchor and 'to' in anchor
                for anchor in node.get('anchors')) for node in token_nodes)

        # Sort nodes according to anchor range
        token_nodes.sort(
            key=
            lambda node: (max(anchor['to'] for anchor in node['anchors']), -min(anchor['from'] for anchor in node['anchors']))
        )

    # For **amr**, nodes without token anchor in jamr alignment
    # is considered abstract

    # TODO(Sunny): The nodes in jamr alignment may have no anchor
    #                i.e. match not complete
    elif framework == 'amr':
        assert alignment
        abstract_node_id_set = set()
        token_nodes = []

        node_id2token_poss = {}
        for node in alignment.get('nodes', []):
            node_id = node.get('id', -1)
            token_poss = node.get('label', [])
            node_id2token_poss[node_id] = token_poss

        for node in nodes:
            node_id = node.get('id', -1)
            if node_id in node_id2token_poss:
                token_nodes.append(node)
            else:
                abstract_node_id_set.add(node_id)

        assert all(
            len(node_id2token_poss.get(node.get('id', -1), [])) >= 1
            for node in token_nodes)

        token_nodes.sort(key=lambda node: (
            max(node_id2token_poss[node.get('id', -1)]),
            -min(node_id2token_poss[node.get('id', -1)]),
        ))
    else:
        token_nodes = nodes
        abstract_node_id_set = set()
    return (token_nodes, abstract_node_id_set)


ERROR = -1
APPEND = 0  # APPEND_TO_CURRENT_GROUP
RESOLVE = 1  # RESOLVE_GROUP
IGNORE = 2  # IGNORE_CURRENT_TOKEN


def sentence_spliter(doc):
    for token in re.findall(r"[\w+'.]+\s*|[,!?;]", doc):
        yield token


def _generate_parser_action_states(
        framework,
        doc,
        nodes,
        edges,
        node_id2node,
        token_nodes,
        abstract_node_id_set,
        child_id2edge_id_set,
        parent_id2indegree,
        parent_id2child_id_set,
        child_id2parent_id_set,
        parent_child_id2edge_id_set,
        parse_nodes_anchors,
        char_pos2tokenized_node_id,
        tokenized_parse_nodes,
):
    """Generate the parse state at each token step"""

    # TODO(Sunny): Order of producing high level node
    #                i.e. when should we produce high level node
    seen_node_id_set = set()
    visited_node_id_set = set()
    resolved_node_id_set = set()
    seen_edge_id_set = set()
    complete_node_id_set = set()
    parser_states = []
    parent_id2child_id_set = copy.deepcopy(parent_id2child_id_set)

    token_node_queue = deque(token_nodes)
    prev_anchor_from = 0
    prev_tokenized_node_id = 0

    complete_node_queue = deque()
    node_state = []
    token_stack = []
    parse_token_stack = copy.deepcopy(tokenized_parse_nodes)
    parse_token_stack.reverse()
    actions = []

    count = 0
    error = 0

    while count <= 1000 and (token_node_queue or complete_node_queue):
        count += 1
        action_state = []
        edge_state = []
        abstract_node_state = []
        complete_node_state = []

        if complete_node_queue:
            node = complete_node_queue.popleft()
            is_complete_parent = True
        else:
            node = token_node_queue.popleft()
            is_complete_parent = False
            if 'anchors' in node:
                anchors = node.get('anchors')
                anchor_from = min(anchor.get('from', -1) for anchor in anchors)
                anchor_to = max(anchor.get('to', -1) for anchor in anchors)
                curr_tokenized_node_id = char_pos2tokenized_node_id[
                    anchor_from]
                logger.debug((
                    'prev anchors',
                    prev_tokenized_node_id,
                ))
                for tokenized_node_id in range(prev_tokenized_node_id,
                                               curr_tokenized_node_id):
                    logger.debug('ignore {}'.format(tokenized_node_id))
                    action_state.append((IGNORE, None))

                prev_tokenized_node_id = curr_tokenized_node_id + 1
                logger.debug((
                    'anchors',
                    anchor_from,
                    anchor_to,
                    curr_tokenized_node_id,
                    prev_tokenized_node_id,
                ))

        curr_node_id = node.get('id', -1)
        curr_node_label = node.get('label', '')
        num_childs = parent_id2indegree[curr_node_id]
        logger.debug(('curr_node_id', curr_node_id))

        is_curr_complete = not parent_id2child_id_set.get(curr_node_id, {})
        is_curr_ignored = all([
            curr_node_id not in parent_id2child_id_set,
            curr_node_id not in child_id2parent_id_set,
        ])
        # is_curr_decoration_node = curr_node_id in decoration_node_id_set
        is_curr_decoration_node = False
        is_curr_no_child = not parent_id2indegree[curr_node_id]
        is_curr_not_seen = not curr_node_id in seen_node_id_set
        is_curr_not_visited = not curr_node_id in visited_node_id_set
        is_curr_not_resolved = not curr_node_id in resolved_node_id_set
        is_curr_not_abstract = not curr_node_id in abstract_node_id_set

        # if is_curr_complete and curr_node_id not in complete_node_id_set:
        #     complete_node_id_set.add(curr_node_id)
        #     complete_node_state.append(curr_node_id)

        # Actions
        logger.debug(
            pprint.pformat(
                (curr_node_id, node_state, is_curr_no_child, is_curr_complete,
                 is_curr_not_seen, is_curr_not_resolved)))
        if is_curr_decoration_node:
            # Do nothing if not is remote node
            pass
        elif is_curr_ignored:
            # Do nothing if node is ignored
            # TODO(Sunny): Handle remote edges
            pass
        else:
            if is_curr_not_visited and is_curr_not_abstract:
                action_state.append((APPEND, None))
                node_state.append((curr_node_id, curr_node_id, None))

            if is_curr_complete and is_curr_not_resolved:
                resolved_node_id_set.add(curr_node_id)
                resolved_node = node_id2node[curr_node_id]
                resolved_edgess = []

                num_pop = num_childs
                if is_curr_not_abstract:
                    num_pop += 1

                new_state = []
                for _ in range(num_pop):
                    if node_state:
                        root_id, last, childs = node_state.pop()
                    else:
                        error = 1
                        logger.warning('pop empty node_state')
                        return [], []
                    new_state.append((root_id, last, childs))
                    resolved_edges = []
                    for edge_id in parent_child_id2edge_id_set[(curr_node_id,
                                                                root_id)]:
                        edge = edges[edge_id]
                        resolved_edges.append(edge)
                    resolved_edgess.append(resolved_edges)
                new_state.reverse()
                resolved_edgess.reverse()

                node_state.append((curr_node_id, curr_node_id, new_state))
                action_state.append((RESOLVE, (num_pop, resolved_node,
                                               resolved_edgess)))

        # elif curr_node_id not in seen_node_id_set:
        #     action_state.append((PENDING, None))
        logger.debug(pprint.pformat((curr_node_id, curr_node_id, node_state)))
        logger.debug(action_state)

        visited_node_id_set.add(curr_node_id)
        seen_node_id_set.add(curr_node_id)
        for edge_id in child_id2edge_id_set[curr_node_id]:
            edge = edges[edge_id]
            is_remote_edge = 'properties' in edge and 'remote' in edge['properties']

            parent_id = edge.get('parent', -1)
            child_id = curr_node_id
            if is_curr_complete and child_id in parent_id2child_id_set[parent_id]:
                parent_id2child_id_set[parent_id].remove(child_id)
            logger.debug((child_id, edge_id, parent_id,
                          parent_id2child_id_set[parent_id]))
            if not parent_id2child_id_set[parent_id]:
                is_abstract = parent_id in abstract_node_id_set
                # is_not_defined_complete = not parent_id in complete_node_id_set
                is_parent_not_seen = not parent_id in seen_node_id_set
                is_parent_not_complete = not parent_id in complete_node_id_set
                is_parent_not_resolved = not parent_id in resolved_node_id_set

                # if (is_abstract
                #         and is_not_defined_complete) or is_parent_not_seen:
                if is_abstract or is_parent_not_complete:
                    complete_node_id_set.add(parent_id)
                    seen_node_id_set.add(parent_id)
                    complete_node_state.append(parent_id)
                    complete_node_queue.append(node_id2node[parent_id])

            # Handle abstract nodes if not remote edge
            # TODO(Sunny): This cannot handle the case of a node having
            #                All childs as abstract_node
            # if not is_remote_edge and curr_node_id not in abstract_node_id_set:
            #     for neig_id in [parent_id, child_id]:
            #         is_abstract = neig_id in abstract_node_id_set
            #         seen = neig_id in seen_node_id_set
            #         if is_abstract and not seen:
            #             abstract_node_state.append(neig_id)
            #             seen_node_id_set.add(neig_id)
            #             token_node_queue.appendleft(node_id2node[neig_id])

            edge_not_seen = edge_id not in seen_edge_id_set
            edge_nodes_seen = all([
                parent_id in seen_node_id_set,
                child_id in seen_node_id_set,
            ])

            # add edge if edge not seen and both ends seen
            if edge_not_seen and edge_nodes_seen:
                edge_state.append(edge_id)
                seen_edge_id_set.add(edge_id)

        # Simulate actions
        for action in action_state:
            action_type, params = action
            if action_type == APPEND:
                token = parse_token_stack.pop()
                token_stack.append((curr_node_id, curr_node_label,
                                    token.get('label')))
            elif action_type == RESOLVE:
                (num_pop, resolved_node, resolved_edges) = params
                new_token_group = []
                while num_pop > 0 and token_stack:
                    new_token_group.append(token_stack.pop())
                    num_pop -= 1
                new_token_group.reverse()
                token_stack.append((curr_node_id, curr_node_label,
                                    new_token_group))
            elif action_type == IGNORE:
                token = parse_token_stack.pop()

        actions.extend(action_state)

        logger.debug(pprint.pformat(('token stack', token_stack)))
        logger.debug(('visited states', seen_node_id_set, visited_node_id_set,
                      complete_node_id_set, resolved_node_id_set))
        parser_states.append((
            curr_node_id,
            action_state,
            edge_state,
            abstract_node_state,
            complete_node_state,
            copy.deepcopy(node_state),
            copy.deepcopy(token_stack),
        ))

    if error:
        return [], []
    return parser_states, actions
