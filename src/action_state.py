import copy
import logging
import pprint
from collections import Counter, defaultdict, deque

logger = logging.getLogger('action_state')


def mrp_json2parser_states(mrp_json, framework, alignment={}):
    nodes = mrp_json.get('nodes', [])
    num_nodes = len(nodes)
    edges = mrp_json.get('edges', [])
    doc = mrp_json.get('input', '')

    node_id2node = {node.get('id', -1): node for node in nodes}
    edge_id2edge = {edge_id: edge for edge_id, edge in enumerate(edges)}

    parent_id2indegree = Counter()
    parent_id2child_id_set = defaultdict(set)
    child_id2parent_id_set = defaultdict(set)
    child_id2edge_id_set = defaultdict(set)
    parent_id2edge_id_set = defaultdict(set)
    node_id2edge_id_set = defaultdict(set)
    token_id2node_id_set = defaultdict(set)
    node_id2node_id2direction = [[0] * num_nodes for _ in range(num_nodes)]

    # parent_to_leaf_framework_set = {}
    # leaf_to_parent_framework_set = {'eds', 'dm', 'psd', 'ucca', 'amr'}

    # By observation,
    # edges of **PSD, UCCA and AMR** are from parent to leaf
    # edges of **EDS and DM** are from leaf to parent

    # We define in here:
    # source as parent and
    # target as child
    for edge_id, edge in enumerate(edges):
        is_remote_edge = any([
            'properties' in edge and 'remote' in edge['properties'],
            edge.get('label', '') == 'APPS.member',
            edge.get('label', '') == 'BV' and framework == 'eds',
        ])
        if is_remote_edge:
            continue
        # if framework in parent_to_leaf_framework_set:
        #     parent_id = edge.get('parent')
        #     child_id = edge.get('child')
        # else:
        #     parent_id = edge.get('child')
        #     child_id = edge.get('parent')
        parent_id = edge.get('source')
        child_id = edge.get('target')
        parent_id2child_id_set[parent_id].add(child_id)
        child_id2parent_id_set[child_id].add(parent_id)
        child_id2edge_id_set[child_id].add(edge_id)
        parent_id2edge_id_set[parent_id].add(edge_id)
        node_id2edge_id_set[child_id].add(edge_id)
        node_id2edge_id_set[parent_id].add(edge_id)
        parent_id2indegree[parent_id] += 1

    (token_nodes, abstract_node_id_set) = _resolve_dependencys(
        nodes,
        edges,
        parent_id2indegree,
        child_id2edge_id_set,
        framework,
        alignment,
    )
    token_node_id_set = {node.get('id', -1) for node in token_nodes}
    # parser_states = _generate_parser_states(doc, nodes, edges, node_id2node,
    #                                         token_nodes, abstract_node_id_set,
    #                                         child_id2edge_id_set)
    parser_states = _generate_parser_action_states(
        doc, nodes, edges, node_id2node, token_nodes, abstract_node_id_set,
        child_id2edge_id_set, parent_id2indegree, parent_id2child_id_set,
        child_id2parent_id_set, token_node_id_set)
    meta_data = (
        node_id2node,
        edge_id2edge,
        parent_id2child_id_set,
        child_id2parent_id_set,
        child_id2edge_id_set,
        parent_id2edge_id_set,
        parent_id2indegree,
        token_nodes,
        abstract_node_id_set,
    )
    return parser_states, meta_data


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
        token_nodes = []
        for node in nodes:
            node_id = node.get('id', -1)
            node_label = node.get('label', '')
            if 'anchors' in node:
                if node_label.startswith('_') or 'properties' in node:
                    token_nodes.append(node)
                else:
                    abstract_node_id_set.add(node_id)
            else:
                abstract_node_id_set.add(node_id)

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


CREATE = 0  # CREATE_NEW_GROUP
APPEND = 1  # APPEND_TO_CURRENT_GROUP
RESOLVE = 2  # RESOLVE_GROUP
COMBINE = 3  # COMBINE_GROUPS
PENDING = 4  # ADD_TO_PENDDING_STACK


def _generate_parser_action_states(
        doc, nodes, edges, node_id2node, token_nodes, abstract_node_id_set,
        child_id2edge_id_set, parent_id2indegree, parent_id2child_id_set,
        child_id2parent_id_set, token_node_id_set):
    """Generate the parse state at each token step"""

    # TODO(Sunny): Order of producing high level node
    #                i.e. when should we produce high level node
    seen_node_id_set = set()
    seen_edge_id_set = set()
    complete_node_id_set = set()
    parser_states = []
    parent_id2child_id_set = copy.deepcopy(parent_id2child_id_set)

    token_node_queue = deque(token_nodes)
    token_pos = 0

    complete_node_queue = deque()
    node_state = []
    token_stack = []
    pending_token_stack = []

    while token_node_queue or complete_node_queue:
        if complete_node_queue:
            node = complete_node_queue.popleft()
            is_complete_parent = True
        else:
            node = token_node_queue.popleft()
            is_complete_parent = False

        curr_node_id = node.get('id', -1)
        logger.info(('curr_node_id', curr_node_id))

        actions = []
        edge_state = []
        abstract_node_state = []
        complete_node_state = []

        is_curr_complete = not parent_id2child_id_set.get(curr_node_id, {})
        is_curr_ignored = all([
            curr_node_id not in parent_id2child_id_set,
            curr_node_id not in child_id2parent_id_set,
        ])
        # is_curr_decoration_node = curr_node_id in decoration_node_id_set
        is_curr_decoration_node = False
        is_curr_no_child = not parent_id2indegree[curr_node_id]

        # Actions
        logger.info((curr_node_id, node_state, is_curr_no_child,
                     is_curr_complete))
        if is_curr_decoration_node:
            # Do nothing if not is remote node
            pass
        elif is_curr_ignored:
            # Do nothing if node is ignored
            # TODO(Sunny): Handle remote edges
            pass
        elif is_curr_no_child:
            actions.append((APPEND, None))
            node_state.append((curr_node_id, curr_node_id, None))
        elif is_curr_complete:
            actions.append((RESOLVE, parent_id2indegree[curr_node_id]))
            new_state = [(curr_node_id, curr_node_id, None)]
            for _ in range(parent_id2indegree[curr_node_id]):
                root, last, childs = node_state.pop()
                new_state.append((root, last, childs))
            node_state.append((curr_node_id, curr_node_id, new_state))
        elif curr_node_id not in seen_node_id_set:
            actions.append((PENDING, None))
        logger.info((curr_node_id, curr_node_id, node_state))

        seen_node_id_set.add(curr_node_id)

        if is_curr_complete and curr_node_id not in complete_node_id_set:
            complete_node_id_set.add(curr_node_id)
            complete_node_state.append(curr_node_id)

        for edge_id in child_id2edge_id_set[curr_node_id]:
            edge = edges[edge_id]
            is_remote_edge = 'properties' in edge and 'remote' in edge['properties']

            parent_id = edge.get('source', -1)
            child_id = curr_node_id
            if is_curr_complete and child_id in parent_id2child_id_set[parent_id]:
                parent_id2child_id_set[parent_id].remove(child_id)
            logger.info((child_id, edge_id, parent_id,
                         parent_id2child_id_set[parent_id]))
            if not parent_id2child_id_set[parent_id]:
                logger.info(('complete', parent_id))
                is_abstract = parent_id in abstract_node_id_set
                is_seen = parent_id in seen_node_id_set
                if is_abstract or is_seen:
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

        # Simulation action
        for action in actions:
            action_type, args = action
            if action_type == APPEND:
                token_stack.append(curr_node_id)
            elif action_type == RESOLVE:
                num_pop = args
                while num_pop > 0 and token_stack:
                    token_stack.pop()
                    num_pop -= 1
                token_stack.append(curr_node_id)
            elif action_type == PENDING:
                pending_token_stack.append(curr_node_id)

        parser_states.append((
            curr_node_id,
            actions,
            edge_state,
            abstract_node_state,
            complete_node_state,
            copy.deepcopy(node_state),
            copy.deepcopy(token_stack),
            copy.deepcopy(pending_token_stack),
        ))
    return parser_states


def _generate_parser_states(doc, nodes, edges, node_id2node, token_nodes,
                            abstract_node_id_set, child_id2edge_id_set):
    """Generate the parse state at each token step"""

    # TODO(Sunny): Order of producing high level node
    #                i.e. when should we produce high level node
    seen_node_id_set = set()
    seen_edge_id_set = set()
    parser_states = []

    token_node_queue = deque(token_nodes)
    token_pos = 0

    while token_node_queue:
        actions = []
        node = token_node_queue.popleft()
        edge_state = []
        abstract_node_state = []
        node_id = node.get('id', -1)
        if node_id not in seen_node_id_set:
            actions.append(CNNWT)
        seen_node_id_set.add(node_id)

        for edge_id in child_id2edge_id_set[node_id]:
            edge = edges[edge_id]

            parent_id = edge.get('source', -1)
            child_id = edge.get('target', -1)

            is_remote_edge = 'properties' in edge and 'remote' in edge['properties']

            # Handle abstract nodes if not remote edge
            # TODO(Sunny): This cannot handle the case of a node having
            #                All childs as abstract_node
            if not is_remote_edge and node_id not in abstract_node_id_set:
                for neig_id in [parent_id, child_id]:
                    is_abstract = neig_id in abstract_node_id_set
                    seen = neig_id in seen_node_id_set
                    if is_abstract and not seen:
                        abstract_node_state.append(neig_id)
                        seen_node_id_set.add(neig_id)
                        token_node_queue.appendleft(node_id2node[neig_id])

            edge_not_seen = edge_id not in seen_edge_id_set
            edge_nodes_seen = all([
                parent_id in seen_node_id_set,
                child_id in seen_node_id_set,
            ])

            # add edge if edge not seen and both ends seen
            if edge_not_seen and edge_nodes_seen:
                edge_state.append(edge_id)
                seen_edge_id_set.add(edge_id)
        parser_states.append((node_id, actions, edge_state,
                              abstract_node_state))
    return parser_states
