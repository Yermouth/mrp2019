from collections import Counter, defaultdict, deque


def mrp_json2parser_states(mrp_json, framework, alignment={}):
    nodes = mrp_json['nodes']
    edges = mrp_json['edges']

    node_id2node = {node.get('id', -1): node for node in nodes}
    edge_id2edge = {edge_id: edge for edge_id, edge in enumerate(edges)}

    node_id2indegree = Counter()
    node_id2neig_id_set = defaultdict(set)
    node_id2edge_id_set = defaultdict(set)

    parent_to_leaf_framework_set = {'psd', 'ucca', 'amr'}
    leaf_to_parent_framework_set = {'eds', 'dm'}

    # By observation,
    # edges of **PSD, UCCA and AMR** are from parent to leaf
    # edges of **EDS and DM** are from leaf to parent
    for edge_id, edge in enumerate(edges):
        #     if framework in parent_to_leaf_framework_set:
        #         source = edge.get('source')
        #         target = edge.get('target')
        #     else:
        #         source = edge.get('target')
        #         target = edge.get('source')
        source_id = edge.get('source')
        target_id = edge.get('target')
        node_id2neig_id_set[source_id].add(target_id)
        node_id2edge_id_set[source_id].add(edge_id)
        node_id2edge_id_set[target_id].add(edge_id)
        node_id2indegree[target_id] += 1

    (anchored_nodes, abstract_node_id_set) = _split_anchored_abstract_nodes(
        nodes,
        edges,
        node_id2indegree,
        node_id2edge_id_set,
        framework,
        alignment,
    )
    parser_states = _generate_parser_states(
        nodes, edges, node_id2node, anchored_nodes, abstract_node_id_set,
        node_id2edge_id_set)
    meta_data = (node_id2node, edge_id2edge)
    return parser_states, meta_data


def _split_anchored_abstract_nodes(nodes,
                                   edges,
                                   node_id2indegree,
                                   node_id2edge_id_set,
                                   framework,
                                   alignment={}):
    """Split nodes into two sets: anchored and abstract

    Anchored nodes: node that anchor on one and only one token
    Abstract nodes: node that anchor on more than one token or zero token
    """

    # For **eds**, nodes connected only with one 'BV' edge
    # is considered abstract
    if framework == 'eds':
        # Split nodes to abstract and anchored set
        abstract_node_id_set = set()
        anchored_nodes = []
        for node in nodes:
            node_id = node.get('id', -1)
            indegree = node_id2indegree[node_id]
            if not indegree and all(
                    edges[edge_id].get('label') == 'BV'
                    for edge_id in node_id2edge_id_set[node_id]):
                abstract_node_id_set.add(node_id)
            else:
                anchored_nodes.append(node)

        assert all(
            len(node.get('anchors', [])) == 1 for node in anchored_nodes)
        assert all('from' in node.get('anchors')[0] for node in anchored_nodes)
        assert all('to' in node.get('anchors')[0] for node in anchored_nodes)

        # Sort nodes according to anchor range
        anchored_nodes.sort(
            key=
            lambda node: (node['anchors'][0]['to'], -node['anchors'][0]['from'])
        )

    # For **ucca**, nodes without anchors
    # is considered abstract
    elif framework == 'ucca':
        abstract_node_id_set = set()
        anchored_nodes = []

        for node in nodes:
            if 'anchors' in node:
                anchored_nodes.append(node)
            else:
                abstract_node_id_set.add(node.get('id', -1))
        assert all(
            len(node.get('anchors', [])) >= 1 for node in anchored_nodes)
        assert all(
            all('from' in anchor and 'to' in anchor
                for anchor in node.get('anchors')) for node in anchored_nodes)

        # Sort nodes according to anchor range
        anchored_nodes.sort(
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
        anchored_nodes = []

        node_id2token_poss = {}
        for node in alignment.get('nodes', []):
            node_id = node.get('id', -1)
            token_poss = node.get('label', [])
            node_id2token_poss[node_id] = token_poss

        for node in nodes:
            node_id = node.get('id', -1)
            if node_id in node_id2token_poss:
                anchored_nodes.append(node)
            else:
                abstract_node_id_set.add(node_id)

        assert all(
            len(node_id2token_poss.get(node.get('id', -1), [])) >= 1
            for node in anchored_nodes)

        anchored_nodes.sort(key=lambda node: (
            max(node_id2token_poss[node.get('id', -1)]),
            -min(node_id2token_poss[node.get('id', -1)]),
        ))
    return (anchored_nodes, abstract_node_id_set)


def _generate_parser_states(nodes, edges, node_id2node, anchored_nodes,
                            abstract_node_id_set, node_id2edge_id_set):
    """Generate the parse state at each token step"""

    # TODO(Sunny): Order of producing high level node
    #                i.e. when should we produce high level node
    seen_node_id_set = set()
    seen_edge_id_set = set()
    parser_states = []

    node_queue = deque(anchored_nodes)

    while node_queue:
        node = node_queue.popleft()
        edge_state = []
        abstract_node_state = []
        node_id = node.get('id', -1)
        seen_node_id_set.add(node_id)

        for edge_id in node_id2edge_id_set[node_id]:
            edge = edges[edge_id]

            source_id = edge.get('source', -1)
            target_id = edge.get('target', -1)

            is_remote_edge = 'properties' in edge and 'remote' in edge[
                'properties']

            # Handle abstract nodes if not remote edge
            # TODO(Sunny): This cannot handle the case of a node having
            #                All childs as abstract_node
            if not is_remote_edge and node_id not in abstract_node_id_set:
                for neig_id in [source_id, target_id]:
                    is_abstract = neig_id in abstract_node_id_set
                    seen = neig_id in seen_node_id_set
                    if is_abstract and not seen:
                        abstract_node_state.append(neig_id)
                        seen_node_id_set.add(neig_id)
                        node_queue.appendleft(node_id2node[neig_id])

            edge_not_seen = edge_id not in seen_edge_id_set
            edge_nodes_seen = all([
                source_id in seen_node_id_set,
                target_id in seen_node_id_set,
            ])

            # add edge if edge not seen and both ends seen
            if edge_not_seen and edge_nodes_seen:
                edge_state.append(edge_id)
                seen_edge_id_set.add(edge_id)
        parser_states.append((node_id, edge_state, abstract_node_state))
    return parser_states
