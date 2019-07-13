import os

import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from PIL import Image

HIDE_FIELD_SET = set(['anchors', 'source', 'target', 'label'])
UNKWOWN = 'UNKWOWN'


def add_nodes_to_directed_graph(nodes, dg):
    for node in nodes:
        node_id = node.get('id', -1)
        dg.add_node(node_id)
        info_texts = [node.get('label', '')] + [
            str((key[:3], value)) for key, value in node.items()
            if key not in HIDE_FIELD_SET
        ]
        dg.nodes[node_id]['label'] = '\n'.join(info_texts)


def add_edges_to_directed_graph(edges, dg):
    for edge in edges:
        edge_source = edge.get('source', -1)
        edge_target = edge.get('target', -1)
        dg.add_edge(edge_source, edge_target)
        info_texts = [edge.get('label', '')] + [
            str((key[:3], value)) for key, value in edge.items()
            if key not in HIDE_FIELD_SET
        ]
        dg[edge_source][edge_target]['label'] = '\n'.join(info_texts)


def parse_to_directed_graph(parse):
    dg = nx.DiGraph()
    nodes = []
    edges = []
    for word_record in parse:
        wid, word, _, POS, _, _, arc_source, arc_type, *_ = word_record
        wid = int(wid)
        arc_source = int(arc_source)
        nodes.append({'id': wid, 'label': '{}({})'.format(word, POS)})
        edges.append({'target': wid, 'source': arc_source, 'label': arc_type})
    add_nodes_to_directed_graph(nodes, dg)
    add_edges_to_directed_graph(edges, dg)
    return dg


def mrp_json_to_directed_graph(mrp_json):
    dg = nx.DiGraph()
    nodes = mrp_json.get('nodes', [])
    edges = mrp_json.get('edges', [])
    add_nodes_to_directed_graph(nodes, dg)
    add_edges_to_directed_graph(edges, dg)
    return dg


def directed_graph_to_graphviz_image(dg, image_path, layout='dot', verbose=0):
    ag = to_agraph(dg)
    ag.layout(layout)
    ag.draw(image_path)
    if verbose:
        pil_im = Image.open(image_path, 'r')
        plt.imshow(pil_im)


def draw_mrp_graphviz(mrp_json, dataset_dir):
    mrp_id = mrp_json.get('id', UNKWOWN)
    dg = mrp_json_to_directed_graph(mrp_json)
    image_path = os.path.join(dataset_dir, '{}.png'.format(mrp_id))
    directed_graph_to_graphviz_image(dg, image_path)
