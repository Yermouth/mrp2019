import os

import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from PIL import Image


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


def mrp_json_to_directed_graph(mrp_json):
    dg = nx.DiGraph()
    nodes = mrp_json.get('nodes', [])
    edges = mrp_json.get('edges', [])
    add_nodes_to_directed_graph(nodes, dg)
    add_edges_to_directed_graph(edges, dg)
    return dg


def draw_graphviz(mrp_json, dataset_dir):
    mrp_id = mrp_json.get('id', UNKWOWN)
    dg = mrp_json_to_directed_graph(mrp_json)
    save_name = os.path.join(dataset_dir, mrp_id)
    dg2graphviz_image(dg, save_name)


def dg2graphviz_image(dg,
                      save_name,
                      layout='dot',
                      verbose=0,
                      USING_IPYTHON=False):
    ag = to_agraph(dg)
    ag.layout(layout)
    image_name = '{}.png'.format(save_name)
    ag.draw(image_name)
    if verbose and USING_IPYTHON:
        pil_im = Image.open(image_name, 'r')
        plt.imshow(pil_im)
