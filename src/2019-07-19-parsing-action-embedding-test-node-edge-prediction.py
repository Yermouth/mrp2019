#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    __IPYTHON__
    USING_IPYTHON = True
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except NameError:
    USING_IPYTHON = False


# #### Argparse

# In[2]:


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('project_root', help='')
ap.add_argument('--mrp-data-dir', default='data', help='')
ap.add_argument('--mrp-test-dir', default='src/tests', help='')
ap.add_argument('--tests-fixtures-file-template', default='fixtures/{}-test.jsonl', help='')

ap.add_argument('--graphviz-sub-dir', default='visualization/graphviz', help='')
ap.add_argument('--train-sub-dir', default='training', help='')
ap.add_argument('--companion-sub-dir', default='companion')
ap.add_argument('--jamr-alignment-file', default='jamr.mrp')

ap.add_argument('--test-input-file', default='evaluation/input.mrp', help='')
ap.add_argument('--test-companion-file', default='evaluation/udpipe.mrp', help='')
ap.add_argument('--allennlp-mrp-json-file-template', default='allennlp-mrp-json-big-{}-{}.jsonl', help='')
ap.add_argument('--data-size-limit', type=int, default=10000, help='')

ap.add_argument('--mrp-file-extension', default='.mrp')
ap.add_argument('--companion-file-extension', default='.conllu')
ap.add_argument('--graphviz-file-template', default='http://localhost:8000/files/proj29_ds1/home/slai/mrp2019/visualization/graphviz/{}/{}.mrp/{}.png')
ap.add_argument('--parse-plot-file-template', default='http://localhost:8000/files/proj29_ds1/home/slai/mrp2019/visualization/graphviz/{}/{}.png')

ap.add_argument('--cuda-device', type=int, default=0)


arg_string = """
    /data/proj29_ds1/home/slai/mrp2019
"""
arguments = [arg for arg_line in arg_string.split(r'\\n') for arg in arg_line.split()]


# In[3]:


if USING_IPYTHON:
    args = ap.parse_args(arguments)
else:
    args = ap.parse_args()


# In[4]:


args


# #### Library imports

# In[5]:


import json
import logging
import os
import pprint
import re
import string
from collections import Counter, defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plot_util
import torch
from action_state import mrp_json2parser_states, _generate_parser_action_states
from action_state import ERROR, APPEND, RESOLVE, IGNORE
from preprocessing import (CompanionParseDataset, MrpDataset, JamrAlignmentDataset,
                           read_companion_parse_json_file, read_mrp_json_file, parse2parse_json)            
from torch import nn
from tqdm import tqdm


# #### ipython notebook specific imports

# In[6]:


if USING_IPYTHON:
    # matplotlib config
    get_ipython().magic(u'matplotlib inline')


# In[7]:


sh = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)-8s [%(name)s:%(lineno)d] %(message)s')
sh.setFormatter(formatter)
logging.basicConfig(
    level=logging.DEBUG, 
    handlers=[sh]
)
mute_logger_names = ['allennlp.data.iterators.data_iterator']
for logger_name in mute_logger_names:
    logger = logging.getLogger(logger_name)  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)


# ### Constants

# In[8]:


UNKWOWN = 'UNKWOWN'


# ### Load data

# In[9]:


train_dir = os.path.join(args.project_root, args.mrp_data_dir, args.train_sub_dir)


# In[10]:


mrp_dataset = MrpDataset()


# In[11]:


frameworks, framework2dataset2mrp_jsons = mrp_dataset.load_mrp_json_dir(
    train_dir, args.mrp_file_extension)


# In[12]:


framework2dataset2mrp_jsons.keys()


# ### Data Preprocessing companion

# In[13]:


companion_dir = os.path.join(args.project_root, args.mrp_data_dir, args.companion_sub_dir)


# In[14]:


cparse_dataset = CompanionParseDataset()


# In[15]:


dataset2cid2parse = cparse_dataset.load_companion_parse_dir(companion_dir, args.companion_file_extension)


# In[16]:


dataset2cid2parse_json = cparse_dataset.convert_parse2parse_json()


# In[17]:


dataset2cid2parse.keys()


# In[18]:


# Some data is missing
'20003001' in dataset2cid2parse['wsj']


# ### Load JAMR alignment data

# In[19]:


jalignment_dataset = JamrAlignmentDataset()


# In[20]:


cid2alignment = jalignment_dataset.load_jamr_alignment_file(os.path.join(
    args.project_root,
    args.mrp_data_dir,
    args.companion_sub_dir,
    args.jamr_alignment_file
))


# ### Load testing data

# In[21]:


test_input_filename = os.path.join(args.project_root, args.mrp_data_dir, args.test_input_file)
test_companion_filename = os.path.join(args.project_root, args.mrp_data_dir, args.test_companion_file)


# In[22]:


test_mrp_jsons = read_mrp_json_file(test_input_filename)
test_parse_jsons = read_companion_parse_json_file(test_companion_filename)


# In[23]:


parse_json = test_parse_jsons['102990']


# In[24]:


mrp_json = framework2dataset2mrp_jsons['psd']['wsj'][1]


# In[25]:


test_configs = [
    ('ucca', 'wiki', 70),
]
framework, dataset, idx = test_configs[0]


# In[26]:


mrp_json = framework2dataset2mrp_jsons[framework][dataset][idx]
cid = mrp_json.get('id')


# In[27]:


parse_json = dataset2cid2parse_json[dataset][cid]


# In[ ]:





# In[28]:


doc = mrp_json['input']


# In[29]:


doc


# In[30]:


token_pos = 0
anchors = []
char_pos2tokenized_parse_node_id = []

for node_id, node in enumerate(parse_json.get('nodes')):
    label = node.get('label')
    label_size = len(label)
    while doc[token_pos] == ' ':
        token_pos += 1
        char_pos2tokenized_parse_node_id.append(node_id)
    anchors.append((token_pos, token_pos + label_size))
    char_pos2tokenized_parse_node_id.extend([node_id] * (label_size))
    print(node_id, doc[token_pos: token_pos + label_size], anchors[-1], len(char_pos2tokenized_parse_node_id))
    token_pos += label_size


# In[ ]:





# In[31]:


doc


# In[32]:


len(char_pos2tokenized_parse_node_id)


# In[33]:


doc = mrp_json['input']


# In[34]:


mrp_json['tops']


# In[35]:


mrp_parser_states, mrp_meta_data = mrp_json2parser_states(
    mrp_json, 
    tokenized_parse_nodes=parse_json['nodes'],
)


# In[36]:


(
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
    curr_node_ids,
    token_states,
    actions,
) = mrp_meta_data


# In[ ]:





# In[37]:


curr_node_ids = mrp_meta_data[-3]
token_states = mrp_meta_data[-2]
actions = mrp_meta_data[-1]


# In[38]:


*_, curr_node_ids, token_states, actions = mrp_meta_data


# In[39]:


actions[1]


# In[40]:


actions[-4:]


# In[ ]:





# In[41]:


for curr_node_id, action, token_state in zip(curr_node_ids, actions, token_states):
    action_type, params = action
#     pprint.pprint((curr_node_id, action[0]))
#     pprint.pprint(([token_group[:4] for token_group in token_state]))
    pprint.pprint((curr_node_id, action[0], [token_group[:5] for token_group in token_state]))


# In[ ]:





# In[42]:


for curr_node_id, action, token_state in zip(curr_node_ids, actions, [[]] + token_states):
    action_type, params = action
    pprint.pprint((curr_node_id, action[0], (action[1] or [None])[:2], [token_group[:4] for token_group in token_state]))
    


# In[43]:


actions


# In[ ]:





# In[44]:


token_states[1]


# In[45]:


[n['label'] for n in parse_json['nodes']]


# In[ ]:





# In[46]:


token_states[-1]


# In[ ]:





# In[ ]:





# In[47]:


companion_parser_states, companion_meta_data = mrp_json2parser_states(
    parse_json,
    mrp_doc=doc,
    tokenized_parse_nodes=parse_json['nodes'],
)


# In[ ]:





# In[ ]:





# In[48]:


logger.info(args.graphviz_file_template.format(
    framework, dataset, cid))


# In[ ]:





# In[49]:


mrp_json['input']


# In[50]:


mrp_parser_states


# In[ ]:





# In[51]:


[(node['id'], node.get('label')) for node in mrp_json['nodes']]


# In[52]:


doc


# In[53]:


parse_json['nodes']


# In[54]:


[(node['id'], node['label']) for node in parse_json['nodes']]


# In[ ]:





# In[55]:


anchors


# In[ ]:





# In[ ]:





# ### Create training instance

# In[ ]:





# In[56]:


# framework = 'ucca'
# ignore_framework_set = {'amr', 'dm', 'psd', 'eds'}
# dataset = 'wiki'
# ignore_dataset_set = {}

# framework = 'dm'
# ignore_framework_set = {'amr', 'ucca', 'psd', 'eds'}
# dataset = 'wsj'
# ignore_dataset_set = {}

framework = 'ucca'
ignore_framework_set = {'amr', 'psd', 'eds'}
dataset = 'wiki'
ignore_dataset_set = {}


# In[57]:


frameworks


# In[58]:


framework_names = '-'.join([
    framework 
    for framework in frameworks 
    if framework not in ignore_framework_set
])
framework_names


# In[59]:


allennlp_tests_fixtures_output_file = os.path.join(
    args.project_root, args.mrp_test_dir, args.tests_fixtures_file_template.format(framework_names))

allennlp_framework_train_output_file = os.path.join(
    args.project_root, args.allennlp_mrp_json_file_template.format(framework_names, 'train'))

allennlp_framework_test_output_file = os.path.join(
    args.project_root, args.allennlp_mrp_json_file_template.format(framework_names, 'test'))


# In[60]:


# Create tests fixture jsonl
fixture_combinations = [
    ('ucca', 'wiki', 70),
    ('dm', 'wsj', 3)
] * 5

with open(allennlp_tests_fixtures_output_file, 'w') as wf:
    for framework, dataset, idx in fixture_combinations:
        mrp_json = framework2dataset2mrp_jsons[framework][dataset][idx]
        cid = mrp_json.get('id')
        doc = mrp_json.get('input')
        
        alignment = {}
        if framework == 'amr':
            alignment = cid2alignment[cid]  
        parse_json = dataset2cid2parse_json.get(dataset, {}).get(cid, {})

        if parse_json:
            mrp_parser_states, mrp_meta_data = mrp_json2parser_states(
                mrp_json, 
                tokenized_parse_nodes=parse_json['nodes'],
                alignment=alignment,
            )
            companion_parser_states, companion_meta_data = mrp_json2parser_states(
                parse_json, 
                mrp_doc=doc,
                tokenized_parse_nodes=parse_json['nodes'],
            )

            data_instance = {
                'mrp_json': mrp_json,
                'parse_json': parse_json,
                'mrp_parser_states': mrp_parser_states,
                'mrp_meta_data': mrp_meta_data,
                'companion_parser_states': companion_parser_states,
                'companion_meta_data': companion_meta_data,
            }
            json_encoded_instance = json.dumps(data_instance)
            wf.write(json_encoded_instance + '\n')


# In[ ]:





# In[61]:


for idx in range(20):
    mrp_json = framework2dataset2mrp_jsons[framework][dataset][idx]
    cid = mrp_json.get('id')
    if cid in dataset2cid2parse[dataset]:
        print(idx)


# In[62]:


[state[-1] for state in mrp_parser_states]


# In[63]:


mrp_meta_data[-1]


# In[64]:


doc


# In[65]:


parse_json


# In[66]:


[n['values'][2] for n in parse_json['nodes']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


# Create train jsonl
if os.path.isfile(allennlp_framework_train_output_file) and os.path.isfile(
    allennlp_framework_train_output_file):
    logger.info('allennlp_train_output_file found, stop generation')
else:
    pass
if 1==1:
    data_size = 0
    with open(allennlp_framework_train_output_file, 'w') as train_wf:
        with open(allennlp_framework_test_output_file, 'w') as test_wf:
            for _, dataset, idx, mrp_json in tqdm(mrp_dataset.mrp_json_generator(
                ignore_framework_set=ignore_framework_set,
                ignore_dataset_set=ignore_dataset_set,
                data_size_limit=args.data_size_limit * 2
            )):
                cid = mrp_json.get('id')
                doc = mrp_json.get('input')

                framework = mrp_json.get('framework')
                alignment = {}
                if framework == 'amr':
                    alignment = cid2alignment[cid]  
                parse_json = dataset2cid2parse_json.get(dataset, {}).get(cid, {})

                if parse_json:
                    mrp_parser_states, mrp_meta_data = mrp_json2parser_states(
                        mrp_json, 
                        tokenized_parse_nodes=parse_json['nodes'],
                        alignment=alignment,
                    )
                    companion_parser_states, companion_meta_data = mrp_json2parser_states(
                        parse_json, 
                        mrp_doc=doc,
                        tokenized_parse_nodes=parse_json['nodes'],
                    )

                    # Continue if error
                    if not mrp_parser_states:
                        continue

                    data_instance = {
                        'mrp_json': mrp_json,
                        'parse_json': parse_json,
                        'mrp_parser_states': mrp_parser_states,
                        'mrp_meta_data': mrp_meta_data,
                        'companion_parser_states': companion_parser_states,
                        'companion_meta_data': companion_meta_data,
                    }
                    json_encoded_instance = json.dumps(data_instance)
                    if idx <= args.data_size_limit:
                        train_wf.write(json_encoded_instance + '\n')
                    else:
                        test_wf.write(json_encoded_instance + '\n')


# In[ ]:





# ### Test allennlp dataset reader

# In[68]:


import torch.optim as optim

from mrp_library.dataset_readers.mrp_jsons import MRPDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

import json
import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from overrides import overrides


# In[812]:


from mrp_library.dataset_readers.mrp_jsons_actions import MRPDatasetActionReader


# In[813]:


reader = MRPDatasetActionReader()


# In[814]:


train_dataset = reader.read(cached_path(allennlp_framework_train_output_file))


# In[815]:


# test_dataset = reader.read(cached_path(allennlp_train_output_file))
test_dataset = reader.read(cached_path(allennlp_framework_test_output_file))


# In[816]:


tests_fixtures_dataset = reader.read(cached_path(allennlp_tests_fixtures_output_file))


# In[ ]:





# In[ ]:





# In[817]:


vocab = Vocabulary.from_instances(
    train_dataset + test_dataset + tests_fixtures_dataset,
    non_padded_namespaces=[
        'action_type', 
        'resolved_label_root_label', 
        'resolved_label_edge_labels',
    ],
)


# In[818]:


vocab.print_statistics()


# In[872]:


vocab.get_token_from_index(0, namespace='resolved_label_edge_labels')


# In[873]:


vocab.get_vocab_size('token_node_label')


# In[874]:


vocab.get_vocab_size('resolved_label_edge_labels')


# In[875]:


vocab.get_vocab_size('word')


# In[876]:


vocab.get_vocab_size('pos')


# In[877]:


vocab.get_vocab_size('label')


# In[878]:


EMBEDDING_DIM = 100
HIDDEN_DIM = 50


# ### Test model

# In[826]:


from mrp_library.models.generalizer import ActionGeneralizer
from mrp_library.iterators.same_instance_type_framework_stack_len_iterator import SameInstanceTypeFrameworkStackLenIterator

from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.nn.activations import Activation
from allennlp.common.params import Params
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper


# In[879]:


if torch.cuda.is_available() and False:
    cuda_device = args.cuda_device
else:
    cuda_device = -1
cuda_device


# In[962]:


def _cuda(variable, cuda_device):
    if cuda_device != -1:
        variable = variable.cuda(cuda_device)
    return variable


# In[963]:


field_types = ['word', 'pos', 
               'ucca_word', 'ucca_pos',
               'dm_word', 'dm_pos', 
               'resolved', 'token_node_label', 'token_node_prev_action']
field_type2embedder = {}
field_type2seq2vec_encoder = {}
field_type2seq2seq_encoder = {}

for field_type in field_types:
    embedding = _cuda(
        Embedding(
            num_embeddings=vocab.get_vocab_size(field_type),
            embedding_dim=EMBEDDING_DIM
        ), cuda_device)
    embedder = BasicTextFieldEmbedder({field_type: embedding})
    field_type2embedder[field_type] = embedder
    
    field_type2seq2vec_encoder[field_type] = PytorchSeq2VecWrapper(
        _cuda(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True), cuda_device)
    )
    field_type2seq2seq_encoder[field_type] = PytorchSeq2SeqWrapper(
        _cuda(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True), cuda_device)
    )


# In[964]:


# word_embedding = Embedding(num_embeddings=vocab.get_vocab_size('word'),
#                             embedding_dim=EMBEDDING_DIM)
# pos_embedding = Embedding(num_embeddings=vocab.get_vocab_size('pos'),
#                             embedding_dim=EMBEDDING_DIM)

# word_embedder = BasicTextFieldEmbedder({
#     "word": word_embedding,
#     "pos": pos_embedding,
# })
# parse_label = {
#     'word': torch.LongTensor(
#         [
#             [ 1,  0,  3,  7,  2,  9,  4],
#             [ 0,  0,  5,  0,  0,  0,  4]
#         ]
#     ),
#     'pos': torch.LongTensor(
#         [
#             [ 1,  0,  3,  7,  2,  9,  4],
#             [ 0,  0,  5,  0,  0,  0,  4]
#         ]
#     )
# }


# In[965]:


# embedded_parse_label = word_embedder(parse_label)


# In[966]:


# embedded_parse_label.shape


# In[967]:


vocab.get_vocab_size('resolved_label_edge_labels')


# In[968]:


action_classifier_params = Params({
  "input_dim": HIDDEN_DIM * 3,
  "num_layers": 2,
  "hidden_dims": [50, 3],
  "activations": ["sigmoid", "linear"],
  "dropout": [0.0, 0.0]
})

action_num_pop_classifier_params = Params({
  "input_dim": HIDDEN_DIM * 2,
  "num_layers": 2,
  "hidden_dims": [50, 1],
  "activations": ["sigmoid", "linear"],
  "dropout": [0.0, 0.0]
})

framework2field_type2feedforward_params = {
    'dm':{
        'child_edges': Params({
            "input_dim": HIDDEN_DIM * 4,
            "num_layers": 2,
            "hidden_dims": [100, vocab.get_vocab_size('resolved_label_edge_labels')],
            "activations": ["sigmoid", "linear"],
            "dropout": [0.0, 0.0]
        })
    },
    'ucca':{
        'child_edges': Params({
            "input_dim": HIDDEN_DIM * 4,
            "num_layers": 2,
            "hidden_dims": [100, vocab.get_vocab_size('resolved_label_edge_labels')],
            "activations": ["sigmoid", "linear"],
            "dropout": [0.0, 0.0]
        })
    },
}


# In[969]:


action_classifier_feedforward = FeedForward.from_params(action_classifier_params)
action_classifier_feedforward = _cuda(action_classifier_feedforward, cuda_device)

action_num_pop_classifier_feedforward = FeedForward.from_params(action_num_pop_classifier_params)
action_num_pop_classifier_feedforward = _cuda(action_num_pop_classifier_feedforward, cuda_device)


# In[970]:


framework2field_type2feedforward = {}

for framework, field_type2feedforward_params in framework2field_type2feedforward_params.items():
    framework2field_type2feedforward[framework] = {}
    for field_type, feedforward_params in field_type2feedforward_params.items():
        feedforward_classifier = FeedForward.from_params(
            feedforward_params)
        feedforward_classifier = _cuda(feedforward_classifier, cuda_device)
        framework2field_type2feedforward[framework][field_type] = feedforward_classifier
    framework2field_type2feedforward[framework] = torch.nn.ModuleDict(framework2field_type2feedforward[framework])

framework2field_type2feedforward = torch.nn.ModuleDict(framework2field_type2feedforward)


# In[ ]:





# In[971]:



field_type = 'word'


# In[972]:


parse_label = {
    field_type: _cuda(
        torch.LongTensor(
            [
                [ 1,  0,  3,  7,  2,  9,  4],
                [ 0,  0,  5,  0,  0,  0,  4]
            ]
        ),
        cuda_device
    )
}
embedded_parse_label = field_type2embedder[field_type](parse_label)


# In[973]:


feature_mask = util.get_text_field_mask(parse_label)


# In[974]:


seq2vec_encoder = field_type2seq2vec_encoder[field_type]


# In[975]:


encoded_feature = seq2vec_encoder(embedded_parse_label, feature_mask)


# In[976]:


encoded_features = [encoded_feature] * 3


# In[977]:


# torch.cat(encoded_features, dim=-1)


# In[ ]:





# In[978]:


logits = action_classifier_feedforward(torch.cat(encoded_features, dim=-1))


# In[979]:


logits.shape


# In[980]:


label = torch.tensor([1, 0])


# In[981]:


# loss_func = torch.nn.CrossEntropyLoss()
# loss = loss_func(logits, label)


# In[982]:


ActionGeneralizer = None


# In[983]:


from mrp_library.models.generalizer import ActionGeneralizer
from mrp_library.iterators.same_instance_type_framework_stack_len_iterator import SameInstanceTypeFrameworkStackLenIterator


# In[ ]:





# In[984]:


ActionGeneralizer


# In[985]:


if torch.cuda.is_available() and False:
    cuda_device = args.cuda_device
    model = ActionGeneralizer(
        cuda_device=cuda_device,
        vocab=vocab,
        field_type2embedder=field_type2embedder,
        field_type2seq2vec_encoder=field_type2seq2vec_encoder,
        field_type2seq2seq_encoder=field_type2seq2seq_encoder,
        action_classifier_feedforward=action_classifier_feedforward,
        action_num_pop_classifier_feedforward=action_num_pop_classifier_feedforward,
        framework2field_type2feedforward=framework2field_type2feedforward,
    )
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
    model = ActionGeneralizer(
        cuda_device=cuda_device,
        vocab=vocab,
        field_type2embedder=field_type2embedder,
        field_type2seq2vec_encoder=field_type2seq2vec_encoder,
        field_type2seq2seq_encoder=field_type2seq2seq_encoder,
        action_classifier_feedforward=action_classifier_feedforward,
        action_num_pop_classifier_feedforward=action_num_pop_classifier_feedforward,
        framework2field_type2feedforward=framework2field_type2feedforward,
    )

# iterator = SameInstanceTypeFrameworkIterator(
#     shuffle=True,
#     batch_size=100, 
#     sorting_keys=[("token_node_resolveds", "num_tokens")],
# )
iterator = SameInstanceTypeFrameworkStackLenIterator(
    shuffle=True,
    batch_size=100, 
    sorting_keys=[("token_node_resolveds", "num_tokens")],
)
iterator.index_with(vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[ ]:





# In[ ]:





# In[986]:


cuda_device


# In[987]:


model.resolve_tensor


# In[988]:


# list(model.named_parameters())


# In[1006]:


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=test_dataset,
#     train_dataset=train_dataset,
#     validation_dataset=train_dataset,
    patience=10,
    num_epochs=20,
    cuda_device=cuda_device
)


# In[1007]:


action_logits = torch.tensor([[-2.2126,  2.6022, -1.1655],
        [ 4.7340, -1.9992, -3.4521],
        [-1.9665,  2.4100, -1.2047],
        [-2.1353,  2.4847, -1.1260],
        [ 4.7492, -2.0234, -3.4460],
        [-1.4369,  1.9822, -1.2885],
        [ 1.0337,  0.3599, -2.0420],
        [ 5.0974, -2.3380, -3.4647],
        [ 5.4187, -2.4720, -3.6469],
        [-3.4045,  2.4903,  0.0773],
        [ 0.6384,  0.6764, -1.9942],
        [-2.2904,  2.7170, -1.2016],
        [ 4.6333, -1.9474, -3.4113],
        [-2.0811,  2.5367, -1.2174],
        [ 5.1840, -2.7536, -3.1499],
        [ 4.7421, -2.0138, -3.4485],
        [ 5.1121, -2.2290, -3.5999],
        [ 0.1843,  0.8324, -1.6990],
        [ 5.3854, -2.4593, -3.6309],
        [ 0.0324,  0.6715, -1.4173]])
action_logits = _cuda(action_logits, cuda_device)

action_type = torch.tensor([0, 2, 2, 1, 1, 0, 1, 2, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
action_type = _cuda(action_type, cuda_device)


# In[ ]:





# In[1008]:


action_probs, action_preds = action_logits.max(1)
action_resolve_preds = action_preds.eq_(model.resolve_tensor)


# In[1009]:


iter([1, 2, 3, 4])


# In[1010]:


defaultdict(lambda: defaultdict(dict))


# In[1011]:


(action_resolve_preds, action_type, action_resolve_preds.eq(action_type))


# In[1012]:


action_resolve_preds.unsqueeze(-1).float() * action_logits


# In[1013]:


embedded_fields = torch.ones(99, 62, 100)


# In[1014]:


embedded_fields.size()


# In[1015]:


# embedded_fields


# In[1016]:


root_position_logits = torch.tensor([[0.1722, 0.1723, 0.1720, 0.1721, 0.1721, 0.1719, 0.1717, 0.1719, 0.1717,
         0.1716, 0.1719, 0.1717, 0.1716, 0.1718, 0.1719, 0.1718, 0.1719, 0.1717,
         0.1720, 0.1720, 0.1722, 0.1722, 0.1719, 0.1721, 0.1718, 0.1717, 0.1719,
         0.1717, 0.1720, 0.1715, 0.1718],
        [0.1722, 0.1723, 0.1720, 0.1721, 0.1721, 0.1719, 0.1717, 0.1719, 0.1717,
         0.1716, 0.1719, 0.1717, 0.1716, 0.1718, 0.1719, 0.1718, 0.1719, 0.1717,
         0.1720, 0.1720, 0.1722, 0.1722, 0.1719, 0.1721, 0.1718, 0.1717, 0.1719,
         0.1717, 0.1716, 0.1719, 0.1708],
        [0.1722, 0.1723, 0.1720, 0.1721, 0.1721, 0.1719, 0.1717, 0.1719, 0.1717,
         0.1716, 0.1719, 0.1717, 0.1716, 0.1718, 0.1719, 0.1718, 0.1719, 0.1717,
         0.1720, 0.1720, 0.1722, 0.1722, 0.1719, 0.1721, 0.1718, 0.1717, 0.1719,
         0.1717, 0.1720, 0.1715, 0.1717],
        [0.1722, 0.1723, 0.1720, 0.1721, 0.1721, 0.1719, 0.1717, 0.1719, 0.1717,
         0.1716, 0.1719, 0.1717, 0.1716, 0.1718, 0.1719, 0.1718, 0.1719, 0.1717,
         0.1720, 0.1720, 0.1722, 0.1722, 0.1719, 0.1721, 0.1718, 0.1717, 0.1719,
         0.1717, 0.1716, 0.1719, 0.1717],
        [0.1722, 0.1723, 0.1720, 0.1721, 0.1721, 0.1719, 0.1717, 0.1719, 0.1717,
         0.1716, 0.1719, 0.1717, 0.1716, 0.1718, 0.1719, 0.1718, 0.1719, 0.1717,
         0.1720, 0.1720, 0.1722, 0.1722, 0.1719, 0.1721, 0.1718, 0.1717, 0.1719,
         0.1717, 0.1720, 0.1715, 0.1709]])


# In[ ]:





# In[1017]:


root_position_logits.shape
root_position = torch.tensor([ 2,  0, -1,  1,  0])
mask = root_position.eq(torch.tensor(-1)).unsqueeze(-1).expand_as(root_position_logits)
mask = root_position.gt(-1)
masked_root_position_logits = root_position_logits * mask.float().unsqueeze(-1).expand_as(root_position_logits)
loss = torch.nn.CrossEntropyLoss()
inp = root_position_logits
# crossEntropy = -torch.log(torch.gather(inp, 1, root_position.view(-1, 1)))
root_position
# root_position.eq(torch.tensor(-2)).expand_as(root_position_logits)


# In[ ]:





# In[1019]:


masked_root_position_logits.shape


# In[1020]:


root_position.shape


# In[1021]:


torch.ones(2, 3, 47).transpose(1, 2).shape


# In[1022]:


trainer.train()


# In[ ]:





# In[ ]:


vocab.get_token_from_index(2, namespace='action_type')


# In[ ]:


vocab.get_token_from_index(1, namespace='resolve_label_root_label')


# In[ ]:


token_state = [[26, True, 'H', [[23, True, 'A', [[0, True, 'E', [[0, False, 'The', []]]], [1, True, 'C', [[1, False, 'Lakers', []]]]]], [2, True, 'P', [[2, False, 'advanced', []]]], [25, True, 'A', [[3, True, 'R', [[3, False, 'through', []]]], [4, True, 'E', [[4, False, 'the', []]]], [24, True, 'P', [[5, True, 'T', [[5, False, '1982', []]]], [6, True, 'C', [[6, False, 'playoffs', []]]]]]]]]], [7, True, 'L', [[7, False, 'and', []]]], [8, True, 'P', [[8, False, 'faced', []]]], [9, True, 'A', [[9, False, 'Philadelphia', []]]], [27, True, 'D', [[10, True, 'R', [[10, False, 'for', []]]], [11, True, 'E', [[11, False, 'the', []]]], [12, True, 'Q', [[12, False, 'second', []]]], [13, True, 'C', [[13, False, 'time', []]]]]], [14, True, 'R', [[14, False, 'in', []]]], [15, True, 'Q', [[15, False, 'three', []]]], [16, False, 'years', []]]


# In[ ]:


pprint.pprint(token_state)


# In[ ]:


vocab.get_token_from_index(0, namespace='labels')


# In[ ]:


vocab.get_token_index('RESOLVE')


# In[ ]:





# In[ ]:


vocab.get_token_from_index(3, namespace='resolved')


# In[ ]:


vocab.get_token_from_index(5, namespace='token_node_prev_action')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




