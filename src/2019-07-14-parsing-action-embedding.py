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
ap.add_argument('--tests-fixtures-file', default='fixtures/test.jsonl', help='')

ap.add_argument('--graphviz-sub-dir', default='visualization/graphviz', help='')
ap.add_argument('--train-sub-dir', default='training', help='')
ap.add_argument('--companion-sub-dir', default='companion')
ap.add_argument('--jamr-alignment-file', default='jamr.mrp')

ap.add_argument('--test-input-file', default='evaluation/input.mrp', help='')
ap.add_argument('--test-companion-file', default='evaluation/udpipe.mrp', help='')
ap.add_argument('--allennlp-mrp-json-file-template', default='allennlp-mrp-json-small-{}.jsonl', help='')


ap.add_argument('--mrp-file-extension', default='.mrp')
ap.add_argument('--companion-file-extension', default='.conllu')
ap.add_argument('--graphviz-file-template', default='http://localhost:8000/files/proj29_ds1/home/slai/mrp2019/visualization/graphviz/{}/{}.mrp/{}.png')
ap.add_argument('--parse-plot-file-template', default='http://localhost:8000/files/proj29_ds1/home/slai/mrp2019/visualization/graphviz/{}/{}.png')

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
logging.basicConfig(level=logging.DEBUG, handlers=[sh])
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


# In[ ]:





# ### Data Preprocessing companion

# In[12]:


companion_dir = os.path.join(args.project_root, args.mrp_data_dir, args.companion_sub_dir)


# In[13]:


cparse_dataset = CompanionParseDataset()


# In[14]:


dataset2cid2parse = cparse_dataset.load_companion_parse_dir(companion_dir, args.companion_file_extension)


# In[15]:


dataset2cid2parse_json = cparse_dataset.convert_parse2parse_json()


# In[16]:


dataset2cid2parse.keys()


# In[17]:


# Some data is missing
'20003001' in dataset2cid2parse['wsj']


# ### Load JAMR alignment data

# In[18]:


jalignment_dataset = JamrAlignmentDataset()


# In[19]:


cid2alignment = jalignment_dataset.load_jamr_alignment_file(os.path.join(
    args.project_root,
    args.mrp_data_dir,
    args.companion_sub_dir,
    args.jamr_alignment_file
))


# ### Load testing data

# In[20]:


test_input_filename = os.path.join(args.project_root, args.mrp_data_dir, args.test_input_file)
test_companion_filename = os.path.join(args.project_root, args.mrp_data_dir, args.test_companion_file)


# In[21]:


test_mrp_jsons = read_mrp_json_file(test_input_filename)
test_parse_jsons = read_companion_parse_json_file(test_companion_filename)


# In[ ]:





# In[22]:


parse_json = test_parse_jsons['102990']


# In[23]:


mrp_json = framework2dataset2mrp_jsons['psd']['wsj'][1]


# In[24]:


# framework = 'ucca'
# dataset = 'wiki'

framework = 'dm'
dataset = 'wsj'


# In[25]:


cid = list(dataset2cid2parse_json[dataset].keys())[0]


# In[26]:


idx, mrp_json = [
    (idx, mrp_json)
    for idx, mrp_json in enumerate(framework2dataset2mrp_jsons[framework][dataset])
    if mrp_json.get('id') == cid
][0]
idx


# In[27]:


parse_json = dataset2cid2parse_json[dataset][cid]


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


mrp_meta_data


# In[37]:


curr_node_ids = mrp_meta_data[-3]
token_states = mrp_meta_data[-2]
actions = mrp_meta_data[-1]


# In[38]:


*_, curr_node_ids, token_states, actions = mrp_meta_data


# In[ ]:





# In[39]:


for curr_node_id, action, token_state in zip(curr_node_ids, actions, [[]] + token_states):
    action_type, params = action
    print(curr_node_id, action, token_state)


# In[ ]:





# In[40]:


companion_parser_states, companion_meta_data = mrp_json2parser_states(
    parse_json,
    mrp_doc=doc,
    tokenized_parse_nodes=parse_json['nodes'],
)


# In[ ]:





# In[ ]:





# In[41]:


logger.info(args.graphviz_file_template.format(
    framework, dataset, cid))


# In[ ]:





# In[42]:


mrp_json['input']


# In[43]:


mrp_parser_states


# In[ ]:





# In[ ]:





# In[44]:


[(node['id'], node.get('label')) for node in mrp_json['nodes']]


# In[45]:


doc


# In[46]:


parse_json['nodes']


# In[47]:


[(node['id'], node['label']) for node in parse_json['nodes']]


# In[ ]:





# In[48]:


anchors


# In[ ]:





# In[ ]:





# ### Create training instance

# In[134]:


total_count = 0
with_parse_count = 0
data_size_limit = 20000
ignore_framework_set = {'amr', 'dm', 'psd', 'eds'}
ignore_dataset_set = {}


# In[50]:


allennlp_tests_fixtures_output_file = os.path.join(
    args.project_root, args.mrp_test_dir, args.tests_fixtures_file)
allennlp_train_output_file = os.path.join(
    args.project_root, args.allennlp_mrp_json_file_template.format('train'))
allennlp_test_output_file = os.path.join(
    args.project_root, args.allennlp_mrp_json_file_template.format('test'))


# In[51]:


# Create tests fixture jsonl
fixture_combinations = [
    ('ucca', 'wiki', 70)
]

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
            with_parse_count += 1
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


# In[52]:


[state[-1] for state in mrp_parser_states]


# In[53]:


mrp_meta_data[-1]


# In[54]:


doc


# In[55]:


parse_json


# In[56]:


[n['values'][2] for n in parse_json['nodes']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


# Create train jsonl
if os.path.isfile(allennlp_train_output_file):
    logger.info('allennlp_train_output_file found, stop generation')
else:
    pass
if 1==1:
    data_size = 0
    with open(allennlp_train_output_file, 'w') as wf:
        for _, dataset, mrp_json in tqdm(mrp_dataset.mrp_json_generator(
            ignore_framework_set=ignore_framework_set,
            ignore_dataset_set=ignore_dataset_set,
        )):
            total_count += 1
            if data_size >= data_size_limit:
                break
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
                    
                data_size += 1
                logger.info(data_size)
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

                
# Create test jsonl
if os.path.isfile(allennlp_test_output_file):
    logger.info('allennlp_test_output_file found, stop generation')
else:
    pass
if 1==1:
    data_size = 0
    with open(allennlp_test_output_file, 'w') as wf:
        alignment = {}
        for mrp_json in tqdm(test_mrp_jsons):
            
            if data_size >= data_size_limit:
                break
            cid = mrp_json.get('id', '')
            framework = mrp_json.get('framework', '')
            if framework in ignore_framework_set:
                continue
            parse_json = test_parse_jsons[cid]
            doc = parse_json.get('input')
            companion_parser_states, companion_meta_data = mrp_json2parser_states(
                parse_json, 
                mrp_doc=doc,
                tokenized_parse_nodes=parse_json['nodes'],
            )
            if not companion_parser_states:
                continue
            data_size += 1
            logger.info(data_size)
            data_instance = {
                'mrp_json': mrp_json,
                'parse_json': parse_json,
                'companion_parser_states': companion_parser_states,
                'companion_meta_data': companion_meta_data,
            }
            json_encoded_instance = json.dumps(data_instance)
            wf.write(json_encoded_instance + '\n')


# In[ ]:





# ### Test allennlp dataset reader

# In[58]:


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


# In[59]:


from mrp_library.dataset_readers.mrp_jsons_actions import MRPDatasetActionReader


# In[60]:


reader = MRPDatasetActionReader()


# In[61]:


train_dataset = reader.read(cached_path(allennlp_train_output_file))


# In[62]:


test_dataset = reader.read(cached_path(allennlp_train_output_file))
#test_dataset = reader.read(cached_path(allennlp_test_output_file))


# In[63]:


tests_fixtures_dataset = reader.read(cached_path(allennlp_tests_fixtures_output_file))


# In[64]:


vocab = Vocabulary.from_instances(train_dataset + test_dataset + tests_fixtures_dataset)


# In[65]:


vocab.print_statistics()


# In[66]:


vocab.get_vocab_size('word')


# In[67]:


vocab.get_vocab_size('pos')


# In[68]:


vocab.get_vocab_size('label')


# In[112]:


EMBEDDING_DIM = 100
HIDDEN_DIM = 50


# ### Test model

# In[113]:


from mrp_library.models.generalizer import Generalizer, ActionGeneralizer
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.nn.activations import Activation
from allennlp.common.params import Params
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper


# In[114]:


word_embedding = Embedding(num_embeddings=vocab.get_vocab_size('word'),
                            embedding_dim=EMBEDDING_DIM)
word_embedder = BasicTextFieldEmbedder({"word": word_embedding})


# In[115]:


pos_embedding = Embedding(num_embeddings=vocab.get_vocab_size('pos'),
                            embedding_dim=EMBEDDING_DIM)
pos_embedder = BasicTextFieldEmbedder({"pos": pos_embedding})


# In[117]:


encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))


# In[118]:


classifier_params = Params({
  "input_dim": HIDDEN_DIM * 4 * 3,
  "num_layers": 2,
  "hidden_dims": [50, 3],
  "activations": ["sigmoid", "linear"],
  "dropout": [0.2, 0.0]
})


# In[119]:


classifier_feedforward = FeedForward.from_params(classifier_params)


# In[120]:


parse_label = {
    'word': torch.LongTensor(
        [
            [ 1,  0,  3,  7,  2,  9,  4],
            [ 0,  0,  5,  0,  0,  0,  4]
        ]
    )
}
embedded_parse_label = word_embedder(parse_label)


# In[121]:


feature_mask = util.get_text_field_mask(parse_label)


# In[122]:


encoded_feature = encoder(embedded_parse_label, feature_mask)


# In[123]:


encoded_features = [encoded_feature] * 4 * 3


# In[124]:


torch.cat(encoded_features, dim=-1).shape


# In[125]:


logits = classifier_feedforward(torch.cat(encoded_features, dim=-1))


# In[126]:


logits.shape


# In[127]:


label = torch.tensor([1, 0])


# In[128]:


loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(logits, label)


# In[ ]:





# In[129]:


model = ActionGeneralizer(
    vocab=vocab,
    word_embedder=word_embedder,
    pos_embedder=pos_embedder,
    encoder=encoder,
    classifier_feedforward=classifier_feedforward
)


# In[130]:


optimizer = optim.SGD(model.parameters(), lr=0.01)
cuda_device = -1


# In[131]:


iterator = BucketIterator(batch_size=20, sorting_keys=[("curr_parse_node_label", "num_tokens")])
iterator.index_with(vocab)


# In[132]:


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=train_dataset,
    patience=10,
    num_epochs=20,
    cuda_device=cuda_device
)


# In[133]:


trainer.train()


# In[ ]:





# In[ ]:





# In[ ]:




