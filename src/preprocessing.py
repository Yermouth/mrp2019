import json
import logging
import math
import os
import string

from tqdm import tqdm

logger = logging.getLogger(__name__)

CID_MISMATCH_PREFIXS = ['reviews-']
ALLOWED_FRAMEWORKS = ['ucca', 'psd', 'eds', 'dm', 'amr']


def read_mrp_json_file(dataset_filename, cid2mrp_json={}):
    """Read mrp formatted json file"""
    with open(dataset_filename) as rf:
        for line in rf:
            mrp_json = json.loads(line.strip())
            cid = mrp_json.get('id')
            framework = mrp_json.get('framework', '')
            is_ucca = framework == 'ucca'

            # If MRP framework is ucca, create text label
            if is_ucca and 'nodes' in mrp_json and 'input' in mrp_json:
                input_text = mrp_json['input']
                nodes = mrp_json['nodes']
                for i, node in enumerate(nodes):
                    if 'anchors' not in node:
                        continue
                    text_segments = []
                    for anchor in node['anchors']:
                        text_segments.append(input_text[anchor.get('from', -1):
                                                        anchor.get('to', -1)])
                    mrp_json['nodes'][i]['label'] = ''.join(text_segments)

            for prefix in CID_MISMATCH_PREFIXS:
                if cid.startswith(prefix):
                    cid = cid[len(prefix):]
            cid2mrp_json[cid] = mrp_json
    return cid2mrp_json


class MrpDataset(object):
    def __init__(self):
        self.framework2cid2mrp_json = {}
        self.frameworks = []

    def load_mrp_json_dir(self, dir_name, file_extension):
        """Load mrp json data from official mrp LDC dataset folder"""
        frameworks = [
            sub_dir for sub_dir in os.listdir(dir_name)
            if os.path.isdir(os.path.join(dir_name, sub_dir))
            and sub_dir in ALLOWED_FRAMEWORKS
        ]

        framework2cid2mrp_json = {}
        for framework in tqdm(frameworks, desc='frameworks'):
            framework_dir = os.path.join(dir_name, framework)
            dataset_names = os.listdir(framework_dir)
            cid2mrp_json = {}

            for dataset_name in tqdm(dataset_names, desc='dataset_name'):
                if not dataset_name.endswith(file_extension):
                    continue
                dataset_filename = os.path.join(framework_dir, dataset_name)

                # Remove mrp extension of dataset_name
                read_mrp_json_file(dataset_filename, cid2mrp_json)

            framework2cid2mrp_json[framework] = cid2mrp_json
        self.frameworks = frameworks
        self.framework2cid2mrp_json = framework2cid2mrp_json
        return frameworks, framework2cid2mrp_json

    def mrp_json_generator(
            self,
            data_size_limit=None,
            ignore_framework_set={},
            total_split=None,
            split_no=None,
    ):
        for framework, cid2mrp_json in self.framework2cid2mrp_json.items():
            if framework in ignore_framework_set:
                continue
            cid_mrp_json_pairs = list(cid2mrp_json.items())
            if data_size_limit:
                cid_mrp_json_pairs = cid_mrp_json_pairs[:data_size_limit]

            if total_split is not None and split_no is not None:
                data_size = len(cid_mrp_json_pairs)
                split_size = math.ceil(data_size / total_split)
                start = split_no * split_size
                end = (split_no + 1) * split_size
                cid_mrp_json_pairs = cid_mrp_json_pairs[start:end]

            for cid, mrp_json in cid_mrp_json_pairs:
                yield framework, cid, mrp_json


def read_companion_parse_file(dataset_filename):
    cid2parse = {}
    with open(dataset_filename) as rf:
        parse = []
        for line in rf:
            line = line.strip()
            if not line:
                cid2parse[cid] = parse
                parse = []
                cid = ''
            elif line.startswith('#'):
                cid = line[1:]
            else:
                parse.append(line.split('\t'))
    return cid2parse


def read_companion_parse_file_as_json(dataset_filename):
    cid2parse = {}
    with open(dataset_filename) as rf:
        parse = []
        for line in rf:
            line = line.strip()
            if not line:
                cid2parse[cid] = parse
                parse = []
                cid = ''
            elif line.startswith('#'):
                cid = line[1:]
            else:
                parse.append(line.split('\t'))
    return cid2parse


PARSE_JSON_NODE_PROPERTIES = ['lemma', 'upos', 'xpos']


def parse2parse_json(parse, cid):
    parse_json = {
        'id': cid,
        'tops': [],
        'nodes': [],
        'edges': [],
    }
    for _, org, lemma, upos, xpos, _, arc_target, arc_label, _, token_range in parse:
        node_id = len(parse_json['nodes'])
        node = {
            'id': node_id,
            'label': org,
            'properties': PARSE_JSON_NODE_PROPERTIES,
            'values': [lemma, upos, xpos],
        }
        parse_json['nodes'].append(node)

        target = int(arc_target)
        if target == 0:
            parse_json['tops'].append(node_id)
        else:
            edge = {
                'source': node_id,
                'target': target - 1,
                'label': arc_label,
            }
            parse_json['edges'].append(edge)
    return parse_json


def read_companion_parse_json_file(dataset_filename):
    cid2parse_json = {}
    with open(dataset_filename) as rf:
        for line in rf:
            parse_json = json.loads(line.strip())
            cid = parse_json.get('id', '')
            cid2parse_json[cid] = parse_json
    return cid2parse_json


class CompanionParseDataset(object):
    def __init__(self):
        self.dataset2cid2parse = {}
        self.dataset2cid2parse_json = {}

    def load_companion_parse_dir(self, dir_name, file_extension):
        dataset2cid2parse = {}
        for framework in os.listdir(dir_name):
            framework_dir = os.path.join(dir_name, framework)
            if not os.path.isdir(framework_dir):
                continue

            logger.info('framework {} found'.format(framework))
            for dataset in tqdm(os.listdir(framework_dir), desc='dataset'):
                if not dataset.endswith(file_extension):
                    continue
                dataset_name = dataset.split('.')[0].rstrip(string.digits)
                dataset_filename = os.path.join(framework_dir, dataset)
                cid2parse = read_companion_parse_file(dataset_filename)
                dataset2cid2parse[dataset_name] = cid2parse
        self.dataset2cid2parse = dataset2cid2parse
        return dataset2cid2parse

    def parse_generator(self):
        for dataset, cid2parse in self.dataset2cid2parse.items():
            for cid, parse in cid2parse.items():
                yield dataset, cid, parse

    def convert_parse2parse_json(self):
        dataset2cid2parse_json = {
            dataset: {}
            for dataset in self.dataset2cid2parse
        }
        for dataset, cid, parse in self.parse_generator():
            dataset2cid2parse_json[dataset][cid] = parse2parse_json(parse, cid)
        self.dataset2cid2parse_json = dataset2cid2parse_json
        return self.dataset2cid2parse_json


class JamrAlignmentDataset(object):
    def __init__(self):
        self.cid2alignment = {}

    def load_jamr_alignment_file(self, alignment_filename):
        cid2alignment = {}
        with open(alignment_filename) as rf:
            for line in rf:
                alignment_json = json.loads(line.strip())
                cid = alignment_json.get('id', '')
                cid2alignment[cid] = alignment_json
        self.cid2alignment = cid2alignment
        return cid2alignment
