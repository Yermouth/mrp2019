import json
import logging
import os
import string

from tqdm import tqdm

logger = logging.getLogger('preprocessing')


class MrpDataset(object):
    def __init__(self):
        self.framework2dataset2mrp_jsons = {}
        self.frameworks = []

    def load_mrp_json_dir(self, dir_name, file_extension):
        """Load mrp json data from official mrp LDC dataset folder"""
        frameworks = [
            sub_dir for sub_dir in os.listdir(dir_name)
            if os.path.isdir(os.path.join(dir_name, sub_dir))
        ]

        framework2dataset2mrp_jsons = {}
        for framework in tqdm(frameworks, desc='frameworks'):
            dataset2mrp_jsons = {}
            framework_dir = os.path.join(dir_name, framework)
            dataset_names = os.listdir(framework_dir)

            for dataset_name in tqdm(dataset_names, desc='dataset_name'):
                if not dataset_name.endswith(file_extension):
                    continue
                dataset_filename = os.path.join(framework_dir, dataset_name)

                # Remove mrp extension of dataset_name
                dataset_name = dataset_name.split('.')[0]
                mrp_jsons = self._read_mrp_jsons(framework, dataset_filename)
                dataset2mrp_jsons[dataset_name] = mrp_jsons

            framework2dataset2mrp_jsons[framework] = dataset2mrp_jsons
        self.frameworks = frameworks
        self.framework2dataset2mrp_jsons = framework2dataset2mrp_jsons
        return frameworks, framework2dataset2mrp_jsons

    def _read_mrp_jsons(self, framework, dataset_filename):
        """Read mrp formatted json file"""
        mrp_jsons = []
        with open(dataset_filename) as rf:
            for line in rf:
                mrp_json = json.loads(line.strip())
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
                            text_segments.append(
                                input_text[anchor.get('from', -1):anchor.
                                           get('to', -1)])
                        mrp_json['nodes'][i]['label'] = ''.join(text_segments)
                mrp_jsons.append(mrp_json)
        return mrp_jsons


class CompanionParseDataset(object):
    def __init__(self):
        self.dataset2cid2parse = {}

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
                cid2parse = {}
                with open(os.path.join(framework_dir, dataset)) as rf:
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
                dataset2cid2parse[dataset_name] = cid2parse
        self.dataset2cid2parse = dataset2cid2parse
        return dataset2cid2parse


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
