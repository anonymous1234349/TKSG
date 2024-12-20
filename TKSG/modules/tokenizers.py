import json
import os
import re
from collections import Counter
from tqdm import tqdm

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold  # 11
        self.dataset_name = args.dataset_name  # Mimic-cxr
        self.concept_num = args.concept_num
        self.stop_words_list = json.load(open('/data/en.json', 'rb'))
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())

        # 'data/mimic_cxr_vocab.json'
        self.vocab_path = args.vocab_path + self.dataset_name + '_vocab.json'
        if not os.path.isfile(self.vocab_path): 
            with open(self.vocab_path, 'w') as f:
                json.dump(self.create_vocabulary(), f)
                print(f'Vocabulary file {self.vocab_path} created.')
        with open(self.vocab_path, 'r') as f:
            self.merged_data = json.load(f)

        self.token2idx, self.idx2token_tmp = self.merged_data['token2idx'], self.merged_data['idx2token']
        self.idx2token = {int(k): v for k, v in self.idx2token_tmp.items()}  
        # self.token2idx, self.idx2token = self.create_vocabulary_sort()

        if 'concepts_dict' not in self.merged_data or 'reverse_concepts_dict' not in self.merged_data or \
            'concept_word_dict' not in self.merged_data:
            self.concepts_dict, self.reverse_concepts_dict, self.concept_word_dict = self.get_concept()
            self.merged_data['concepts_dict'] = self.concepts_dict
            self.merged_data['reverse_concepts_dict'] = self.reverse_concepts_dict
            self.merged_data['concept_word_dict'] = self.concept_word_dict
            with open(self.vocab_path, 'w') as f:
                json.dump(self.merged_data, f)
                print(f'concepts_dict, reverse_concepts_dict and concept_word_dict save in {self.vocab_path} .')
        else:
            self.concepts_dict_tmp = self.merged_data['concepts_dict']
            self.concepts_dict = {int(k): v for k, v in self.concepts_dict_tmp.items()}  
            self.reverse_concepts_dict_tmp = self.merged_data['reverse_concepts_dict']
            self.reverse_concepts_dict = {int(k): v for k, v in self.reverse_concepts_dict_tmp.items()}  
            self.concept_word_dict = self.merged_data['concept_word_dict']

        self.concept_targets = self.create_concept_target()

    def create_concept_target(self):
        print(f'Selecting {self.concept_num} concepts...')
        self.concept_word_dict = dict(list(self.concept_word_dict.items())[:self.concept_num])
        print('Creating concept targets...')
        concept_target = {}
        for split in ['train', 'val', 'test']:
            for example in tqdm(self.ann[split]):
                tokens = self.clean_report(example['report']).split()
                array = [0] * self.concept_num
                array[0] = 1  # <pad>
                for token in tokens:
                    if token in self.concept_word_dict:
                        array[self.reverse_concepts_dict[self.concept_word_dict[token]]] = 1
                if self.dataset_name == 'mimic_cxr':
                    concept_target.update({example['image_path'][0]: array})
                else: # 'iu_xray'
                    concept_target.update({example['id']: array})

        return concept_target

    def get_concept(self):
        total_tokens = []

        for example in tqdm(self.ann['train']):
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens) 
        tmp = {k: v for k, v in counter.items() if v >= self.threshold}  

        counter_items_desc = sorted(tmp.items(), key=lambda x: x[1], reverse=True) 
        counter_dict_desc = dict(counter_items_desc) 
        words = [k for k, v in counter_dict_desc.items()]
        stop_words = []
        useful_words = []
        for word in words:
            if word in self.stop_words_list:  
                stop_words.append(word)  
            else:
                useful_words.append(word)

        concept_words = useful_words[ : 1000-2]
        concept_words = concept_words + ['<unk>']  # <pad> and <unk>
        concept_dict = {0:0}
        concept_word_dict = {'<pad>':0}
        i = 1
        for word in concept_words:
            concept_dict.update({i: self.token2idx[word]})
            concept_word_dict.update({word: self.token2idx[word]})
            i = i + 1
        concept_words = ['<pad>'] + concept_words
        reverse_concept_dict = {v:k for k,v in concept_dict.items()}
        return concept_dict, reverse_concept_dict, concept_word_dict

    def create_vocabulary_sort(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)
        # len(Counter) = 1501
        counter = Counter(total_tokens) 
        tmp = {k: v for k, v in counter.items() if v >= self.threshold}  

        counter_items_desc = sorted(tmp.items(), key=lambda x: x[1], reverse=True) 
        counter_dict_desc = dict(counter_items_desc) 
        words = [k for k, v in counter_dict_desc.items()]
        stop_words = []
        useful_words = []
        for word in words:
            if word in self.stop_words_list: 
                stop_words.append(word)  
            else:
                useful_words.append(word)

        concept_words = useful_words[ : self.concept_num]
        useful_words = useful_words[self.concept_num : ]

        vocab = concept_words + useful_words + stop_words + ['<unk>']

        print("stop words numbers:",len(stop_words))
        print("vocabulary:", len(vocab))

        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1  # idx + 1
            idx2token[idx + 1] = token  # idx + 1
        return token2idx, idx2token

    def create_vocabulary(self):
        total_tokens = []

        for example in tqdm(self.ann['train']):
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        merged_data = {}
        merged_data.update({'idx2token': idx2token, 'token2idx': token2idx})
        return merged_data

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
