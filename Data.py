import numpy as np
import Config
from transformers import BertTokenizer
import nltk
import json
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    def __init__(self, split):
        super(EmotionDataset, self).__init__()
        self.source = []
        self.mask = []
        self.type = []
        self.emotion = []
        self.emotion_weight = []
        self.emotion_weight_once = [0, 0, 0, 0]
        self.emo_idx = {}
        self.idx_emo = {}
        self.emo_num = {}
        self.emoidx = 0
        self.six_people = ['Rachel', 'Monica', 'Phoebe', 'Joey', 'Chandler', 'Ross']
        self.target_emotion = ['anger', 'joy', 'neutral', 'sadness']
        self.json_data = None
        self.batch_size = Config.args.batch_size
        self.t = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./my_cache')
        self.max_length = 0
        self.split = split
        if split == 'train':
            self.data_path = '/home/ramon/wy_uci/torch/data/friends.augmented.json'
        else:
            self.data_path = '/home/ramon/wy_uci/torch/data/friends.augmented.json'

    def build(self):
        with open(self.data_path, 'r', encoding='utf8')as fp:
            if self.split == 'val':
                self.json_data = json.load(fp)
                self.json_data = self.json_data[800:]
            else:
                self.json_data = json.load(fp)
                self.json_data = self.json_data[:800]

        for i in range(len(self.json_data)):
            for j in range(len(self.json_data[i])):
                if self.json_data[i][j]['emotion'] not in self.target_emotion:
                    continue
                if self.json_data[i][j]['emotion'] == 'anger' or self.json_data[i][j]['emotion'] == 'sadness':
                    self.add(i, j, 'utterance')
                    # self.add(i, j, 'utterance_de')
                    # self.add(i, j, 'utterance_fr')
                    # self.add(i, j, 'utterance_it')
                else:
                    self.add(i, j, 'utterance')

        min_emo_num = min(list(self.emo_num.values()))
        for i in range(len(self.source)):
            self.source[i].extend([self.t.convert_tokens_to_ids('[PAD]')] * (self.max_length - len(self.source[i])))
            self.mask[i].extend([0] * (self.max_length - len(self.mask[i])))
            self.type[i].extend([1] * (self.max_length - len(self.type[i])))
            self.emotion_weight.append(float(min_emo_num / self.emo_num[self.idx_emo[self.emotion[i]]]))

        for emo in self.target_emotion:
            self.emotion_weight_once[self.emo_idx[emo]] = float(min_emo_num) / self.emo_num[emo]

    def get_batch_num(self, batch_size):
        return len(self.source) // batch_size

    def get_emotion_num(self):
        return len(self.emo_idx)

    def add(self, i, j, string):
        # emotion
        if self.json_data[i][j]['emotion'] not in self.emo_idx:
            self.idx_emo[self.emoidx] = self.json_data[i][j]['emotion']
            self.emo_idx[self.json_data[i][j]['emotion']] = self.emoidx
            self.emoidx += 1
        if self.json_data[i][j]['emotion'] not in self.emo_num:
            self.emo_num[self.json_data[i][j]['emotion']] = 0
        else:
            self.emo_num[self.json_data[i][j]['emotion']] += 1
        self.emotion.append(self.emo_idx[self.json_data[i][j]['emotion']])
        # input: [CLS]
        source = [self.t.convert_tokens_to_ids('[CLS]')]
        mask = [1]
        type = [0]
        # personal tokenization
        if self.json_data[i][j]['speaker'] in self.six_people:
            source += self.t.convert_tokens_to_ids(['[{}]'.format(self.json_data[i][j]['speaker']),
                                                    '[says]'])
            mask += [1, 1]
            type += [0, 0]
        # the utterance
        source += self.t.convert_tokens_to_ids([word.lower() for word in nltk.word_tokenize(
            self.json_data[i][j][string])])
        source.append(self.t.convert_tokens_to_ids('[SEP]'))
        mask.extend([1] * (len(source) - len(mask)))
        type.extend([0] * (len(source) - len(type)))
        # the previous utterance
        if j == 0:  # the first utterance in this conversation
            source += self.t.convert_tokens_to_ids(['[None]', '[SEP]'])
            mask += [1, 1]
            type += [1, 1]
        else:
            if self.json_data[i][j - 1]['speaker'] in self.six_people:
                source += self.t.convert_tokens_to_ids(['[{}]'.format(self.json_data[i][j - 1]['speaker']),
                                                        '[says]'])
                mask += [1, 1]
                type += [1, 1]
            source += self.t.convert_tokens_to_ids([word.lower() for word in nltk.word_tokenize(
                self.json_data[i][j - 1][string])])
            source.append(self.t.convert_tokens_to_ids('[SEP]'))
            mask.extend([1] * (len(source) - len(mask)))
            type.extend([1] * (len(source) - len(type)))
        self.max_length = max(self.max_length, len(source))
        self.source.append(source)
        self.mask.append(mask)
        self.type.append(type)

    def __getitem__(self, item):
        return {'source': np.array(self.source[item]),
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'target': self.emotion[item],
                'weight': self.emotion_weight[item],
                'weight_class': np.array(self.emotion_weight_once)
                }

    def __len__(self):
        return len(self.source)


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cuda"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            out_dict[name] = data_dict[name].to(device)
        yield out_dict