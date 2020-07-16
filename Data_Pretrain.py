import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import numpy as np


class EmotionDatasetforPretraining(Dataset):
    def __init__(self):
        super(EmotionDatasetforPretraining, self).__init__()
        self.json_data = None
        # self.data_path = '/home/ramon/wy_uci/torch/data/Friends10/friends_season_01.json'
        self.input = []
        self.mask = []
        self.type = []
        self.source = []
        self.source_MLMlabel = []
        self.scene = []
        self.scene_accum = 0
        self.MLMlabel = []
        self.NSPlabel = []
        self.order_location = []
        self.tokens = []
        self.t = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./my_cache')
        self.sentnum = 0

    def load_from_json(self, data_path):
        with open(data_path, 'r', encoding='utf8')as fp:
            self.json_data = json.load(fp)
            # print(self.json_data['episodes'][0]['scenes'][0]['utterances'][0]['tokens'][0][0])

        for episode in self.json_data['episodes']:
            for scene in episode['scenes']:
                self.scene_accum += 1
                for uttr in scene['utterances']:
                    one_source = []
                    wordnum = 0
                    one_source_mlmlabel = []
                    for sent in uttr['tokens']:
                        for tk in sent:
                            one_source.append(self.t.convert_tokens_to_ids(tk))
                            self.order_location.append((self.sentnum, wordnum))
                            wordnum += 1
                            self.tokens.append(self.t.convert_tokens_to_ids(tk))
                            one_source_mlmlabel.append(-100)
                    self.sentnum += 1
                    self.source.append(one_source)
                    self.source_MLMlabel.append(one_source_mlmlabel)
                    self.scene.append(self.scene_accum)
        print(len(self.source))

    def build(self):
        print('Now preparing data for MLM ...')
        masked_word_num = int(len(self.order_location) * 0.15)
        while masked_word_num > 0:
            if masked_word_num % 1000 == 0:
                print(masked_word_num)
            target_location = random.choice(self.order_location)
            self.order_location.remove(target_location)
            self.source_MLMlabel[target_location[0]][target_location[1]] = self.source[target_location[0]][target_location[1]]
            situation = random.randint(0, 9)
            if situation < 8:  # MASK
                self.source[target_location[0]][target_location[1]] = self.t.convert_tokens_to_ids('[MASK]')
            elif situation == 8:  # Replace
                self.source[target_location[0]][target_location[1]] = random.choice(self.tokens)
            else:  # Reserve
                pass
            masked_word_num -= 1

        print('Now concatenating input ...')
        max_length = 128
        for i in range(len(self.source) - 1):
            if i % 100 == 0:
                print(i, len(self.source))
            if self.scene[i] == self.scene[i + 1]:
                # Continues two sentences
                one_input = [self.t.convert_tokens_to_ids('[CLS]')]
                one_mask = [1]
                one_type = [0]
                one_MLMlable = [-100]
                for j in range(len(self.source[i])):
                    one_input.append(self.source[i][j])
                    one_mask.append(1)
                    one_type.append(0)
                    one_MLMlable.append(self.source_MLMlabel[i][j])
                one_input.append(self.t.convert_tokens_to_ids('[SEP]'))
                one_mask.append(1)
                one_type.append(0)
                one_MLMlable.append(-100)
                for j in range(len(self.source[i + 1])):
                    one_input.append(self.source[i + 1][j])
                    one_mask.append(1)
                    one_type.append(1)
                    one_MLMlable.append(self.source_MLMlabel[i + 1][j])
                self.input.append(one_input)
                self.mask.append(one_mask)
                self.type.append(one_type)
                self.MLMlabel.append(one_MLMlable)
                self.NSPlabel.append(0)
                max_length = max(max_length, len(one_input))

                # non-continues two sentences (different scenes)
                while True:
                    k = random.randint(0, len(self.source) - 1)
                    if self.scene[k] != self.scene[i]:
                        break
                one_input = [self.t.convert_tokens_to_ids('[CLS]')]
                one_mask = [1]
                one_type = [0]
                one_MLMlable = [-100]
                for j in range(len(self.source[i])):
                    one_input.append(self.source[i][j])
                    one_mask.append(1)
                    one_type.append(0)
                    one_MLMlable.append(self.source_MLMlabel[i][j])
                one_input.append(self.t.convert_tokens_to_ids('[SEP]'))
                one_mask.append(1)
                one_type.append(0)
                one_MLMlable.append(-100)
                for j in range(len(self.source[k])):
                    one_input.append(self.source[k][j])
                    one_mask.append(1)
                    one_type.append(1)
                    one_MLMlable.append(self.source_MLMlabel[k][j])
                self.input.append(one_input)
                self.mask.append(one_mask)
                self.type.append(one_type)
                self.MLMlabel.append(one_MLMlable)
                self.NSPlabel.append(1)
                max_length = max(max_length, len(one_input))

        print('Padding...')
        for i in range(len(self.input)):
            self.input[i].extend([self.t.convert_tokens_to_ids('[PAD]')] * (max_length - len(self.input[i])))
            self.mask[i].extend([0] * (max_length - len(self.mask[i])))
            self.type[i].extend([1] * (max_length - len(self.type[i])))
            self.MLMlabel[i].extend([-100] * (max_length - len(self.MLMlabel[i])))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return {'input': np.array(self.input[item]),
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'MLM': np.array(self.MLMlabel[item]),
                'NSP': self.NSPlabel[item]
                }

    def get_batch_num(self, bs):
        return len(self.input) // bs


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cuda"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            out_dict[name] = data_dict[name].to(device)
        yield out_dict