from transformers import BertForPreTraining, AdamW, get_linear_schedule_with_warmup, BertTokenizer
import Data_Pretrain
import Config
import logging
import torch
import json
from torch.utils.data import Dataset
import random
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(Config.args.cuda_index)
torch.cuda.set_device(Config.args.cuda_index)


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


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# dataset = Data_Pretrain.EmotionDatasetforPretraining()
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_01.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_02.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_03.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_04.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_05.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_06.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_07.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_08.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_09.json')
# dataset.load_from_json('/home/ramon/wy_uci/torch/data/Friends10/friends_season_10.json')
# dataset.build()
# torch.save(dataset, './my_cache/Friendsforpretraining.pth')
dataset = torch.load('/home/ramon/wy_uci/torch/my_cache/Friendsforpretraining.pth')
batch_num = dataset.get_batch_num(Config.args.pretrain_batch_size)
print('+++++++++++++++++')

model = BertForPreTraining.from_pretrained('bert-base-uncased')
model = model.to(Config.args.device)

optimizer = AdamW(model.parameters(), lr=Config.args.pretrain_rate,
                  weight_decay=Config.args.pretrain_weight_decay, correct_bias=False)
training_steps = Config.args.pretrain_epoch_num * batch_num
warmup_steps = int(training_steps * Config.args.pretrain_warm_up)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

logger = get_logger(Config.args.pretrain_log_path)
logger.info('Pretraining FriendsBERT. Batch size: {}, Learning Rate: {}, Weight Decay: {}'.format(
    Config.args.pretrain_batch_size, Config.args.pretrain_rate, Config.args.pretrain_weight_decay
))

for epoch in range(Config.args.pretrain_epoch_num):
    batch_generator = Data_Pretrain.generate_batches(dataset=dataset,
                                                     batch_size=Config.args.pretrain_batch_size,
                                                     device=Config.args.device)
    for batch_index, batch_dict in enumerate(batch_generator):
        loss = model(input_ids=batch_dict['input'],
                     attention_mask=batch_dict['mask'],
                     token_type_ids=batch_dict['type'],
                     masked_lm_labels=batch_dict['MLM'],
                     next_sentence_label=batch_dict['NSP'])[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        if batch_index % 100 == 0:
            logger.info('Epoch: {}/{}\tBatch: {}/{}\tLoss: {}'.format(
                epoch, Config.args.pretrain_epoch_num, batch_index, batch_num, loss.item()
            ))

model.save_pretrained('/home/ramon/wy_uci/torch/pretrained_model/')