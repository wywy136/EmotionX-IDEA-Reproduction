from argparse import Namespace
import torch


args = Namespace(model_path='./model/17_',
                 log_path='./log/17.log',
                 pretrain_log_path='./pretrain_log/1.log',
                 pretrained_model_path='./pretrained_model/',
                 device=torch.device('cuda'),
                 cuda_index=1,
                 epoch_num=40,
                 pretrain_epoch_num=20,
                 batch_size=16,
                 pretrain_batch_size=16,
                 emo_num=4,
                 tuning_rate=1e-6,
                 learning_rate=2.5e-6,
                 pretrain_rate=1e-7,
                 warm_up=0.1,
                 pretrain_warm_up=0.1,
                 weight_decay=0.1,
                 pretrain_weight_decay=0.01,
                 dropout_rate=0.75
                 )