import Data
import Model
import Config
import Evaluate
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import os


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


def calculate_loss(pred, gold, weight=None):
    return F.cross_entropy(pred, gold.long(), weight=weight, reduction='sum')


args = Config.args
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda_index)
seed_num = 57
torch.manual_seed(seed_num)
dataset_train = Data.EmotionDataset('train')
dataset_train.build()
batch_num_train = dataset_train.get_batch_num(args.batch_size)
dataset_val = Data.EmotionDataset('val')
dataset_val.build()
batch_num_val = dataset_val.get_batch_num(1)

model = Model.IDEAModel()
model = model.to(args.device)
# model.load_state_dict(torch.load('/home/ramon/wy_uci/torch/model/15_19.pth'))

param_optimizer = list(model.named_parameters())
optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in param_optimizer if "_bert" not in n],
                 'lr': args.learning_rate, 'weight_decay': args.weight_decay}]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_rate, correct_bias=False)

training_steps = args.epoch_num * batch_num_train
warmup_steps = int(training_steps * args.warm_up)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
logger = get_logger(args.log_path)

best_f1 = 0.
for epoch in range(args.epoch_num):

    model.train()
    batch_generator = Data.generate_batches(dataset=dataset_train, batch_size=args.batch_size, device=args.device)

    for batch_index, batch_dict in enumerate(batch_generator):
        pred = model(batch_dict['source'].long(),
                     batch_dict['mask'].float(),
                     batch_dict['type'].long())
        if epoch == 0:
            loss = calculate_loss(pred, batch_dict['target'], weight=batch_dict['weight_class'][0].float())  #, weight=batch_dict['weight_class'][0].float()
        else:
            loss = calculate_loss(pred, batch_dict['target'])  #
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        if batch_index % 50 == 0:
            logger.info('Epoch: [{}/{}]\tBatch: [{}/{}]\t Loss: {}'.format(
                epoch, args.epoch_num, batch_index, batch_num_train, loss.item()
            ))

    model.eval()
    batch_generator = Data.generate_batches(dataset=dataset_val, batch_size=1, device=args.device)
    f1 = Evaluate.evaluation(model, batch_generator, logger)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), args.model_path + '{}.pth'.format(epoch))
        logger.info('Model saved after epoch {}'.format(epoch))