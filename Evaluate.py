import torch
import Data
import Model
import Config


def evaluation(model, batch_generator_val, logger):
    precision = [0, 0, 0, 0]  # 0: neutral; 1: joy; 2: sadness; 3: anger
    recall = [0, 0, 0, 0]
    f1 = [0, 0, 0, 0]
    match = [0, 0, 0, 0]
    pre_denom = [1e-5, 1e-5, 1e-5, 1e-5]
    rec_denom = [1e-5, 1e-5, 1e-5, 1e-5]
    for batch_index, batch_dict in enumerate(batch_generator_val):
        # if batch_index % 20 == 0:
        #     print(batch_index)
        pred = model(batch_dict['source'].long(),
                     batch_dict['mask'].float(),
                     batch_dict['type'].long())
        if int(torch.argmax(pred[0], 0)) == int(batch_dict['target'][0]):
            match[int(batch_dict['target'][0])] += 1
        pre_denom[int(torch.argmax(pred[0], 0))] += 1
        rec_denom[int(batch_dict['target'][0])] += 1

    for i in range(4):
        precision[i] = float(match[i]) / float(pre_denom[i])
        recall[i] = float(match[i]) / float(rec_denom[i])
        f1[i] = 2 * float(precision[i]) * float(recall[i]) / (float(precision[i]) + float(recall[i]) + 1e-5)

    pre_overall = float(sum(match)) / float(sum(pre_denom))
    rec_overall = float(sum(match)) / float(sum(rec_denom))
    f1_overall = 2 * pre_overall * rec_overall / (pre_overall + rec_overall)
    if logger is not None:
        logger.info('Validation: ')
        logger.info('Anger: Precision: {}\tRecall: {}\tF1: {}'.format(precision[3], recall[3], f1[3]))
        logger.info('Joy: Precision: {}\tRecall: {}\tF1: {}'.format(precision[1], recall[1], f1[1]))
        logger.info('Neutral: Precision: {}\tRecall: {}\tF1: {}'.format(precision[0], recall[0], f1[0]))
        logger.info('Sadness: Precision: {}\tRecall: {}\tF1: {}'.format(precision[2], recall[2], f1[2]))
        logger.info('Overall: Precision: {}\tRecall: {}\tF1: {}'.format(pre_overall, rec_overall, f1_overall))
    else:
        print('Anger: Precision: {}\tRecall: {}\tF1: {}'.format(precision[3], recall[3], f1[3]))
        print('Joy: Precision: {}\tRecall: {}\tF1: {}'.format(precision[1], recall[1], f1[1]))
        print('Neutral: Precision: {}\tRecall: {}\tF1: {}'.format(precision[0], recall[0], f1[0]))
        print('Sadness: Precision: {}\tRecall: {}\tF1: {}'.format(precision[2], recall[2], f1[2]))
        print('Overall: Precision: {}\tRecall: {}\tF1: {}'.format(pre_overall, rec_overall, f1_overall))

    return f1_overall


if __name__ == '__main__':
    dataset_val = Data.EmotionDataset('val')
    dataset_val.build()
    batch_num_val = dataset_val.get_batch_num(1)
    print(batch_num_val)

    model = Model.IDEAModel()
    model = model.to(Config.args.device)
    model.load_state_dict(torch.load('/home/ramon/wy_uci/torch/model/15_19.pth'))

    batch_generator = Data.generate_batches(dataset=dataset_val, batch_size=1, device=Config.args.device)
    f1 = evaluation(model, batch_generator, None)
