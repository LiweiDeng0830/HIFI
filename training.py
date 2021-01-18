from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import os
from testing import get_scores, get_metrics
from pprint import pprint

from pytorchtools import EarlyStopping


def test_epoch(model, test_dataloader, config):
    model.eval()

    start = time.time()
    test_score, test_label = get_scores(model, test_dataloader, config)
    best_valid_metrics = get_metrics(test_score, test_label, config.bf_search_min, config.bf_search_max,
                                     config.bf_search_step_size, config.display_freq)
    print("testing time cost: ", time.time() - start)
    print('=' * 30 + 'result' + '=' * 30)
    pprint(best_valid_metrics)


def eval_epoch(model, eval_dataloader):

    model.eval()
    desc = '  - (validation)   '
    total_loss = 0.0
    batch_num = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = batch
            trg_seq = src_seq.detach().clone()

            # forward
            # KL : batch_size, sequence_length
            reconstruct_input, KL = model(src_seq, trg_seq)

            # reconstruct_loss : batch_size, sequence_length, hidden_size
            reconstruct_loss = cal_performance(
                input=src_seq,
                reconstruct_input=reconstruct_input
            )
            # reconstruct_loss: batch_size, sequence_length
            reconstruct_loss = reconstruct_loss.sum(-1)

            loss = reconstruct_loss
            loss = loss.mean()

            # note keeping
            total_loss += loss.item()
            batch_num += 1

    return total_loss/batch_num

def cal_performance(input, reconstruct_input):
    return F.mse_loss(input, reconstruct_input, reduction="none")

def train_epoch(model, train_dataloader, optimizer, KL_weight):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0.0
    total_reconstruct_loss = 0.0
    total_KL_loss = 0.0
    batch_num = 0

    desc = '  - (Training)   '
    for batch in tqdm(train_dataloader, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = batch
        trg_seq = src_seq.detach().clone()

        # forward
        optimizer.zero_grad()
        # KL : batch_size, sequence_length
        reconstruct_input, KL_loss = model(src_seq, trg_seq)

        # reconstruct_loss : batch_size, sequence_length, hidden_size
        reconstruct_loss = cal_performance(
            input=src_seq,
            reconstruct_input=reconstruct_input
        )
        # reconstruct_loss: batch_siz, sequence_length
        reconstruct_loss = reconstruct_loss.sum(-1)

        # In the begining of training, KL_weight is small which will make model easier to reconstruct.
        # With the increase of training epoch, KL_weight will increase which will prevent model from overfitting.
        # loss : batch_size, sequence_length
        loss = reconstruct_loss + KL_weight * KL_loss
        #loss = reconstruct_loss
        loss = loss.mean()

        # note keeping
        total_loss += loss.item()
        total_reconstruct_loss += reconstruct_loss.mean().item()
        total_KL_loss += KL_loss.mean().item()

        # backward and update parameters
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        batch_num += 1

    return total_loss/batch_num, total_KL_loss/batch_num, total_reconstruct_loss/batch_num


def train(model, train_dataloader, valid_dataloader, optimizer, config, test_dataloader=None):
    ''' Start training '''

    save_path = config.save_mode+ "_dmodel_" + str(config.d_model) + "_nlayer_" + str(config.n_layers) + "_dk_"+ str(config.d_k)+ "_dinner_"+ str(config.d_inner) + '.chkpt'
    save_path = os.path.join(config.model_save_path, save_path)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True, save_path=save_path)
    KL_weight = config.kl_start
    if config.kl_warmup > 0:
        anneal_rate = (1.0 - config.kl_start) / (config.kl_warmup * len(train_dataloader.dataset)/config.batch_size)
    else:
        anneal_rate = 0

    valid_losses = []
    for epoch_i in range(config.epoch):
        print('[ Epoch', epoch_i, ']')

        KL_weight = min(1.0, KL_weight+anneal_rate)

        start = time.time()
        train_loss, train_KL_loss, train_reconstruct_loss = train_epoch(
            model, train_dataloader, optimizer, KL_weight)
        print("train loss ", train_loss,
              "train KL loss", train_KL_loss,
              "train reconstruct loss", train_reconstruct_loss,
              "training time cost: ", time.time()-start)

        start = time.time()
        valid_loss = eval_epoch(model, valid_dataloader)
        print("valid loss ", valid_loss,
              "validation time cost: ", time.time()-start)

        if test_dataloader is not None:
            test_epoch(model, test_dataloader, config)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': config, 'model': model.state_dict()}

        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)
        early_stopping(valid_loss, checkpoint)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
