from torch import optim
import torch
import os
from transformers import HfArgumentParser, set_seed
from pprint import pprint

from SMDconfig import Config
from data import get_data
from Models.Models import HIFI
from Models.Optim import ScheduledOptim
from training import train
from testing import test

def main():

    parser = HfArgumentParser((Config))
    config = parser.parse_args_into_dataclasses()[0]
    print(config)

    device = torch.device(config.device)

    # load data
    data = get_data(datastring=config.datastring,
                    filepath=config.filepath,
                    max_length=config.max_length,
                    step=config.step,
                    labelfilepath=config.labelfilepath,
                    testfilepath=config.testfilepath,
                    valid_portation=config.valid_portation,
                    shuffle=config.shuffle,
                    batch_size=config.batch_size,
                    device=device)

    model = HIFI(
        ad_size=config.ad_size,
        d_word_vec=config.d_word_vec,
        d_model=config.d_model,
        d_inner=config.d_inner,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout,
        n_position=config.n_position,
        sequence_length=config.max_length,
        gcn_layers=config.gcn_layers,
        gcn_alpha=config.gcn_alpha,
        k=config.gcn_k
    )
    model.to(device)
    print(HIFI)

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.lr, config.d_model, config.n_warmup_steps)

    test_dataloader = None
    if config.train_test:
        test_dataloader = data["test_dataloader"]
    train(model, data["train_dataloader"], data["valid_dataloader"], optimizer, config, test_dataloader=test_dataloader)

    model_name = config.save_mode+ "_dmodel_" + str(config.d_model) + "_nlayer_" + str(config.n_layers) + "_dk_"+ str(config.d_k)+ "_dinner_"+ str(config.d_inner) + '.chkpt'
    model_name = os.path.join(config.model_save_path, model_name)
    checkpoint = torch.load(model_name)
    print("load best epoch from ", model_name, " best epoch: ", checkpoint["epoch"])
    model.load_state_dict(checkpoint["model"])
    test(model, data["test_dataloader"], config)

if __name__=="__main__":
    main()
