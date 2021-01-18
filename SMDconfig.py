from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:

    # device for running
    device: Optional[str] = "cuda:1"

    # parameters about dataset
    datastring: Optional[str] = "SMD"
    # dimension of multivariate
    ad_size: Optional[int] = 38
    # train data file path
    filepath: Optional[str] = "../BertAD/data/ServerMachineDataset/train/machine-1-3.txt"
    # windows_size
    max_length: Optional[int] = 100
    # for now, step must be one, which is the step of sliding window
    step: Optional[int] = 1
    # test label file path
    labelfilepath: Optional[str] = "../BertAD/data/ServerMachineDataset/test_label/machine-1-3.txt"
    # test data file path
    testfilepath: Optional[str] = "../BertAD/data/ServerMachineDataset/test/machine-1-3.txt"
    # validation portion from training dataset
    valid_portation: Optional[float] = 0.3
    # for training dataset, shuffle or not
    shuffle: Optional[bool] = True
    # training and test batch_size
    batch_size: Optional[int] = 64

    # parameters about model
    # d_1
    d_word_vec: Optional[int] = 128
    # d_1 : d_word_vec and d_model should be same
    d_model: Optional[int] = 128
    # d_4 for non_linear layer
    d_inner: Optional[int] = 512
    # number of attention-based temporal module
    n_layers: Optional[int] = 4
    # number of head in terms of attention layer
    n_head: Optional[int] = 8
    # Key dimension
    d_k: Optional[int] = 32
    # Value dimension
    d_v: Optional[int] = 32
    # dropout for preventing overfitting
    dropout: Optional[float] = 0.2
    # positional encoding, which should be same with max_length
    n_position: Optional[int] = 100

    # parameters for multivariate feature interaction module
    # number of layer of GCN
    gcn_layers = 2
    # alpha retains the original feature at every convolution
    gcn_alpha = 0.2
    # topk to convert the dense similairty to sparse interaction graph
    gcn_k = 2

    # Parameters for training
    # max epochs
    epoch: Optional[int] = 50
    # number of warmup steps
    n_warmup_steps: Optional[int] = 1000
    # KL weight warmup steps
    kl_warmup: Optional[int] = 10
    # start value of KL weight
    kl_start: Optional[int] = 0.5
    # learning rate
    lr: Optional[float] = 0.005
    train_test: Optional[bool] = False

    # Parameters for model save
    # model save path
    model_save_path: Optional[str] = "./models/"
    # patience in early stop
    patience: Optional[int] = 10

    # Parameters for test results save
    save_mode: Optional[str] = "best"
    # results output dir
    test_score_label_save: Optional[str] = "./results/SMD/"
    # search for best F1
    bf_search_min: Optional[int] = 0
    bf_search_max: Optional[int] = 30
    bf_search_step_size: Optional[float] = 0.01
    display_freq: Optional[int] = 200
