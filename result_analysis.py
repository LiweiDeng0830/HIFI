import json
import os
from dataclasses import dataclass
from transformers import HfArgumentParser
import numpy as np
from pprint import pprint


@dataclass
class Config:
    #root_dir: str = "./results/SMAP_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005/"
    #root_dir: str = "./results/SMAP_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005/"
    #root_dir: str = "./results/SMD_dmodel_64_dinner_128_nlayers_5_dk_16_lr_0.005/"
    #root_dir: str = "./results/SMAP_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005"
    #root_dir: str = "./results/SMAP_dmodel_64_dinner_128_nlayers_1_dk_16_lr_0.005"
    #root_dir: str = "./results/MSL_dmodel_64_nlayer_2_dk_16_dinner_256_lr_0.005/"
    #root_dir: str = "./results/SMAP_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005/"
    #root_dir: str = "./results/SMD_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005_without_compress/"
    #root_dir: str = "./results/MSL_dmodel_64_nlayer_2_dk_16_dinner_256_lr_0.005_back/"
    #root_dir: str = "./results/MSL_dmodel_64_nlayer_2_dk_16_dinner_256_lr_0.005_k_2/"
    #root_dir: str = "./results/MSL_dmodel_64_nlayer_2_dk_16_dinner_256_lr_0.005_k_2_noseed/"
    root_dir: str = "./results/SMD_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005_k_2/"
    filename: str = "result.json"


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

if __name__=="__main__":

    parser = HfArgumentParser((Config))
    config = parser.parse_args_into_dataclasses()[0]

    sub_dir = os.listdir(config.root_dir)
    print("total files: {}".format(len(sub_dir)))

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for dir in sub_dir:
        result_file = os.path.join(config.root_dir, dir, config.filename)
        result_dict = load_json(result_file)

        TP += result_dict["TP"]
        TN += result_dict["TN"]
        FN += result_dict["FN"]
        FP += result_dict["FP"]

    print("TP", TP)
    print("TN", TN)
    print("FN", FN)
    print("FP", FP)

    print("-"*40)
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    f1 = recall*precision*2/(recall + precision)
    print("recall = ", recall)
    print("precision = ", precision)
    print("f1 = ", f1)
