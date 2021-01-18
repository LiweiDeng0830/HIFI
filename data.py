from torch.utils.data import DataLoader
from dataset import SMDDatasetTest, SMDDataset
import torch


def get_data(datastring, filepath, max_length, step, labelfilepath, testfilepath,
             valid_portation=0.3, shuffle=True, data_collator=None,
             batch_size=64, device=torch.device("cpu")):

    if datastring == "SMD":
        train_set = SMDDataset(max_length=max_length,
                               step=step,
                               filepath=filepath,
                               device=device)
        valid_examples = train_set.get_valid_examples(valid_portation=valid_portation, shuffle=shuffle)
        valid_set = SMDDataset(max_length=None,
                               step=None,
                               examples=valid_examples,
                               device=device)
        test_set = SMDDatasetTest(max_length=max_length,
                                  step=step,
                                  filepath=testfilepath,
                                  labelfilepath=labelfilepath,
                                  device=device)
    else:
        raise NotImplementedError

    return {
        "train_dataloader": DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=data_collator),
        "valid_dataloader": DataLoader(dataset=valid_set,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=data_collator),
        "test_dataloader": DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      collate_fn=data_collator)
    }
