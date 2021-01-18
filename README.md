# HIFI: Unsupervised Anomaly Detection for Multivariate Time Series with High-order Feature Interactions

## Datasets Discription
In the paper, three datasets **MSL**, **SMAP** and **SMD** are used. Due to the memery limit of Github, we only put some samples in <kbd>/data/ServiceMachineDataset</kbd> which are a subset of **SMD**.  

The whole three datasets can be found in https://github.com/NetManAIOps/OmniAnomaly.

## Usage
To run on the samples, the following command can be used.

```
sh run_SMD.sh
```

To reproduce the results stated in the paper, you put the dataset in <kbd>/data/</kbd> and set the correct data path in bash scripts, i.e. run_SMD.sh, run_MSL.sh or run_SMAP.sh. 
