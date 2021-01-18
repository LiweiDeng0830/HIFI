# HIFI: Unsupervised Anomaly Detection for Multivariate Time Series with High-order Feature Interactions

## Datasets Discription
In the paper, three datasets **MSL**, **SMAP** and **SMD** are used. Due to the memery limit of Github, we only put some samples in <kbd>/data/ServiceMachineDataset</kbd> which are a subset of **SMD**.  

The whole three datasets can be found in https://github.com/NetManAIOps/OmniAnomaly.

## Usage
To run on the samples, the following command can be used.

```
sh run_SMD.sh
```

To reproduce the results stated in the paper, you put the whole dataset in <kbd>/data/</kbd> and set the correct data path in bash scripts, i.e. run_SMD.sh, run_MSL.sh or run_SMAP.sh. 

For example, if you want to run on **SMD** dataset, you can use the following steps.

```
Step1: Put the whole dataset in /data/ which can be download from OmniAnomaly rep.
Step2: Change the datapath in run_SMD.sh if they are not right.
Step3: run the bash script as shown in before.
```
