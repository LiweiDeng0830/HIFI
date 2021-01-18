train_dir="./data/ServerMachineDataset/train/"
test_dir="./data/ServerMachineDataset/test/"
test_label_dir="./data/ServerMachineDataset/test_label/"
d_word_vec=64
d_model=64
d_inner=128
n_layers=2
n_head=8
d_k=16
d_v=16
dropout=0.2
epoch=200
batch_size=64
n_warmup_steps=500
lr=0.005
model_save_path_="./models/"
test_score_label_save_="./results/SMD_dmodel_64_dinner_128_nlayers_2_dk_16_lr_0.005_k_2/"
device="cuda:1"
ad_size=38

#files=$(ls ${train_dir})
files=("machine-1-1.txt")

for file in $files
do
    train_file=${train_dir}${file}
    test_file=${test_dir}${file}
    test_label_file=${test_label_dir}${file}
    model_save_path=${model_save_path_}${file}
    test_score_label_save=${test_score_label_save_}${file}
    echo $train_file
    echo $test_file
    echo $test_label_file
    python main.py --filepath=$train_file \
        --labelfilepath=$test_label_file \
        --testfilepath=$test_file \
        --d_word_vec=$d_word_vec \
        --d_model=$d_model \
        --d_inner=$d_inner \
        --n_layers=$n_layers \
        --n_head=$n_head \
        --d_k=$d_k \
        --d_v=$d_v \
        --dropout=$dropout \
        --epoch=$epoch \
        --n_warmup_steps=$n_warmup_steps \
        --lr=$lr \
        --save_mode=$save_mode \
        --model_save_path=$model_save_path \
        --test_score_label_save=$test_score_label_save \
        --batch_size=$batch_size \
        --device=$device \
        --ad_size=$ad_size
    echo ""
done
