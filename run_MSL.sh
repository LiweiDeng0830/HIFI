train_dir="./data/MSL/train/"
test_dir="./data/MSL/test/"
test_label_dir="./data/MSL/test_label/"
d_word_vec=64
d_model=64
d_inner=256
n_layers=2
n_head=8
d_k=16
d_v=16
dropout=0.2
epoch=200
batch_size=64
n_warmup_steps=200
lr=0.005
gcn_k=2
model_save_path_="./models/"
test_score_label_save_="./results/MSL_dmodel_64_nlayer_2_dk_16_dinner_256_lr_0.005_k_2/"
device="cuda:3"
ad_size=55

#files=$(ls ${train_dir})
files=("T-8.txt")

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
        --ad_size=$ad_size \
        --gcn_k=$gcn_k
    echo ""
done
