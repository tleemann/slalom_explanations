#!/bin/bash
device="cuda"
epochs=5
savepath=models_lin_bin_rerun
for model in bert distilbert gpt2
do
for layers in 1 2 4 8 12
do
export PYTHONPATH="."; python3 slalom_explanations/train_models.py --epochs $epochs --model $model --device $device --n_heads 12 --batch_size 20 --layers $layers --dataset sparsebow --model_path $savepath
done
done
echo DONE.
