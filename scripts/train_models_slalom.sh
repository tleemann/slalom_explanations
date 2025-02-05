#!/bin/bash
epochs=20
savepath=models_slalom # /mnt/ssd3/tobias/
model=distilbert
device="cuda:1"
for vocabsize in 16 32 64 128 200
do
for layers in 2 # Can try others too: 1 2 4 6 8 12
do
currsp=${savepath}/v${vocabsize}
export PYTHONPATH="."; python3 experiment_utils/train_models.py --epochs $epochs --model $model --device $device --n_heads 12 --evalevery true --batch_size 20 --layers $layers --dataset slalom200 --model_path $currsp --vocab_size $vocabsize
done
done
echo DONE.
