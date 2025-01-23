#!/bin/bash
## Call using one argument to pass the number of runs or without arguments (one run only).
device="cuda"
epochs=5
savepath=models_lin_bin_rerun

# Default number of iterations to 1
iterations=1

# If an argument is provided, set iterations to the argument
if [ "$1" ]; then
  # Validate that the argument is a positive integer
  iterations=$1
fi


for ((runid = 0; runid < iterations; runid++)); do 
for model in bert distilbert gpt2
do
for layers in 1 2 4 8 12
do
export PYTHONPATH="."; python3 experiment_utils/train_models.py --epochs $epochs --model $model --device $device --n_heads 12 --batch_size 20 --layers $layers --dataset sparsebow --model_path $savepath --runid $runid
done
done
done
echo DONE.
