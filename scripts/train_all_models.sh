modelpath="models_real"
device=cuda:0
for layers in 2 6; do
for model in distilbert bert gpt2; do
runid=0
export PYTHONPATH="."; python3 experiment_utils/train_models.py --layers $layers --model $model --dataset yelp --epochs 5 --log_attn_steps 200 --lr 1e-5 --runid $runid --pretrained --device $device --model_path $modelpath
export PYTHONPATH="."; python3 experiment_utils/train_models.py --layers $layers --model $model --dataset imdb --epochs 5 --log_attn_steps 200 --lr 1e-5 --runid $runid --pretrained --device $device --model_path $modelpath
export PYTHONPATH="."; python3 experiment_utils/train_models.py --layers $layers --model $model --dataset imdb --epochs 5 --log_attn_steps 200 --lr 1e-5 --runid $runid --device $device --model_path $modelpath
export PYTHONPATH="."; python3 experiment_utils/train_models.py --layers $layers --model $model --dataset yelp --epochs 5 --log_attn_steps 200 --lr 1e-5 --runid $runid --device $device --model_path $modelpath
done
done
