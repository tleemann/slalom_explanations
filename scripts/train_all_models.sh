for layers in 2 6; do
for model in distilbert bert gpt2; do # bert roberta gpt2 bert gpt2
for runid in 7; do
#export PYTHONPATH="."; python3 slalom_explanations/train_models.py --layers $layers --model $model --dataset yelp --epochs 2 --log_attn_steps 200 --lr 1e-5 --runid $runid --device cuda:0
export PYTHONPATH="."; python3 slalom_explanations/train_models.py --layers $layers --model $model --dataset yelp --epochs 2 --log_attn_steps 200 --lr 1e-5 --runid $runid --device cuda:0 --pretrained
#export PYTHONPATH="."; python3 slalom_explanations/train_models.py --layers $layers --model $model --dataset imdb --epochs 2 --log_attn_steps 200 --lr 1e-5 --runid $runid --device cuda:0
export PYTHONPATH="."; python3 slalom_explanations/train_models.py --layers $layers --model $model --dataset imdb --epochs 2 --log_attn_steps 200 --lr 1e-5 --runid $runid --device cuda:0 --pretrained
done
done
done
