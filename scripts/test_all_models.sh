device=cuda:3
export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 200 scripts/fidelity_config_imdb_test.json --run imdb_gpt2_2_r7 --model_type Transformer --device $device
for model in distilbert bert gpt2; do
for layers in 1 6; do
export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 200 scripts/fidelity_config_imdb_test.json --run imdb_${model}_${layers}_r7 --model_type Transformer --device $device
#export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 200 scripts/fidelity_config_imdb_test.json --run imdb_${model}_${layers}_pretrained_r7 --model_type Transformer --device $device
#export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 200 scripts/fidelity_config_imdb_test.json --run yelp_${model}_${layers}_r7 --model_type Transformer --dataset yelp --device $device
#export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 40 scripts/fidelity_config_imdb_test.json --run yelp_${model}_${layers}_pretrained_r7 --model_type Transformer --dataset yelp --device $device
#export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 200 scripts/fidelity_config_imdb_test.json --run yelp_${model}_${layers}_r7 --model_type Transformer --dataset yelp --device $device
done
done

