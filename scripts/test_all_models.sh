for model in distilbert bert roberta gpt2; do
for layers in 2 6; do
export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 40 scripts/fidelity_config_imdb_test.json --filter_length 100 --run imdb_${model}_${layers}_pretrained_r0 --model_type Transformer
done
done
