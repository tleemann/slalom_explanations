device=cuda:3
for model in distilbert bert gpt2; do
for layers in 2 6; do
# Faithfulness, NaiveBayes, Multiremoval-Fidelity
export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 100 scripts/fidelity_config_imdb_test.json --run imdb_${model}_${layers}_r7 --model_type Transformer --device $device 
export PYTHONPATH="."; python3 scripts/faithfulness_eval.py 100 scripts/fidelity_config_imdb_test.json --run yelp_${model}_${layers}_pretrained_r7 --model_type Transformer --dataset yelp 
## Yelp Hat
export PYTHONPATH="."; python3 scripts/yelp_hat_LIME_SHAP.py 100 scripts/fidelity_config_yelp_human.json --device $device yelp_${model}_${layers}_pretrained_r7 --model_type Transformer
done
done

