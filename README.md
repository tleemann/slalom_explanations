# Code for submission "Attention Mechanisms Don’t Learn Additive Models: Rethinking Feature Importance For Transformers"

## Setup
First setup a new envirionment by installing the dependencies listed in the file and activate it.

```
conda env create -f code/environment.yml
conda activate transformers
```
You will additionally need to install the shap package by running
```
conda install -c conda-forge shap
```

## Important files

The important code files for this project are located in the folder ```slalom_explanations```.
The experiments in Sections 6.1. / 6.2. are organized as Notebooks in the folder ```notebooks```. 

Follow these intructions to reproduce experiments:

Here is an overview over the most prominent ones:
- ```train_models.py``` The main training script to train the transformer models used in this work.
- ```A_Motivation.ipynb```: The Recovery example with Shapley values in Appendix A.
- ```B_LearningLinearModels.ipynb```: Code for Figure 3 showing that transformers do not learn linear models. Train models using ```scripts/train_models_linear.sh```
- ```C_Explaining_SLALOM_synth.ipynb```: Code to explain the outputs of the transformers trained on the linear model with SLALOM (Figure 4ab)
- ```D_ComputeGroundTruthOffline``` and ```eval_faithfulness.py```: Compute faithfulness Metrics as Deletion/Insertion Scores (Table 1c, Appendix) as well as correlation with linear scores.
    - First run the notebook to compute importances offline
    - Train models using ```scripts/train_all_models.sh```
    - then use the faithfulness script as outlined in ```scripts/test_all_models.sh```
    - ```F_Tables.ipynb``` to collect results from log-files for naive bayes and Human-Attention ROC
    - ```G_FidelityMetrics.ipynb```to print tables with results for Fidelity metrics (Insertion/removal, multiremoval in Figure 5)
- ```E_RealWorldPlots.ipynb```: Create quantitative results for IMDB datasets in Figure 6
- ```E_RealWorldPlotsHAT.ipynb```: Create quantitative results for YELP-HAT (Appendix, Figure 11)
the models were stored.

Our implementation of the Local SLALOM fitting can be found in ```slalom_explanations/slalom_helpers.py``` and the implementations of other methods are in ```slalom_explanations/attribution_methods.py```