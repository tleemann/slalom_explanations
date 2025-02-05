<p align="center"><img width="700" height="200" src="https://raw.githubusercontent.com/tleemann/slalom_explanations/main/SLALOMCrop.png"></p>


# SLALOM: High-Fidelity Token-Level Explanations for Transformers

[![pytest](https://github.com/tleemann/slalom_explanations/actions/workflows/python-test.yml/badge.svg?branch=refactoring)](https://github.com/tleemann/slalom_explanations/actions/workflows/python-test.yml)

## Quickstart

A lightweight version of SLALOM can be installed using PIP:

```
pip install slalom-explanations
```
**Note:** The PIP installer contains a minimal version that allows only for computing explanations. To reproduce experiments, follow the extented setup with Conda described below.

## Basic usage

SLALOM contains an easy-to-use interface. The main commands to compute SLALOM explanations for Huggingface transformer models (SequenceClassification) are as follows:

```python
from slalom_explanations import SLALOMLocalExplanantions 
from slalom_explanations import slalom_scatter_plot

# Initialize explainer with an initialized SequenceClassification model and corresponding tokenizer
slalom_explainer = SLALOMLocalExplanantions(model, tokenizer, modes=["value", "imp"])

# Compute SLALOM explanation
example_text = "This was an amazing movie!"
res_explanation = slalom_explainer.tokenize_and_explain(example_text)

# Scatter plot
slalom_scatter_plot(res_explanation, sizey=8, sizex=8)

```


We provide the notebook ```notebooks/0_Quickstart.ipynb```, which fully runs all the steps with a pretrained model from the Huggingface hub.

## What is special about SLALOM?

<p><img align="right" width="362" height="567" src="https://raw.githubusercontent.com/tleemann/slalom_explanations/main/SLALOM2.PNG"></p>

* SLALOM is a surrogate model explanation method that is **specifically designed for the transformer architecture.** It uses a surrogate model class that is specifically designed to model the non-linearies in attention-based models, resulting in high-fidelity explanations.
* The **explanation can be visualized in a 2D-plane** that contains a dot for each token in an input sequence. One axis describes the token *value* (its impact on the classification on its own) and the other describes the token *importance* (its interaction weight when seen in combination with other tokens).

This repository accompanies the TMLR Paper

[Attention Mechanisms Don’t Learn Additive Models: Rethinking Feature Importance For Transformers](https://openreview.net/forum?id=yawWz4qWkF) 

by Tobias Leemann, Alina Fastowski, Felix Pfeiffer, and Gjergji Kasneci. The technical details of the method are described in the paper.

<br>
<br>
<br>
<br>
<br>
<br>

## Installing the full repository
To reproduce the experiments (including the competing explanation methods) you need to install additional dependencies. These cannot be installed via pip but we provide a environment file to set up a dedicated conda environment.
First setup a new environment by installing the dependencies listed in the file and activate it.

```
conda env create -f environment.yml
conda activate slalom
```
You will additionally need to install the shap package by running
```
conda install -c conda-forge shap
```

Add the corresponding kernel to your existing JupyterLab installation by executing:
```
python -m ipykernel install --user --name slalom
```

## Important files

The important code files for this project are located in the folder ```slalom_explanations```.
The experiments in Sections 6.1. / 6.2. are organized as Jupyter Notebooks in the folder ```notebooks```. 

Follow the intructions in the notebooks to reproduce experiments.
Here is an overview over the provided notebooks and where to find each individual experiment.

- ```A_MotivationalExperiments.ipynb```: The Recovery example with Shapley values in Appendix A.
- ```B_LearningLinearModels.ipynb```: Code for Figure 3 showing that transformers do not learn linear models. Train models using ```scripts/train_models_linear.sh```
- ```C_Explaining_SLALOM_synth.ipynb```: Code to explain the outputs of the transformers trained on the linear model with SLALOM as well as training models on a SLALOM distribution and verifying recovery (Figure 4ab, Figure 4cd)
- ```D_ComputeGroundTruthOffline``` Compute attribution scores for linear Naive Bayes models for different tokenizers offline. These are require to compute correlation with as well as correlation with linear scores (used in Table 1a)
- ```E_RealWorldPlots.ipynb```: Create the qualitative plots in Figure 
- ```F_Tables.ipynb``` to collect results from log-files for naive bayes and Human-Attention ROC and runtimes (Table 1a, 1b, Table 2)
- ```G_FidelityMetrics.ipynb```to print tables with results for Fidelity metrics (Insertion/removal Table 1c, multiremoval in Figure 6)
- ```H_SLALOM_OpenAI.ipynb```: Use SLALOM to explain an OpenAI model (Appendix F.7)
- ```I_CaseStudy.ipynb```: A case study for how SLALOM can be used to identify vulnerability and spurious correlations and create adversarial examples (Appendix F.8)

## Reference

Please cite our paper if you find the work or the code provided here helpful, e.g., using the following BibTeX entry:

```
@article{leemann2025attention,
    title={Attention Mechanisms Don{\textquoteright}t Learn Additive Models: Rethinking Feature Importance for Transformers},
    author={Tobias Leemann and Alina Fastowski and Felix Pfeiffer and Gjergji Kasneci},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=yawWz4qWkF},
}
```
