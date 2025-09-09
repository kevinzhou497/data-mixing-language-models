# data-mixing-language-models

This repository contains the code for the "Optimal Data Mixing Strategies For Language Model Pre-Training" MSc project. 

The train_gpt.py is the main training script which does the training and validation for our experiments. 
The results_viz.ipynb notebook is used heavily for the results and evaluation sections of the dissertation, including the several plots and regression predictions. 
The other Python files are primarily used for data retrieval and pre-processing of the data corpora from Hugging Face, as well as carrying out the subsampling of documents required for the Repeat-Aware experiments. 

The .sh shell files are used to run the different Python files using the Imperial HPC resources.  
