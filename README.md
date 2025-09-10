# data-mixing-language-models

This repository contains the code for the "Optimal Data Mixing Strategies For Language Model Pre-Training" MSc project. 

The train_gpt.py file is the main training script which does the training and validation for our experiments. 
The results_viz.ipynb notebook is used heavily for the results and evaluation sections of the dissertation, including the several plots and regression predictions. 
The other Python files are primarily used for data retrieval and pre-processing of the data corpora from Hugging Face, with a subsample factor used to carry out any potential subsampling of documents required for the Repeat-Aware experiments. Specifically, repeat_aware_docs.py is used for FineWeb and WikiText, and repeat_aware_documents_pubmed.py is used for PubMed. Importantly, for the full sample of data from these sources, we still use these repeat_aware files but with a subsample factor of 1. 

Due to the incredibly large amount of data available in the PubMed corpus, we end up needing to use create_subset.py to subsample the PubMed data processed from repeat_aware_documents_pubmed.py with a subsample factor of 1 into a smaller set of documents with a relatively similar total token count to the WikiText data we use. Then, create_subsamples_subset.py is used to create the subsamples from this smaller set for the Repeat-Aware experiments using PubMed.

The .sh shell files are used to run the different Python files with the Imperial HPC resources.  
