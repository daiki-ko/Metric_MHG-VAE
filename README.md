# Embedding of Molecular Structure Using Molecular Hypergraph Variational Autoencoder with Metric Learning

Core code for the paper "Embedding of Molecular Structure Using Molecular Hypergraph Variational Autoencoder with Metric Learning" (https://doi.org/10.1002/minf.202000203)
 by Daiki Koge, Naoaki ONO, Ming Huang, Md. Altaf-Ul-Amin, Shigehiko Kanaya.


## Requirements

*PyTorch*  
We have updated the code such that it is using version 0.4.1.

*RDKit*  
version 2017.09.1.

*Python*  
 version 3.6.6.
 
*Jupyter*  
 version 1.0.0.


## Running the Code

### Prepare hypergraphs and descriptors

`mol_smooth_embedding/extract_mhg.py`

`mol_smooth_embedding/Prepare_RDKit_descriptors.ipynb`

### Training mode

`mol_smooth_embedding/Metric_MHG-VAE_with_DRL.ipynb`

### Evaluation of the models

using qm9 physical properties <br>
`mol_smooth_embedding/Evaluate_Model.ipynb`

using rdkit descriptors <br>
`mol_smooth_embedding/Evaluate_Model_RDkit_desc.ipynb`

## Abount Scripts

As mentioned in our paper, The VAE architecture uses the same model as kajino's MHG-VAE (http://proceedings.mlr.press/v97/kajino19a/kajino19a.pdf). 
The code for the MHG-VAE can be found in `mol_smooth_embedding/mhg`. And kajino's original code can be found in https://github.com/ibm-research-tokyo/graph_grammar.

