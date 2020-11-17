# Embedding of Molecular Structure Using Molecular Hypergraph Variational Autoencoder with Metric Learning

Core code for the paper "Embedding of Molecular Structure Using Molecular Hypergraph Variational Autoencoder with Metric Learning" (https://doi.org/10.1002/minf.202000203)
 by Daiki Koge, Naoaki ONO, Ming Huang, Md. Altaf-Ul-Amin, Shigehiko Kanaya.


## Requirements

*PyTorch*  
We have updated the code such that it is using version 0.4.1.

*RDKit*  
version 2017.09.1.
We recommend installing rdkit through Anaconda, see e.g.
https://anaconda.org/rdkit/rdkit:  
`conda install -c rdkit rdkit`

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

using qm9 physical properties
`mol_smooth_embedding/Evaluate_Model.ipynb`

using rdkit descriptors
`mol_smooth_embedding/Evaluate_Model_RDkit_desc.ipynb`
