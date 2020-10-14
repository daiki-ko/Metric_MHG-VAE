from mhg.smi import HGGen, hg_to_mol
from mhg.hrg import HyperedgeReplacementGrammar as HRG
from mhg.tree_decomposition import (tree_decomposition,
                                                   tree_decomposition_with_hrg,
                                                   tree_decomposition_from_leaf,
                                                   topological_tree_decomposition,
                                                   molecular_tree_decomposition)
import os
import logging
import gzip
import pickle

logger = logging.getLogger('luigi-interface')
smiles_path = "datasets/qm9/qm9_smiles_shuffle.txt"

td_catalog = {
    'tree_decomposition': tree_decomposition,
    'tree_decomposition_with_hrg': tree_decomposition_with_hrg,
    'tree_decomposition_from_leaf': tree_decomposition_from_leaf,
    'topological_tree_decomposition': topological_tree_decomposition,
    'molecular_tree_decomposition': molecular_tree_decomposition
}

class DataPreprocessing():
   
    output_dir_path = "OUTPUT/data_prep_for_qm9"
    
    DataPreprocessing_params = {
    'kekulize': True, # Use kekulized representation or not
    'add_Hs': False, # Add hydrogens explicitly or not
    'all_single': True, # Represent every bond as a labeled single edge
    'tree_decomposition': 'molecular_tree_decomposition', # Tree decomposition algorithm
    'tree_decomposition_kwargs': {}, # Parameters for the tree decomposition algorithm
    'ignore_order': False # Ignore the orders of nodes in production rules
    }

    def requires(self):
        return []

    def run(self): 
        
        print("---------MHG preprocess----------")
        
        print("start mol to HG")
        
        hg_list = HGGen(smiles_path,
                        kekulize=True,
                        add_Hs=False,
                        all_single=True)
        
        print("finish mol to HG")
        
        print("----------------------------")
        
        print("start tree decomp and prod_rules")
        
        hrg = HRG(tree_decomposition=molecular_tree_decomposition,
                  ignore_order=False,
                  **self.DataPreprocessing_params['tree_decomposition_kwargs'])
        

        print("finish tree decomp and prod_rules ")
        
        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)
        
        print("----------------------------")
        
        print("start HRG")
        #prod_rule_seq_list = hrg.learn(hg_list, logger=logger.info, max_mol=self.DataPreprocessing_params.get('max_mol', np.inf))
        prod_rule_seq_list = hrg.learn(hg_list, logger=logger.info, max_mol=53019)
        print("finish HRG")
        
        print("----------------------------")
        
        #logger.info(" * the number of prod rules is {}".format(hrg.num_prod_rule))
        print(" * the number of prod rules is {}".format(hrg.num_prod_rule))

 
        #if self.DataPreprocessing_params.get('draw_prod_rule', False)
            
        if not os.path.exists(os.path.join(self.output_dir_path, 'prod_rules')):
            os.mkdir(os.path.join(self.output_dir_path, 'prod_rules'))
        for each_idx, each_prod_rule in enumerate(hrg.prod_rule_corpus.prod_rule_list):
            each_prod_rule.draw(os.path.join(self.output_dir_path, 'prod_rules', f'{each_idx}'))
            
        if not os.path.exists(os.path.join(self.output_dir_path, 'prod_rules')):
            os.mkdir(os.path.join(self.output_dir_path, 'prod_rules'))
            
        with gzip.open(os.path.join(self.output_dir_path, 'mhg_prod_rules.pklz'), "wb") as f:
            pickle.dump((hrg, prod_rule_seq_list), f)

    def load_output(self):
        with gzip.open(os.path.join(self.output_dir_path, 'mhg_prod_rules.pklz'), "rb") as f:
            hrg, prod_rule_seq_list = pickle.load(f)
        return hrg, prod_rule_seq_list

if __name__ == "__main__":
    hgg = DataPreprocessing()                   
                        
    hgg.run()
