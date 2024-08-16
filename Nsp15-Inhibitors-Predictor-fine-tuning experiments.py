import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from typing import List

# from rdkit import Chem
# from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from simpletransformers.classification import ClassificationModel, ClassificationArgs

# import MoleculeNet loaders from DeepChem
# from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21

# import MoleculeNet dataloder from bert-loves-chemistry fork
# %cd ./bert-loves-chemistry
# from chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop


# set the logging directories
project_name = "ChemBERTA_ict"
output_path = '/home/parsa/smiles_classification/LLM_Fine_balanced_Tuning_Molecular_Properties_output' 
model_name = 'model_balanced_v3_dataset2_w_14tests'

model_folder = os.path.join(output_path, model_name)

evaluation_folder = os.path.join(output_path, model_name + '_evaluation')
if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)

# set the parameters
EPOCHS = 200
BATCH_SIZE = 256
patience = 15
optimizer = "AdamW"
learning_rate = 5e-5
manual_seed = 110
print(model_folder)


train_df = pd.read_csv('/home/parsa/smiles_classification/data_training.csv')
valid_df = pd.read_csv('/home/parsa/smiles_classification/Test-1.csv').rename({'Results':'RESULT'},axis=1) #pd.read_csv('/home/parsa/smiles_classification/data_validation.csv')

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# configure Weights & Biases logging
wandb_kwargs = {'name' : model_name}

# configure training
classification_args = {'evaluate_each_epoch': True,
                       'evaluate_during_training_verbose': False,
                       'evaluate_during_training' : True,
                       'best_model_dir' : model_folder,
                       'no_save': False,
                       'save_eval_checkpoints': False,
                       'save_model_every_epoch': False,
                       'save_best_model' : True,
                       'save_steps': -1,
                    #    'num_train_epochs': EPOCHS,
                       'use_early_stopping': False,
                       'early_stopping_patience': 20,
                       'early_stopping_delta': 0.001,
                       'early_stopping_metrics': 'precision',
                       'early_stopping_metrics_minimize': True,
                       'early_stopping_consider_epochs' : True,
                       'fp16' : False,
                       'optimizer' : optimizer,
                       'adam_betas' : (0.95, 0.999),
                    #    'learning_rate' : 1e-6,
                       'manual_seed': manual_seed,
                       'train_batch_size' : 500,
                       'eval_batch_size' : 500,
                       'logging_steps' : len(train_df) / 500,
                       'auto_weights': True, # change to true
                       'wandb_project': project_name,
                       'overwrite_output_dir':True,
                       "use_multiprocessing": False,
                        'use_multiprocessing_for_evaluation': False,
                        'multiprocessing_chunksize':1,
                       'wandb_kwargs': wandb_kwargs}
import os, wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sweep_config = {
    "name": "vanilla-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "precision", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 5, "max": 100},
        "learning_rate": {"min": 1e-7, "max": 0.009},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 4,},
}
sweep_id = wandb.sweep(sweep_config, project="RTE - Hyperparameter Optimization3")
# model = ClassificationModel('roberta', 'DeepChem/ChemBERTa-77M-MLM', args=classification_args, cuda_device=1
#                             ,sweep_config=wandb.config,
#                             )


# Train the model
from sklearn.metrics import precision_score
def train():
    # Initialize a new wandb run
    wandb.init()
    model = ClassificationModel('roberta', 'DeepChem/ChemBERTa-77M-MLM', args=classification_args, cuda_device=0,sweep_config=wandb.config,)
    model.train_model(train_df, eval_df=valid_df, output_dir=model_folder,
                                precision=lambda truth, predictions: precision_score(
                truth, [round(p) for p in predictions]
            ),)
    wandb.join()
wandb.agent(sweep_id, train)