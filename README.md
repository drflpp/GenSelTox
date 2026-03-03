**Guidelines**

This repository contains all the necessary files for screening selectevely antimicrobial inorganic NPs. The repository includes folders for `LoadingGenomicData`, `Data and Preprocessing`, `ModelBuilding`, `ModelOptimization`, `GeneticAlgorithm` and `ReinforcementLearning`.

**Loading Genomic Data**

Folder contains a file for fetching genomical datasets used: a functional orthologs dataset and metabolical pathways dataset.

**Data and Preprocessing**

Folder contains preprocessed data with genomic descriptors: one dataset with merged with functional orthologs, and one merged with metabological pathways.

**ML Model Building**

Folder contains a file with necessary code for feature selection based on statistical approach (correlation and varience threshold), feature importance and permutation importance as well as model training code.

**ML Model Optimization:**  

Top models obtained from model selection were optimized by hyperparameter tuning to identify the best parameters. CatBoost and XGB regressor models showed the best performance after optimization and were used for predicting MC of NPs. Folder contains python scripts used for optimization of mentioned models. 

**Genetic Algorithm**

To identify selectevely antimicrobial NPs, the script for the genetic algorithm is stored in the `Genetic Algorithm` folder. It consists of subfolders `Catboost` and `XGBoost`, each containing necessary datasets and Python scripts for unique compound generation, crossover and mutation, and a main script (`ga_main.py`) for evolution up to user-defined population size and generation number. The optimized models were used to predict antimicrobial activity in these generated unique compounds. The best NP combinations were selected by identifying compounds with the highest fitness scores, and an example of screening candidates is stored in the `output` folder.
