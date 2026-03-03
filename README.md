![](/Figure.png)

**Guidelines**

This repository contains all the necessary files for screening selectevely antimicrobial inorganic NPs. The repository includes folders for `LoadingGenomicData`, `Data and Preprocessing`, `ModelBuilding`, `ModelOptimization`, `GeneticAlgorithm` and `ReinforcementLearning`.

**Loading Genomic Data**

Contains scripts for fetching genomic datasets from KEGG: a functional orthologs dataset and a metabolic pathways dataset.

**Data and Preprocessing**

Contains preprocessed datasets with genomic descriptors: one merged with functional orthologs and one merged with metabolic pathways.

**ML Model Building**

Contains code for feature selection based on correlation and variance thresholds, feature importance analysis, permutation importance, and model training.

**ML Model Optimization:**  

Top-performing models from the selection stage were further optimized via hyperparameter tuning. CatBoost and XGBoost regressors achieved the best performance after optimization and were used for predicting the minimal inhibitory concentration (MC) of NPs. This folder contains Python scripts for optimization of both models.

**Genetic Algorithm**

Contains scripts for generating selectively antimicrobial NP candidates using a genetic algorithm. The folder is organized into CatBoost and XGBoost subfolders, each containing the required datasets and Python scripts for unique compound generation, crossover, and mutation. The main script (ga_main.py) runs the evolutionary optimization for a user-defined population size and number of generations. Optimized models are used to predict antimicrobial activity of generated candidates. Top candidates selected by highest fitness score are saved to the output folder.
