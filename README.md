# GenSelTox: Genome-Informed AI-Driven Discovery and Experimental Validation of Inorganic Materials with Selective Antibacterial Action

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)

## Abstract

The rational design of nanoparticles (NPs) with selective antimicrobial activity remains a major challenge in nanomedicine, particularly for combating antimicrobial resistance while preserving beneficial microbiota. In this work, we present an AI-driven discovery platform that integrates predictive modeling and generative design to prioritize selectively antibacterial inorganic NPs, which are subsequently validated experimentally. The platform combines a curated dataset of 2098 antibacterial activity measurements with enriched descriptors, including physicochemical properties, synthesis parameters, and Kyoto Encyclopedia of Genes and Genomes (KEGG) derived functional orthologs and metabolic pathways. Gradient boosting models trained on this gene informed dataset achieved high predictive performance, especially XGB regression acheived cross-validated R² score of 0.79, 0.83 and 0.80 for predicting minimal concentration (MC) with experimental parameters, functional ortholog features and metabolic pathways. Next, GenSelTox couples these models with a generative framework, integrating genetic algorithms, and reinforcement learning to explore NP formulation space for optimal selective toxicity. As proof-of-concept, one of top-ranked ZnO NP was synthesized and tested against pathogenic (Staphylococcus aureus, Pseudomonas aeruginosa) and non-pathogenic (Bacillus subtilis, Escherichia coli) strains. Experimental results confirmed species-selective inhibition consistent with model predictions, and interpretable features revealed key genomic drivers of susceptibility, including oxidative stress and DNA repair pathways. By uniting AI with mechanistic insight and empirical validation, GenSelTox advances the field toward precision nanotherapeutics. The platform offers a scalable, interpretable, and generalizable framework for selective nanoparticle design, with broad potential for adaptation to antifungal, antiviral, and anticancer applications. 
![GenSelTox Platform Overview](Figure.jpg)

---

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Guidelines](#guidelines)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Generative Design](#generative-design)
- [Experimental Validation](#experimental-validation)
- [Model Performance Summary](#model-performance-summary)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Key Features](#key-features)

---

## Installation

The platform requires Python 3.8 or higher. We recommend using a virtual environment to manage dependencies. Begin by cloning the repository and navigating to the project directory:

```bash
git clone https://github.com/yourusername/GenSelTox.git
cd GenSelTox
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

The core dependencies include pandas (≥1.5.0), numpy (≥1.23.0), polars (≥0.19.0), scikit-learn (≥1.2.0), xgboost (≥2.0.0), catboost (≥1.2.0), lightgbm (≥4.0.0), optuna (≥3.0.0), shap (≥0.42.0), matplotlib (≥3.6.0), seaborn (≥0.12.0), joblib (≥1.2.0), stable-baselines3 (≥2.0.0), gymnasium (≥0.28.0), and pyyaml (≥6.0). For GPU-accelerated training, ensure you have a CUDA-compatible GPU with appropriate drivers installed.

---

## Datasets

The platform operates on three primary dataset configurations. The base dataset `MIC_df_preprocessed.csv` contains core antibacterial activity measurements including nanoparticle properties such as size distribution (minimum, average, maximum), morphological characteristics (shape), surface modifications (coating), and synthesis parameters (method, temperature, duration, solvent). This dataset also includes experimental conditions, bacterial strain identifiers, and corresponding MIC values measured in µg/mL.

Two enriched versions extend this foundation with genomic information. The dataset `MIC_df_ko.csv` incorporates 11,273 binary features representing KEGG functional orthologs (K numbers), which encode the presence or absence of specific enzymatic functions and molecular processes in each bacterial strain. The dataset `MIC_df_path.csv` similarly enriches the base data with binary features corresponding to KEGG metabolic pathways, capturing higher-level functional organization of bacterial metabolism and cellular processes.

These genomic features are acquired through the KEGG REST API. Navigate to the data preprocessing directory and execute the provided Jupyter notebook:

```bash
cd "Data and Preprocessing"
jupyter notebook GET_KEGG_ko_path.ipynb
```

This notebook queries the KEGG database for bacterial genome annotations, retrieves functional ortholog assignments and pathway memberships for each strain in the dataset, and generates binary feature matrices that are subsequently merged with the experimental measurements. The resulting files `kegg_ko_matrix.csv` and `kegg_pathway_matrix.csv` serve as the genomic annotation layers for model training.

---

## Guidelines

### Data Preprocessing

To prepare the raw experimental data for modeling, execute the preprocessing script located in the data processing directory. This can be accomplished by running:

```bash
cd "Data and Preprocessing"
python preprocessing.py
```

The preprocessing pipeline performs several critical transformations on the raw measurements. Outlier removal is applied to MIC values (restricting the range to 0-5000 µg/mL), nanoparticle size parameters (filtering particles with average diameter below 200 nm and maximum diameter below 300 nm), and incubation conditions (limiting incubation periods to physiologically relevant ranges). Missing values in categorical features such as synthesis method and particle shape are imputed using modal values from the distribution, while numerical features including synthesis temperature and duration are imputed using mean values. Categorical features with insufficient representation (appearing in fewer than 0.5% of observations) are filtered to ensure statistical reliability. Finally, MIC values undergo logarithmic transformation to normalize their distribution and improve model performance.

After preprocessing the base experimental data, merge the genomic annotations by executing:

```bash
python merging.py
```

This script integrates the KEGG-derived functional ortholog and pathway matrices with the cleaned experimental measurements, producing the three final datasets: `MIC_df_preprocessed.csv` containing only experimental features, `MIC_df_ko.csv` augmented with 11,273 functional ortholog features, and `MIC_df_path.csv` augmented with metabolic pathway annotations. These datasets represent different feature spaces for subsequent model training and evaluation.

### Model Training

Training predictive models on the prepared datasets can be performed using the standardized pipeline infrastructure. The following example demonstrates training on the functional ortholog-enriched dataset:

```python
import polars as pl
from MLModelBuilding.BasePipeline import PipelineFIPI
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

df = pl.read_csv("MIC_df_ko.csv")

models = [
    XGBRegressor(random_state=42),
    CatBoostRegressor(random_state=42, verbose=False),
    LGBMRegressor(random_state=42, verbose=0),
    RandomForestRegressor(random_state=42)
]

pipeline = PipelineFIPI(
    df=df,
    dataset_name="MIC_KO_Dataset",
    models=models,
    cutoff_FI=0.95,
    cutoff_PI=0.99
)

final_df = pipeline.run()
```

This pipeline automatically handles data cleaning, dimensionality reduction through variance and correlation thresholds, ordinal encoding of categorical variables, standard scaling of numerical features, and train-test splitting with a fixed random seed for reproducibility. The `PipelineFIPI` class implements sequential feature selection using both feature importance and permutation importance. The `cutoff_FI` parameter specifies the cumulative importance threshold for retaining features based on model-intrinsic importance scores (0.95 retains features accounting for 95% of cumulative importance), while `cutoff_PI` applies the same logic to permutation-based importance estimates. Feature importance visualizations are automatically generated and saved during the pipeline execution.

For more granular control over the training process, the `TrainablePipeline1` class enables custom model training, evaluation, and persistence:

```python
from MLModelBuilding.TrainablePipeline import TrainablePipeline1

pipeline = TrainablePipeline1(
    df=df,
    dataset_name="MIC_Dataset",
    models=[]
)

model = CatBoostRegressor(
    iterations=1400,
    learning_rate=0.093,
    depth=4,
    random_state=42,
    verbose=False
)

r2, rmse = pipeline.fit(model)
pipeline.plot_top_features_gradient1(top_n=40, fname="feature_importance.png")
pipeline.save("trained_model.joblib")
```

The fitted model along with its associated preprocessing transformations (encoders, scalers) is serialized to disk for subsequent use in generative design or prediction on new formulations.

### Hyperparameter Optimization

Optimal hyperparameters for XGBoost models are identified through Bayesian optimization using Optuna. Navigate to the optimization directory and execute:

```bash
cd MLModelOptimization
python optimize_xgb.py
```

This script performs 100 trials of hyperparameter search across a predefined space including tree architecture parameters (maximum depth from 3 to 8, minimum child weight from 1.0 to 10.0), learning dynamics (learning rate from 0.03 to 0.2 on a logarithmic scale, number of estimators from 300 to 1000), regularization terms (L1 alpha from 1e-4 to 0.3, L2 lambda from 0.5 to 3.0), and sampling strategies (subsample ratio from 0.6 to 0.9, column sampling from 0.5 to 0.8). Each trial is evaluated using 5-fold cross-validation with R² as the objective metric. The search employs tree-structured Parzen estimators to efficiently explore the hyperparameter landscape. Best parameters from multiple optimization runs are automatically saved to `best_params_2.py` with unique timestamps for version tracking.

Similarly, CatBoost hyperparameters are optimized through:

```bash
python optimize_catboost.py
```

The CatBoost optimization space includes iteration count (300 to 2000), learning rate (0.01 to 0.3 logarithmically scaled), tree depth (4 to 10 levels), L2 regularization strength (1 to 10), random strength for split selection (0.5 to 3.0), and bootstrap strategy (Bayesian or Bernoulli). When Bayesian bootstrap is selected, the bagging temperature parameter is sampled from 0 to 1; when Bernoulli bootstrap is chosen, the subsample ratio is sampled from 0.6 to 1.0. Results are stored in `best_params_cat.py` with corresponding R² performance metrics. Multiple sequential runs are recommended to ensure convergence to global optima rather than local minima.

### Generative Design

The platform implements two complementary approaches for discovering novel nanoparticle formulations with enhanced selective toxicity. The genetic algorithm framework evolves candidate formulations through iterative selection, crossover, and mutation operations guided by predicted antimicrobial selectivity.

To execute genetic algorithm-based discovery using the CatBoost predictor, navigate to the appropriate directory and run:

```bash
cd GeneticAlgorithm/Catboost
python ga_main.py
```

For discovery using XGBoost predictions:

```bash
cd GeneticAlgorithm/XGBoost
python ga_main.py
```

Prior to execution, configure the target bacterial strains in `ga_compd_generation.py` by setting the pathogenic and beneficial bacteria identifiers. For example:

```python
name_of_pathogenic_bacteria = 'Escherichia coli ATCC 25922'
name_of_good_bacteria = 'Pseudomonas aeruginosa nan'
```

The genetic algorithm maintains a population of 100 candidate formulations per generation and evolves them over 100 generations using a mutation rate of 0.2 and crossover rate of 0.2. The crossover operator treats certain feature groups as linked units: nanoparticle size parameters (minimum, average, maximum) are transferred together to maintain realistic size distributions, synthesis-related parameters (method, precursor, capping agent) are co-inherited to preserve chemically feasible combinations, while independent features undergo individual crossover events with the specified probability. Fitness is evaluated as the difference in predicted log(MIC) between the beneficial and pathogenic strains, with higher values indicating greater selectivity favoring beneficial bacteria survival.

Output files are organized by generation in the `output/` directory, with each generation's population saved as a CSV file containing all candidate parameters and their predicted antimicrobial activities against both target strains. Summary statistics tracking mean and maximum fitness across generations are saved for convergence analysis.

The reinforcement learning approach provides an alternative generative strategy using policy gradient methods. Execute the RL-based discovery through:

```bash
cd ReinforcementLearning
python rl_en.py
```

The RL environment models nanoparticle design as a sequential decision process where the agent selects formulation parameters (particle size distribution, synthesis method, incubation conditions) to maximize the selectivity reward defined as the predicted MIC difference between target strains. The state space comprises all formulation parameters plus genomic features of the target bacteria, while the action space covers continuous ranges for size parameters and discrete choices for categorical variables like synthesis method. The default implementation uses Proximal Policy Optimization (PPO) with a learning rate of 1e-4, trained over 5120 timesteps with periodic checkpointing every 200 episodes. Alternative algorithms including Soft Actor-Critic (SAC) and Deep Deterministic Policy Gradient (DDPG) can be configured by modifying the model initialization in the script. Generated candidates and their predicted activities are continuously saved to `result.csv` for post-analysis and ranking.

---

## Experimental Validation

As proof-of-concept, we synthesized and experimentally validated a top-ranked ZnO nanoparticle formulation predicted to exhibit selective toxicity. The candidate was tested against two pathogenic strains (*Staphylococcus aureus*, *Pseudomonas aeruginosa*) and two non-pathogenic reference strains (*Bacillus subtilis*, *Escherichia coli* ATCC 25922). Experimental MIC measurements confirmed species-selective inhibition patterns consistent with computational predictions. Detailed synthesis protocols, characterization data (transmission electron microscopy, dynamic light scattering, zeta potential measurements), and antimicrobial susceptibility testing procedures are provided in the supplementary materials of the associated publication. Interpretability analysis using SHAP values identified key genomic features driving differential susceptibility including genes involved in oxidative stress response (catalase K00549) and DNA repair pathways (RecN K03111), offering mechanistic insights into the selective antimicrobial action.

---

## Model Performance Summary

Cross-validated performance metrics across different feature sets demonstrate the value of genomic enrichment for predictive accuracy:

| Dataset | Model | Train R² | Val R² | Test R² | Test RMSE | Test MAE |
|---------|-------|----------|---------|---------|-----------|----------|
| Experimental Parameters | XGBoost | 0.92 | 0.81 | 0.79 | 0.87 | 0.65 |
| Functional Orthologs | XGBoost | 0.95 | 0.85 | 0.83 | 0.78 | 0.58 |
| Metabolic Pathways | XGBoost | 0.94 | 0.82 | 0.80 | 0.84 | 0.62 |
| Functional Orthologs | CatBoost | 0.93 | 0.84 | 0.81 | 0.82 | 0.60 |

The functional ortholog-enriched dataset yielded the highest test set performance (R² = 0.83, RMSE = 0.78), indicating that fine-grained functional annotations provide superior predictive power compared to higher-level pathway groupings or experimental parameters alone.

---

## Reproducibility

All computational results reported in the manuscript can be reproduced using the provided code and datasets. Random number generation is controlled through fixed seeds (RANDOM_STATE = 42) set consistently across all modules including NumPy random operations, Python's native random library, and scikit-learn's random state parameters. The reported results were obtained on hardware comprising an Intel Xeon E5-2680 v4 processor (2.40 GHz), NVIDIA Tesla V100 GPU (32 GB memory), 128 GB DDR4 RAM, running Ubuntu 20.04 LTS. Approximate execution times are: data preprocessing (10 minutes), single model training (30-60 minutes), hyperparameter optimization via Optuna (4-8 hours for 100 trials), genetic algorithm execution (2-4 hours for 100 generations with population size 100), and reinforcement learning training (6-12 hours for 5000+ timesteps). Variance in runtime depends primarily on dataset size, feature dimensionality, and hardware specifications.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{GenSelTox2024,
  title={AI-Driven Discovery of Selective Antimicrobial Nanoparticles: 
         Integrating Predictive Modeling, Generative Design, and Experimental Validation},
  author={[Authors]},
  journal={[Journal Name]},
  year={2024},
  volume={XX},
  pages={XXX-XXX},
  doi={10.xxxx/xxxxxx}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for full terms. Note that the KEGG database has separate usage restrictions; please consult the [KEGG License](https://www.kegg.jp/kegg/legal.html) for details on commercial applications.

---

## Contact

**Corresponding Author:** [Name]  
**Email:** [email@institution.edu]  
**Lab Website:** [https://lab-website.edu]

For bug reports, feature requests, usage questions, and methodological discussions, please use the [GitHub Issues](https://github.com/yourusername/GenSelTox/issues) page. Contributions are welcome; please refer to CONTRIBUTING.md for submission guidelines.

---

## Key Features

✅ Comprehensive antibacterial activity dataset (2,098 curated measurements) with genomic enrichment via KEGG functional orthologs and metabolic pathways

✅ Multi-algorithm machine learning framework supporting XGBoost, CatBoost, LightGBM, and Random Forest implementations

✅ Automated feature selection pipeline incorporating correlation analysis, variance thresholding, model-intrinsic feature importance, and permutation importance

✅ Hyperparameter optimization infrastructure using Bayesian optimization (Optuna) for top-performing models

✅ Dual generative design approaches: genetic algorithms and reinforcement learning for nanoparticle discovery

✅ Model interpretability through SHAP values and feature importance analysis revealing mechanistic drivers of antimicrobial selectivity

✅ Experimental validation demonstrating successful prediction-guided synthesis and testing of selective ZnO nanoparticles

---

**Last Updated:** March 2026  
**Version:** 1.0.0
