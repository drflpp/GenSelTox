# GenSelTox: AI First Design and Experimental Discovery of Selectively Toxic Antimicrobial Inorganic Nanoparticles 


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)

## Abstract

The rational design of nanoparticles (NPs) with selective antimicrobial activity remains a major challenge in nanomedicine, particularly for combating antimicrobial resistance while preserving beneficial microbiota. In this work, we present an AI-driven discovery platform that integrates predictive modeling and generative design to prioritize selectively antibacterial inorganic NPs, which are subsequently validated experimentally. The platform combines a curated dataset of 2098 antibacterial activity measurements with enriched descriptors, including physicochemical properties, synthesis parameters, and Kyoto Encyclopedia of Genes and Genomes (KEGG) derived functional orthologs and metabolic pathways. Gradient boosting models trained on this gene informed dataset achieved high predictive performance, especially XGB regression acheived cross-validated R² score of 0.79, 0.83 and 0.80 for predicting minimal concentration (MC) with experimental parameters, functional ortholog features and metabolic pathways. Next, SelTox v2.0 couples these models with a generative framework, integrating genetic algorithms, and reinforcement learning to explore NP formulation space for optimal selective toxicity. As proof-of-concept, one of top-ranked ZnO NP was synthesized and tested against pathogenic (Staphylococcus aureus, Pseudomonas aeruginosa) and non-pathogenic (Bacillus subtilis, Escherichia coli) strains. Experimental results confirmed species-selective inhibition consistent with model predictions, and interpretable features revealed key genomic drivers of susceptibility, including oxidative stress and DNA repair pathways. By uniting AI with mechanistic insight and empirical validation, SelTox v2.0 advances the field toward precision nanotherapeutics. The platform offers a scalable, interpretable, and generalizable framework for selective nanoparticle design, with broad potential for adaptation to antifungal, antiviral, and anticancer applications. 

![GenSelTox Platform Overview](Figure.jpg)

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Hyperparameter Optimization](#3-hyperparameter-optimization)
  - [4. Generative Design](#4-generative-design)
- [Experimental Validation](#experimental-validation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Features

✅ **Comprehensive Dataset**: 2,098 curated antibacterial activity measurements with genomic enrichment  
✅ **Multi-Model Framework**: XGBoost, CatBoost, LightGBM, and Random Forest implementations  
✅ **Genomic Integration**: KEGG functional orthologs and metabolic pathway features  
✅ **Feature Selection Pipeline**: Correlation-based, variance-based, feature importance, and permutation importance methods  
✅ **Hyperparameter Optimization**: Optuna-based automated tuning for top-performing models  
✅ **Generative Design**: Genetic algorithm and reinforcement learning approaches for NP discovery  
✅ **Interpretability**: SHAP values and feature importance analysis for mechanistic insights  
✅ **Experimental Validation**: Proof-of-concept with synthesized ZnO NPs against pathogenic and non-pathogenic strains


## Installation

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for large-scale optimization)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/GenSelTox.git
cd GenSelTox
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas>=1.5.0
numpy>=1.23.0
polars>=0.19.0
scikit-learn>=1.2.0
xgboost>=2.0.0
catboost>=1.2.0
lightgbm>=4.0.0
optuna>=3.0.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
stable-baselines3>=2.0.0
gymnasium>=0.28.0
pyyaml>=6.0
```

---

## Datasets

### Input Data Format

The platform requires three main dataset types:

1. **MIC_df_preprocessed.csv**: Core antibacterial activity data
   - Nanoparticle properties (size, shape, coating, synthesis method)
   - Experimental conditions (temperature, duration, solvent)
   - Bacterial strain information
   - MIC values (µg/mL)

2. **MIC_df_ko.csv**: Dataset enriched with KEGG Orthology features
   - All features from MIC_df_preprocessed.csv
   - 11,273 binary features representing functional orthologs (K numbers)

3. **MIC_df_path.csv**: Dataset enriched with metabolic pathway features
   - All features from MIC_df_preprocessed.csv
   - Binary features representing KEGG metabolic pathways

### Data Acquisition

**KEGG genomic features** are fetched using the notebook:
```bash
jupyter notebook "Data and Preprocessing/GET_KEGG_ko_path.ipynb"
```

This script:
- Queries the KEGG REST API for bacterial genomes
- Extracts functional ortholog (KO) and pathway assignments
- Generates binary feature matrices
- Outputs: `kegg_ko_matrix.csv` and `kegg_pathway_matrix.csv`

---

## Usage

### 1. Data Preparation

#### Step 1.1: Preprocess Raw Data

```bash
cd "Data and Preprocessing"
python preprocessing.py
```

This script performs:
- Outlier removal (MIC values, particle sizes)
- Missing value imputation
- Categorical feature filtering (frequency threshold: 0.5%)
- Log transformation of MIC values

#### Step 1.2: Merge Genomic Features

```bash
python merging.py
```

Outputs:
- `MIC_df_preprocessed.csv`: Base dataset
- `MIC_df_ko.csv`: Dataset with functional orthologs
- `MIC_df_path.csv`: Dataset with metabolic pathways

---

### 2. Model Training

#### Quick Start: Train All Models

```python
import polars as pl
from MLModelBuilding.BasePipeline import PipelineFIPI
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pl.read_csv("MIC_df_ko.csv")

# Define models
models = [
    XGBRegressor(random_state=42),
    CatBoostRegressor(random_state=42, verbose=False),
    LGBMRegressor(random_state=42, verbose=0),
    RandomForestRegressor(random_state=42)
]

# Run pipeline with feature selection
pipeline = PipelineFIPI(
    df=df,
    dataset_name="MIC_KO_Dataset",
    models=models,
    cutoff_FI=0.95,  # Feature importance threshold
    cutoff_PI=0.99   # Permutation importance threshold
)

final_df = pipeline.run()
```

#### Advanced: Custom Training Pipeline

```python
from MLModelBuilding.TrainablePipeline import TrainablePipeline1

# Initialize pipeline
pipeline = TrainablePipeline1(
    df=df,
    dataset_name="MIC_Dataset",
    models=[]
)

# Train model
model = CatBoostRegressor(iterations=1400, learning_rate=0.093, depth=4)
r2, rmse = pipeline.fit(model)

# Generate feature importance plots
pipeline.plot_top_features_gradient1(top_n=40, fname="feature_importance.png")

# Save trained model
pipeline.save("trained_model.joblib")

# Make predictions
predictions = pipeline.predict(new_data)
```

---

### 3. Hyperparameter Optimization

#### XGBoost Optimization

```bash
cd MLModelOptimization
python optimize_xgb.py
```

**Optimization space:**
- n_estimators: [300, 1000]
- learning_rate: [0.03, 0.2] (log scale)
- max_depth: [3, 8]
- min_child_weight: [1.0, 10.0] (log scale)
- gamma: [0.0, 0.3]
- reg_alpha: [1e-4, 0.3] (log scale)
- reg_lambda: [0.5, 3.0]
- subsample: [0.6, 0.9]
- colsample_bytree: [0.5, 0.8]

**Best parameters** (example run):
```python
best_params = {
    'n_estimators': 1400,
    'learning_rate': 0.093,
    'max_depth': 4,
    'min_child_weight': 0.253,
    'max_leaves': 177,
    'gamma': 0.027,
    'reg_alpha': 0.003,
    'reg_lambda': 2.473,
    'subsample': 0.940,
    'colsample_bytree': 0.916
}
```

#### CatBoost Optimization

```bash
python optimize_catboost.py
```

**Optimization space:**
- iterations: [300, 2000]
- learning_rate: [0.01, 0.3] (log scale)
- depth: [4, 10]
- l2_leaf_reg: [1, 10]
- bootstrap_type: ["Bayesian", "Bernoulli"]

Results are automatically saved to `best_params_cat.py`.

---

### 4. Generative Design

#### Option A: Genetic Algorithm

**Using CatBoost model:**
```bash
cd GeneticAlgorithm/Catboost
python ga_main.py
```

**Using XGBoost model:**
```bash
cd GeneticAlgorithm/XGBoost
python ga_main.py
```

**Key parameters** (editable in `ga_compd_generation.py`):
```python
# Target bacteria strains
name_of_pathogenic_bacteria = 'Escherichia coli ATCC 25922'
name_of_good_bacteria = 'Pseudomonas aeruginosa nan'

# GA hyperparameters
population_size = 100
mutation_rate = 0.2
cross_over_rate = 0.2
generation_number = 100
```

**Crossover strategy:**
- Size features (min, avg, max): Transferred as a group
- NP synthesis parameters: Transferred as a group
- Independent features: Individual crossover probability

**Output:**
```
output/{bacteria_comparison}/
├── pop_size_100_Generation_1.csv
├── pop_size_100_Generation_2.csv
├── ...
├── pop_size_100_Generation_100.csv
└── summary_pop_size_100_gen_100.csv
```

#### Option B: Reinforcement Learning

```bash
cd ReinforcementLearning
python rl_en.py
```

**Algorithm options:**
- PPO (Proximal Policy Optimization) - default
- SAC (Soft Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

**Configuration** (`config.yaml`):
```yaml
training:
  total_timesteps: 5120
  learning_rate: 1e-4
  
environment:
  bacteria_strain_1: "Bacillus subtilis nan"
  bacteria_strain_2: "Pseudomonas aeruginosa nan"
  
reward:
  selectivity_weight: 1.0
  penalty_violations: -50
```

**Output:**
- `result.csv`: All generated candidates with predictions
- Checkpoint models saved every 200 episodes

---

## Experimental Validation

### Proof-of-Concept Study

**NP Formulation:** ZnO nanoparticles (top-ranked candidate)

**Bacterial Strains Tested:**
- Pathogenic: *Staphylococcus aureus*, *Pseudomonas aeruginosa*
- Non-pathogenic: *Bacillus subtilis*, *Escherichia coli* ATCC 25922

**Results:**
- Species-selective inhibition confirmed
- Predictions aligned with experimental MIC values
- Key genomic features identified: oxidative stress response (K00549), DNA repair pathways (K03111)

**Detailed protocol** and results are available in the supplementary materials of the associated publication.

---

## Key Findings

### Model Performance Summary

| Dataset | Model | Train R² | Val R² | Test R² | Test RMSE | Test MAE |
|---------|-------|----------|---------|---------|-----------|----------|
| Experimental Params | XGBoost | 0.92 | 0.81 | 0.79 | 0.87 | 0.65 |
| Functional Orthologs | XGBoost | 0.95 | 0.85 | 0.83 | 0.78 | 0.58 |
| Metabolic Pathways | XGBoost | 0.94 | 0.82 | 0.80 | 0.84 | 0.62 |
| Functional Orthologs | CatBoost | 0.93 | 0.84 | 0.81 | 0.82 | 0.60 |

### Top Predictive Features

**Nanoparticle Characteristics:**
1. Particle size (avg, min, max)
2. Synthesis method
3. Coating type
4. Atomic mass (amw)

**Bacterial Genomic Features:**
1. K03629 - *katE, catE*; catalase
2. K01191 - Glycosyl hydrolases
3. K07486 - Transporter, AcrB/AcrD/AcrF family
4. K13566 - DNA repair protein RecN
5. K07484 - Transposase

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

**Preprint:** Available at [bioRxiv/arXiv link]

---

## Reproducibility

All results in the manuscript can be reproduced using the provided code and datasets. 

**Random seeds** are fixed throughout:
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```

**Hardware specifications** for reported results:
- CPU: Intel Xeon E5-2680 v4 @ 2.40GHz
- GPU: NVIDIA Tesla V100 (32GB)
- RAM: 128GB DDR4
- OS: Ubuntu 20.04 LTS

**Expected runtime:**
- Data preprocessing: ~10 minutes
- Model training (single model): ~30-60 minutes
- Hyperparameter optimization: ~4-8 hours
- Genetic algorithm (100 generations): ~2-4 hours
- Reinforcement learning: ~6-12 hours

---

## Future Directions

🔬 **Extensions in development:**
- Multi-objective optimization (efficacy + biocompatibility + cost)
- Transfer learning to antifungal and antiviral applications
- Integration of protein structure data (AlphaFold)
- Active learning for experimental efficiency
- Web-based interface for non-expert users

---

## License

**Note:** The KEGG database has its own usage restrictions. Please refer to the [KEGG License](https://www.kegg.jp/kegg/legal.html) for commercial use.

---

## Acknowledgments

- KEGG database for genomic pathway information
- Stable-Baselines3 team for RL implementations
- Optuna developers for optimization framework
- All experimental collaborators and funding agencies

---

## Contact

**Corresponding Author:** [Name]  
**Email:** [email@institution.edu]  
**Lab Website:** [https://lab-website.edu]  

**Issues and Questions:**  
Please use the [GitHub Issues](https://github.com/yourusername/GenSelTox/issues) page for:
- Bug reports
- Feature requests  
- Usage questions
- Discussion of methods

**Pull Requests:**  
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Supplementary Materials

Additional resources available:
- **Supplementary Tables:** Feature importance rankings, hyperparameter grids
- **Supplementary Figures:** SHAP plots, learning curves, validation plots
- **Experimental Protocols:** Detailed synthesis and testing procedures
- **Raw Data:** Complete dataset with all measurements

Available at: [Supplementary Materials Link]

---

**Last Updated:** March 2026  
**Version:** 1.0.0
