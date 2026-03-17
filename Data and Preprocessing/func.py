import re
import time
import random
import pickle
import datetime

import numpy as np
import pandas as pd
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

import joblib
import optuna

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
plt.rcParams['figure.dpi'] = 100

results_df = pd.DataFrame(columns=[
    'dataset_name', 'dataset_type', 'model_name', 'data_shape', 'r2_score', 'rmse', 'top_features'
])

cv_logs_df = pd.DataFrame(columns=[
    "timestamp", "dataset_name", "dataset_type", "model_name", "n_splits", "stratified",
    "hyperparameters", "train_R2", "train_RMSE", "val_R2", "val_RMSE",
    "test_R2", "test_RMSE", "train_MAE", "val_MAE", "test_MAE",
])

PLOTLY_PURPLE_SEQ = ["#2a0a3d", "#3f116f", "#5621a8", "#7b3fe4", "#ab63fa"]


def clean_feature_names(feature_names):
    clean_names = []
    for name in feature_names:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        clean_names.append(clean_name)
    return clean_names


def get_model_name(model):
    return model.__class__.__name__


def log_result(dataset_name, dataset_type, model, data_shape, r2, rmse, feature_names=None):
    global results_df
    model_name = get_model_name(model)
    top_features = None
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:10]
        top_features = [feature_names[i] for i in indices]
    new_row = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': model_name,
        'data_shape': f"{data_shape[0]}x{data_shape[1]}",
        'r2_score': r2,
        'rmse': rmse,
        'top_features': top_features
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


def df_fit_transformer(df: pl.DataFrame):
    oe_dict = {}
    df_copy = df.clone()
    cat_cols = df_copy.select(pl.col(pl.String)).columns
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        col_data = df_copy.select(col).to_numpy()
        oe.fit(col_data)
        transformed = oe.transform(col_data)
        df_copy = df_copy.with_columns(pl.Series(name=col, values=transformed.flatten()))
        oe_dict[col] = oe
    num_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    num_cols = df_copy.select(pl.col(num_types)).columns
    scaler = StandardScaler()
    num_data = df_copy.select(num_cols).to_numpy()
    scaler.fit(num_data)
    scaled = scaler.transform(num_data)
    scaled_df = pl.DataFrame(scaled, schema=num_cols)
    df_copy = df_copy.drop(num_cols).hstack(scaled_df)
    return df_copy, oe_dict, scaler


def reduce_dimensionality_fast(df: pl.DataFrame, var_thresh=1e-5, corr_thresh=0.95):
    num_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    num_cols = df.select(pl.col(num_types)).columns
    if not num_cols:
        return df
    num_df = df.select(num_cols)
    num_np = num_df.to_numpy()
    selector = VarianceThreshold(threshold=var_thresh)
    num_np_var = selector.fit_transform(num_np)
    support = selector.get_support()
    selected_cols_var = [num_cols[i] for i in range(len(num_cols)) if support[i]]
    num_df = pl.DataFrame(num_np_var, schema=selected_cols_var)
    corr_matrix = np.corrcoef(num_np_var, rowvar=False)
    to_drop = set()
    for i in range(len(selected_cols_var)):
        if selected_cols_var[i] in to_drop:
            continue
        for j in range(i + 1, len(selected_cols_var)):
            if selected_cols_var[j] in to_drop:
                continue
            if abs(corr_matrix[i, j]) > corr_thresh:
                to_drop.add(selected_cols_var[j])
    reduced_df = num_df.drop(list(to_drop))
    non_num_cols = [c for c in df.columns if c not in num_cols]
    non_num_df = df.select(non_num_cols)
    final_df = reduced_df.hstack(non_num_df)
    return final_df


def filter_cat(df, col, perc):
    df = df.filter(pl.col(col)
                   .is_in(df.group_by(col).count()
                          .filter(pl.col("count") >= df.height * perc).select(col).to_series()))
    return df


def my_preprocessing(df):
    filtered_df1 = df.filter(pl.col("MIC_NP (µg/mL)") < 5000)
    filtered_df1 = filtered_df1.filter(pl.col("MIC_NP (µg/mL)") > 0)
    filtered_df1 = filtered_df1.filter(pl.col('np_size_min (nm)') < 150)
    filtered_df1 = filtered_df1.filter(pl.col('np_size_avg (nm)') < 200)
    filtered_df1 = filtered_df1.filter(pl.col('np_size_max (nm)') < 300)
    filtered_df1 = filtered_df1.filter(pl.col('min_Incub_period, h') < 100)
    filtered_df1 = filtered_df1.filter(pl.col('avg_Incub_period, h') < 150)
    filtered_df1 = filtered_df1.filter(pl.col('max_Incub_period, h') < 200)
    filtered_df1 = filtered_df1.filter(pl.col('time_set (hours)') < 60)
    filtered_df1 = filtered_df1.filter(pl.col('growth_temp, C ') > 25)
    print(f'Before preprocessing: {df.shape} \nAfter preprocessing numerical columns:{filtered_df1.shape}')
    filtered_df1 = filter_cat(filtered_df1, 'shape', 0.005)
    filtered_df1 = filter_cat(filtered_df1, 'np', 0.005)
    filtered_df1 = filter_cat(filtered_df1, 'method', 0.005)
    filtered_df1 = filtered_df1.with_columns(pl.col('MIC_NP (µg/mL)').log().alias('MIC_NP (µg/mL)'))
    filtered_df1 = filtered_df1.filter(pl.col("MIC_NP (µg/mL)") > -4)
    print(f'After preprocessing categorical columns too {filtered_df1.shape}')
    return filtered_df1


def preprocess_features(df, target_col, cat_cols=None, scaler=None, encoders=None, fit=True):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values
    if cat_cols is None:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if fit:
        encoders = {}
        for col in cat_cols:
            categories = sorted(X[col].astype(str).unique())
            oe = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1)
            X[col] = oe.fit_transform(X[[col]].astype(str)).ravel()
            encoders[col] = oe
    else:
        for col in cat_cols:
            X[col] = encoders[col].transform(X[[col]].astype(str)).ravel()
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])
    return X, y, scaler, encoders, cat_cols


class BasePipeline:
    def __init__(self, df, dataset_name, models, cutoff_FI=0.95, cutoff_PI=0.95, strat=None,
                 strat_feature=None, cv_type='skfold', n_splits=5):
        self.df = df.clone()
        self.dataset_name = dataset_name
        self.models = models
        self.cutoff_FI = cutoff_FI
        self.cutoff_PI = cutoff_PI
        self.strat = strat
        self.random_state = 42
        self.strat_feature = strat_feature
        self.cv_type = cv_type
        self.n_splits = n_splits
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.df_cleaned = None
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.encoders = None
        self.scaler = None

    def clean_data(self, cols=None):
        df = self.df
        if cols is not None:
            df = df.drop(cols)
        df = df.unique(maintain_order=True)
        self.df_cleaned = df
        print(f"Cleaned data shape: {df.shape}")

    def reduce_dimensions(self):
        reduced_df = reduce_dimensionality_fast(self.df_cleaned.clone())
        print(f"Reduced dimensions: {reduced_df.shape}")
        return reduced_df

    def preprocess(self, df):
        df.columns = clean_feature_names(df.columns)
        X = df.drop('MIC_NP___g_mL_')
        Y = df.select('MIC_NP___g_mL_').to_series()
        X_transformed, encoders, scaler = df_fit_transformer(X)
        self.encoders = encoders
        self.scaler = scaler
        return X_transformed.to_pandas(), Y.to_pandas()

    def split_data(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )
        print("Data split complete.")

    def run_model_pipeline(self):
        for model in self.models:
            print(f"Running model: {get_model_name(model)}")
            self.train_and_log(model)

    def train_and_log(self, model):
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        full = pd.concat([self.X_train, self.X_test])
        feature_names = full.columns
        log_result(self.dataset_name, 'preprocessed', model, full.shape, r2, rmse, feature_names)
        print(f"Initial Model {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        self.plot_top_features(model, self.X_train.columns)

    def plot_top_features(self, model, feature_names, top_n=20):
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names, scores = zip(*top_features)
        plt.figure(figsize=(10, 6))
        plt.barh(names, scores, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title(f'{get_model_name(model)} Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()

    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        X, Y = self.preprocess(reduced_df)
        self.split_data(X, Y)
        self.run_model_pipeline()
        print(f"Pipeline completed in {(time.time() - start):.2f} seconds.")


class LazyPreprocess(BasePipeline):
    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        X, Y = self.preprocess(reduced_df)
        print(f"Pipeline completed in {(time.time() - start):.2f} seconds.")
        return X, Y


class PipelineGetDF(BasePipeline):
    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        return reduced_df


class PipelineFIPI(BasePipeline):

    def run_model_pipeline(self):
        for model in self.models:
            print(f"Running model: {get_model_name(model)}")
            self.train_and_log(model)
            self.feature_importance_selection(model)
            self.permutation_importance_selection(model)

    def feature_importance_selection(self, model):
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            print(f"Model {get_model_name(model)} has no feature_importances_. Skipping FI step.")
            return
        df_imp = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        df_imp['cumulative'] = df_imp['importance'].cumsum() / df_imp['importance'].sum()
        selected = df_imp[df_imp['cumulative'] <= self.cutoff_FI]['feature'].tolist()
        if len(selected) < len(df_imp):
            selected.append(df_imp.iloc[len(selected)]['feature'])
        self.X_train = self.X_train[selected]
        self.X_test = self.X_test[selected]
        full = pd.concat([self.X_train, self.X_test])
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        print(f"Initial Model {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        log_result(self.dataset_name, f'after keeping {self.cutoff_FI} FI',
                   model, full.shape, r2, rmse, self.X_train.columns)
        print(f"Feature Importance selection done. Remaining features: {len(selected)}")
        self.plot_top_features(model, self.X_train.columns)

    def permutation_importance_selection(self, model):
        print("Calculating permutation importance...")
        result = permutation_importance(
            model, self.X_test, self.Y_test, n_repeats=10, random_state=self.random_state
        )
        df_perm = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance_mean': result.importances_mean
        }).sort_values(by='importance_mean', ascending=False)
        df_perm = df_perm[df_perm['importance_mean'] > 0].dropna()
        df_perm['cumulative_importance'] = (
            df_perm['importance_mean'].cumsum() / df_perm['importance_mean'].sum()
        )
        selected = df_perm[df_perm['cumulative_importance'] <= self.cutoff_PI]['feature'].tolist()
        if len(selected) < len(df_perm):
            selected.append(df_perm.iloc[len(selected)]['feature'])
        self.X_train = self.X_train[selected]
        self.X_test = self.X_test[selected]
        full = pd.concat([self.X_train, self.X_test])
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        print(f"Initial Model {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        log_result(self.dataset_name, f'after keeping {self.cutoff_FI} FI + {self.cutoff_PI} PI',
                   model, full.shape, r2, rmse, self.X_train.columns)
        print(f"Permutation Importance selection done. Remaining features: {len(selected)}")
        self.plot_top_features(model, self.X_train.columns)

    def run(self):
        start = time.time()
        print("Cleaning data...")
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        print("Preprocessing...")
        X, Y = self.preprocess(reduced_df)
        self.split_data(X, Y)
        print("Running full model pipeline (FI + PI)...")
        self.run_model_pipeline()
        final_df = pd.concat([
            pd.concat([self.X_train, self.Y_train], axis=1),
            pd.concat([self.X_test, self.Y_test], axis=1)
        ], axis=0).reset_index(drop=True)
        print(f"Pipeline completed in {(time.time() - start):.2f} seconds.")
        print(f"Final DF shape after PI: {final_df.shape}")
        return final_df


class TrainablePipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.feature_names = None
        self.cat_cols = None
        self.num_cols = None

    def _fit_transform_X(self, X):
        X = X.to_pandas().copy()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.encoders = {}
        for col in self.cat_cols:
            categories = sorted(X[col].astype(str).unique())
            oe = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1)
            X[col] = oe.fit_transform(X[[col]].astype(str)).ravel()
            self.encoders[col] = oe
        self.scaler = StandardScaler()
        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
        return X

    def _transform_X(self, X):
        X = X.copy()
        for col in self.cat_cols:
            X[col] = self.encoders[col].transform(X[[col]].astype(str)).ravel()
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X

    def fit(self, model):
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        reduced_df.columns = clean_feature_names(reduced_df.columns)
        X = reduced_df.drop('MIC_NP___g_mL_')
        y = reduced_df['MIC_NP___g_mL_'].to_pandas().values
        X = self._fit_transform_X(X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.feature_names = self.X_train.columns.tolist()
        self.model = model
        self.model.fit(self.X_train, self.Y_train)
        self.feature_names = self.X_train.columns.tolist()
        preds = self.model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        print(f"[FIT] {get_model_name(model)} | R2={r2:.4f} | RMSE={rmse:.4f}")
        return r2, rmse

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded")
        df = df.clone()
        df.columns = clean_feature_names(df.columns)
        if 'MIC_NP___g_mL_' in df.columns:
            X = df.to_pandas().drop('MIC_NP___g_mL_', axis=1)
        else:
            X = df.to_pandas()
        X = self._transform_X(X)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.astype(np.float32)
        return self.model.predict(X)

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "encoders": self.encoders,
            "scaler": self.scaler,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "feature_names": self.feature_names,
            "dataset_name": self.dataset_name,
        }, path)
        print(f"[SAVED] {path}")

    def load(self, path):
        payload = joblib.load(path)
        self.model = payload["model"]
        self.encoders = payload["encoders"]
        self.scaler = payload["scaler"]
        self.cat_cols = payload["cat_cols"]
        self.num_cols = payload["num_cols"]
        self.feature_names = payload["feature_names"]
        self.dataset_name = payload["dataset_name"]
        print(f"[LOADED] {path}")


def optimization_catboost(df_x_scaled, df_y, dataset_name='dataset', n_trials=70, results_file='best_params_catboost.py'):
    import os
    from datetime import datetime

    def objective(trial, df_x, df_y):
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        params = {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 0.5, 3.0),
            "rsm": trial.suggest_float("rsm", 0.6, 1.0),
            "bootstrap_type": bootstrap_type,
            "grow_policy": trial.suggest_categorical("grow_policy", ["Depthwise", "Lossguide"]),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": 0,
            "random_state": 42
        }
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        model = CatBoostRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_indices, test_indices in cv.split(df_x, df_y):
            x_train, x_test = df_x.iloc[train_indices], df_x.iloc[test_indices]
            y_train, y_test = df_y.iloc[train_indices], df_y.iloc[test_indices]
            model.fit(x_train, y_train)
            cv_scores.append(r2_score(y_test, model.predict(x_test)))
        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_x_scaled, df_y), n_trials=n_trials)
    best_params = study.best_params
    print(f"Best R2: {study.best_value:.5f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("# CatBoost best hyperparameter results\n\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    var_name = f"best_params_{dataset_name}_{timestamp}"
    with open(results_file, "a") as f:
        f.write(f"{var_name} = {best_params}\n\n")
    return best_params


def optimization_xgboost(df_x, df_y, dataset_name='dataset', n_trials=100, results_file='best_params_xgboost.py'):
    import os
    from datetime import datetime

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 0.3, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'tree_method': 'hist',
            'grow_policy': 'depthwise',
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': 1
        }
        model = XGBRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(df_x):
            X_train, X_test = df_x.iloc[train_idx], df_x.iloc[test_idx]
            y_train, y_test = df_y.iloc[train_idx], df_y.iloc[test_idx]
            model.fit(X_train, y_train)
            scores.append(r2_score(y_test, model.predict(X_test)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best R2: {study.best_value:.5f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("# XGBoost best hyperparameter results\n\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    var_name = f"best_params_{dataset_name}_{timestamp}"
    with open(results_file, "a") as f:
        f.write(f"{var_name} = {best_params}\n\n")
    return best_params
