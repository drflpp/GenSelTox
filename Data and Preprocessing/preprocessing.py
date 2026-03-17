import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def fill_na_mode(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mode()[0])
    return df


def fill_na_mean(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mean())
    return df


def fill_size_missing_values(df, avg_col='np_size_avg (nm)', max_col='np_size_max (nm)', min_col='np_size_min (nm)'):
    result_df = df.copy()
    avg_mask = result_df[avg_col].isna()
    result_df.loc[avg_mask, avg_col] = (
        result_df.loc[avg_mask, max_col] + result_df.loc[avg_mask, min_col]
    ) / 2
    max_mask = result_df[max_col].isna()
    result_df.loc[max_mask, max_col] = result_df.loc[max_mask, avg_col]
    min_mask = result_df[min_col].isna()
    result_df.loc[min_mask, min_col] = result_df.loc[min_mask, avg_col]
    return result_df


def detect_outliers_iqr(df, numerical_columns=None):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = {}
    for col in numerical_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_report[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': outliers[col].tolist()
            }
    return outlier_report


def detect_outliers_zscore(df, numerical_columns=None, threshold=3):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = {}
    for col in numerical_columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[z_scores > threshold]
            outlier_report[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'threshold': threshold,
                'outliers': outliers[col].tolist()
            }
    return outlier_report


def handle_outliers(df, method='iqr', numerical_columns=None):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df.copy()
    for col in numerical_columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df_clean[col] = df[col].clip(lower=mean_val - 3 * std_val, upper=mean_val + 3 * std_val)
    return df_clean


def visualize_outliers(df, numerical_columns=None):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(15, 5 * n_rows))
    axes = axes.ravel()
    for i, col in enumerate(numerical_columns):
        if i * 2 < len(axes):
            axes[i * 2].boxplot(df[col].dropna())
            axes[i * 2].set_title(f'Box Plot - {col}')
            axes[i * 2 + 1].hist(df[col].dropna(), bins=30, alpha=0.7)
            axes[i * 2 + 1].set_title(f'Histogram - {col}')
    for j in range(i * 2 + 2, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def analyze_categorical_distribution(df, categorical_columns=None, top_n=10):
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    distribution_report = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            value_percentage = df[col].value_counts(normalize=True) * 100
            distribution_report[col] = {
                'unique_values': len(value_counts),
                'missing_values': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'top_values': value_counts.head(top_n).to_dict(),
                'value_percentage': value_percentage.head(top_n).to_dict()
            }
    return distribution_report


def visualize_categorical_distribution(df, categorical_columns=None, top_n=10):
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    n_cols = min(2, len(categorical_columns))
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if len(categorical_columns) == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, col in enumerate(categorical_columns):
        if i < len(axes):
            top_values = df[col].value_counts().head(top_n)
            axes[i].barh(range(len(top_values)), top_values.values)
            axes[i].set_yticks(range(len(top_values)))
            axes[i].set_yticklabels(top_values.index)
            axes[i].set_title(f'Top {top_n} values - {col}')
            axes[i].set_xlabel('Count')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def print_outlier_report(outlier_report, method_name):
    print(f"\n=== Outlier Report ({method_name}) ===")
    for col, info in outlier_report.items():
        print(f"\nColumn: {col}")
        print(f"  Outlier count: {info['outlier_count']} ({info['outlier_percentage']:.2f}%)")


def print_categorical_report(distribution_report):
    print("=== Categorical Distribution Report ===\n")
    for col, col_stats in distribution_report.items():
        print(f"Column: {col}")
        print(f"  Unique values: {col_stats['unique_values']}")
        print(f"  Missing values: {col_stats['missing_values']} ({col_stats['missing_percentage']:.2f}%)")
        print("  Top values:")
        for value, count in col_stats['top_values'].items():
            percentage = col_stats['value_percentage'].get(value, 0)
            print(f"    {value}: {count} ({percentage:.2f}%)")
        print("-" * 50)


def get_data_summary(df):
    return {
        'shape': df.shape,
        'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }


def analyze_dataframe(df, numerical_threshold=3):
    print("Starting Data Analysis\n")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}\n")
    if numerical_cols:
        print("Outlier Analysis (IQR Method)")
        iqr_outliers = detect_outliers_iqr(df, numerical_cols)
        print_outlier_report(iqr_outliers, "IQR")
        print("\nOutlier Analysis (Z-score Method)")
        zscore_outliers = detect_outliers_zscore(df, numerical_cols, numerical_threshold)
        print_outlier_report(zscore_outliers, "Z-score")
        print("\nVisualizing Outliers")
        visualize_outliers(df, numerical_cols)
    if categorical_cols:
        print("\nCategorical Distribution Analysis")
        cat_distribution = analyze_categorical_distribution(df, categorical_cols)
        print_categorical_report(cat_distribution)
        print("\nVisualizing Categorical Distributions")
        visualize_categorical_distribution(df, categorical_cols)
    print("\nAnalysis complete!")


def visualize_first_40_columns(df: pl.DataFrame, max_cols: int = 40):
    df_pd = df[:, :max_cols].to_pandas()
    num_cols = df_pd.select_dtypes(include=['number']).columns
    cat_cols = df_pd.select_dtypes(include=['object', 'category', 'bool']).columns
    total_cols = len(num_cols) + len(cat_cols)
    ncols = 4
    nrows = (total_cols + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
    col_idx = 0
    for col in num_cols:
        sns.violinplot(df_pd[col], ax=axes[col_idx])
        axes[col_idx].set_title(f'Histo: {col}')
        col_idx += 1
    for col in cat_cols:
        counts = df_pd[col].value_counts().sort_values(ascending=True).head(20)
        sns.barplot(x=counts.values, y=counts.index, ax=axes[col_idx])
        axes[col_idx].set_title(f'Bar: {col}')
        col_idx += 1
    for i in range(col_idx, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    MIC_method = ['MIC', 'MBC', 'MBEC', 'MBIC', 'MIc', 'MFC', 'MMC']

    data = pd.read_excel('validated_data_merged_tax.xlsx', index_col=0)

    cols_to_drop = [
        'new sn', 'sn', 'new', 'article_list', 'journal_name', 'publisher', 'year', 'title',
        'journal_is_oa', 'is_oa', 'oa_status', 'verification required',
        'verified_by', 'verification_date', 'has_mistake_in_data',
        'has_mistake_in_matadata', 'entry_status', 'comment', 'accept/reject',
        'Unnamed: 44', 'IdList', 'zeta_potential (mV)', 'pH during synthesis',
        'Concentration of precursor (mM)', 'hydrodynamic diameter', 'Precursor of NP',
        'Clade', 'Class', 'Family'
    ]

    df = data.copy()
    df['np_size_avg (nm)'] = df['np_size_avg (nm)'].astype('float64')

    fill_mode = ['time_set (hours)', 'Solvent for extract', 'shape']
    fill_mean = ['Duration preparing extract, min', 'Temperature for extract, C']
    for col_name in fill_mode:
        fill_na_mode(df, col_name)
    for col_name in fill_mean:
        fill_na_mean(df, col_name)

    filled_df = fill_size_missing_values(df)

    MIC_df = filled_df[filled_df['method'].isin(MIC_method)]
    MIC_df = MIC_df.drop(['concentration for ZOI (µg/ml)', 'zoi_np (mm)'], axis=1)


    MIC_df.to_csv('MIC_df.csv')
