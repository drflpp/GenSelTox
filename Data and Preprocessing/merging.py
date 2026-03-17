import pandas as pd
import numpy as np
import polars as pl


def normalize_water(df, column):
    df[column] = df[column].str.lower().apply(
        lambda x: "water" if "water" in x else x
    )
    return df


def missing_value_percent(df):
    total_rows = len(df)
    missing = df.isna().sum()
    percent = (missing / total_rows) * 100
    return (
        pd.DataFrame({'missing_count': missing, 'missing_percent': percent})
        .sort_values('missing_percent', ascending=False)
    )


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


def fill_basics(df1):
    df = df1.copy()
    df['np_size_avg (nm)'] = df['np_size_avg (nm)'].astype('float64')
    fill_mode = ['time_set (hours)', 'shape']
    fill_mean = []
    for col_name in fill_mode:
        fill_na_mode(df, col_name)
    for col_name in fill_mean:
        fill_na_mean(df, col_name)
    filled_df = fill_size_missing_values(df)
    return filled_df


def make_bact_strain(df):
    df.strain = df.strain.fillna('nan')
    df.strain = df.strain.astype('str')
    df['bacteria_strain'] = df['bacteria'] + ' ' + df['strain']
    return df.copy()


def merging_gene(df, gene_df):
    data = df.copy()
    gene_matrix = gene_df.copy().rename({'Unnamed: 0': 'bacteria_strain'}, axis='columns')
    df_merged = pd.merge(data, gene_matrix, on='bacteria_strain')
    return df_merged


MIC_method = ['MIC', 'MBC', 'MBEC', 'MBIC', 'MIc', 'MFC', 'MMC']

useless_cols_MIC = [
    'Precursor of NP', 'Temperature for extract, C',
    'Concentration of precursor (mM)', 'Duration preparing extract, min',
    'Clade', 'zeta_potential (mV)', 'pH during synthesis',
    'has_mistake_in_matadata', 'entry_status', 'verification_date',
    'has_mistake_in_data', 'verified_by', 'hydrodynamic diameter',
    'comment', 'Unnamed: 44', 'accept/reject',
    'concentration for ZOI (µg/ml)', 'zoi_np (mm)',
    'new sn', 'sn', 'new', 'CID',
    'journal_is_oa', 'is_oa', 'oa_status', 'verification required',
    'IdList',
]


if __name__ == '__main__':
    data = pd.read_csv('new_synth_validated_data_merged_tax.csv', index_col=0, decimal=',')
    nps_desc = pd.read_csv('nanoparticle_descriptors.csv', index_col=0).rename(columns={"NP": 'np'})

    data1 = pd.merge(data, nps_desc, on='np')

    MIC_data = data1[data1["method"].isin(MIC_method)]
    ZOI_data = data1[data1["method"] == "ZOI"]

    MIC_data['Solvent for extract'] = MIC_data['Solvent for extract'].fillna('water')
    MIC_data = normalize_water(MIC_data, 'Solvent for extract')

    MIC_df = MIC_data.drop(useless_cols_MIC, axis=1, errors='ignore')
    MIC_df = fill_basics(MIC_df).drop_duplicates()

    ko_matrix = pd.read_csv('kegg_ko_matrix.csv', index_col=0)
    path_matrix = pd.read_csv('kegg_pathway_matrix.csv')

    MIC_m = make_bact_strain(MIC_df)

    MIC_ko = merging_gene(MIC_m, ko_matrix)
    MIC_path = merging_gene(MIC_m, path_matrix)

    MIC_ko.to_csv('MIC_df_ko.csv')
    MIC_path.to_csv('MIC_df_path.csv')
    MIC_m.to_csv('MIC_df_preprocessed.csv')
