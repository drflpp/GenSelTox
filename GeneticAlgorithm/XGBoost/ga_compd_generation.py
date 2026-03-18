import time
import numpy as np
import pandas as pd
# import V4_models
import random
import polars as pl
import math
import pyarrow

from models import pipeline

name_of_pathogenic_bacteria = 'Escherichia coli ATCC 25922'  # Pseudomonas aeruginosa None, Escherichia coli, Pseudomonas aeruginosa, Staphylococcus aureus ATCC 29213, Staphylococcus aureus None, Acinetobacter baumannii, Salmonella typhimurium
name_of_good_bacteria = 'Pseudomonas aeruginosa nan'  # Bacillus subtilis None, Escherichia coli ATCC 25922

population_size = 100
df_MIC = pd.read_csv(r'D:\Projects\SelTox\GeneticAlgorithm\Catboost\data\final_df1_catboost_orig.csv', low_memory=False,
                     index_col=0)
df_MIC_bacteria = pd.read_csv(r'D:\Projects\SelTox\GeneticAlgorithm\Catboost\data\final_df1_catboost_orig_bact.csv',
                              low_memory=False, index_col=0)

X = df_MIC_bacteria.drop(['MIC_NP___g_mL_'],
                         axis=1)  # no need for concentration, zoi or gi as all of these parameters will be predicted
X['Bio_component_class'] = X['Bio_component_class'].fillna('none')

old_df = pd.read_csv(r'D:\Projects\SelTox\GeneticAlgorithm\Catboost\data\final_df1_catboost_orig_bact.csv', index_col=0)

material_list = ['amw', 'Source_origin', 'Capping_type', 'chi0v', 'Template_type',
                 'Bio_component_class', 'Red_env_strength',
                 'Red_env_type', 'Solvent_for_extract', 'Valance_electron',
                 'Solvent_polar', 'lipinskiHBA', 'NumHBA',
                 'CrippenClogP', 'hallKierAlpha', 'kappa1', 'np_synthesis'
                 ]

uniq_bacteria_data = X
print(uniq_bacteria_data)
"""uniq value datasets"""
uniq = []  # stores all the unique characters available in the dataset, it helps to make a new population with random parameters
for a in range(len(X.columns)):
    uni = pd.unique(X.iloc[:, a])
    uniq.append(uni)

drop_bacteria = ['bacteria_strain']


def individuals():
    indv = []
    for a in range(len(X.columns)):
        if a == 1:
            diff_mid = random.uniform(3, 13)
            uniqas = indv[0] + diff_mid

        elif a == 2:
            diff_max = random.uniform(5, 10)
            uniqas = indv[1] + diff_max

        elif a == 4:
            uniqas = 'MIC'
        elif a == 6:
            uniqas = 24


        elif a == 8:
            values = ['spherical', 'rod-shaped', 'quasi-spherical', 'spheroidal', 'cubic',
                      'oval', 'flat', 'pseudo-spherical', 'ellipsoidal', 'hexagonal',
                      'tubular']  # например, [10, 20, 30, 40]
            probabilities = [0.7983425414364641,
                             0.04488950276243094,
                             0.03383977900552486,
                             0.029005524861878452,
                             0.022790055248618785,
                             0.020718232044198894,
                             0.012430939226519336,
                             0.011049723756906077,
                             0.009668508287292817,
                             0.008977900552486187,
                             0.008287292817679558]  # вероятности для каждого значения

            # Взвешенный случайный выбор
            uniqas = random.choices(values, weights=probabilities, k=1)[0]



        else:
            uniqas = random.choice(uniq[a])

        indv.append(uniqas)
    return indv


indiv1 = individuals()
indiv2 = individuals()
print(indiv1)
print(indiv2)


def population(size):
    pops = []
    for indv in range(2 * size):
        single = individuals()
        pops.append(single)
    new_one = pd.DataFrame(data=pops, columns=X.columns)

    new = new_one.head(size)

    new = new.reset_index(drop=True)
    material_descriptor = X.drop(drop_bacteria, axis=1).iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]

    material_descriptor = material_descriptor.reset_index(drop=True)

    new[material_list] = material_descriptor[material_list]
    return new


dff = population(population_size)


def bacteria_type(population_df):
    single_bacteria_pathogen = uniq_bacteria_data.loc[uniq_bacteria_data[
                                                          'bacteria_strain'] == name_of_pathogenic_bacteria]  # Pseudomonas aeruginosa None, Escherichia coli, Pseudomonas aeruginosa, Staphylococcus aureus ATCC 29213, Staphylococcus aureus None, Acinetobacter baumannii, Salmonella typhimurium
    single_bacteria_nonpathogen = uniq_bacteria_data.loc[uniq_bacteria_data[
                                                             'bacteria_strain'] == name_of_good_bacteria]  # Bacillus subtilis None, Escherichia coli ATCC 25922
    pop_non_pathogen = pd.concat([single_bacteria_nonpathogen] * len(population_df), ignore_index=True)
    pop_pathogen = pd.concat([single_bacteria_pathogen] * len(population_df), ignore_index=True)
    df_pathogen = population_df.copy()
    df_non_pathogen = population_df.copy()
    print(population_df)
    feature_list = ['mdr', 'prim_specific_habitat', 'max_Incub_period__h', 'avg_Incub_period__h',
                    'K07123', 'K03629', 'min_Incub_period__h', 'K07050', 'K20345', 'sec_habitat', 'bac_type', 'K23945',
                    'common_environment', 'K01191', 'K13566', 'K07484', 'K25602', 'K11206', 'K07486', 'K12942',
                    'K01153',
                    'K02027', 'K00849', 'K01878', 'K00432', 'K01026', 'K10844', 'K03741', 'K00252', 'K01190', 'K03703',
                    'K09936',
                    'K07485', 'K07778', 'K16148', 'bacteria_strain'  # , 'bacteria_strain_mdr'
                    ]
    df_non_pathogen[feature_list] = pop_non_pathogen[feature_list]
    df_pathogen[feature_list] = pop_pathogen[feature_list]

    return df_non_pathogen, df_pathogen


print('here', bacteria_type(dff))


def fitness(df):
    n_path, path_gen = bacteria_type(df)

    normal_b = pipeline.predict(pl.from_pandas(n_path))
    pathogen_b = pipeline.predict(pl.from_pandas(path_gen))
    fitness = []
    norm_v = []
    path_v = []
    for a in range(len(normal_b)):
        n = normal_b[a]
        c = pathogen_b[a]
        fit = n - c  # higher MIC for normal bacteria and lower MIC for pathogenic bacteria
        fitnn = fit.tolist()
        norm_v.append(n)
        path_v.append(c)
        fitness.append(fitnn)
    copy = n_path.assign(pred_MIC_norm=norm_v)
    copy1 = copy.assign(pathogenic_bacteria=path_gen['bacteria_strain'].tolist())
    copy2 = copy1.assign(pred_MIC_pathogen=path_v)
    copy3 = copy2.assign(Fitness=fitness)
    copy3 = copy3.sort_values('Fitness', ascending=False)
    copy3['pred_norm_MIC_original'] = math.e ** copy3['pred_MIC_norm']
    copy3['pred_path_MIC_original'] = math.e ** copy3['pred_MIC_pathogen']
    return copy3


fitness(dff)
