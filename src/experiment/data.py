"""
Helpers to load (and pre-process?) the ACIC 2018 data
dataset description: https://www.researchgate.net/publication/11523952_Infant_Mortality_Statistics_from_the_1999_Period_Linked_BirthInfant_Death_Data_Set
"""
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing




def load_and_format_covariates(file_path='~/ml/IBM-Causal-Inference-Benchmarking-Framework/data/LBIDD/x.csv'):
    df = pd.read_csv(file_path, index_col='sample_id',header=0, sep=',')
    return df

def load_treatment_and_outcome(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path,index_col='sample_id',header=0, sep=',')

    dataset = covariates.join(output, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    x = dataset.values[:,:-2]
    if standardize:
        normal_scalar = preprocessing.StandardScaler()
        x = normal_scalar.fit_transform(x)
    return t.reshape(-1,1), y.reshape(-1,1), dataset.index, x

def load_ufids(file_path='/Users/claudiashi/data/small_sweep/params.csv'):
    df = pd.read_csv(file_path, header=0, sep=',')
    df = df[df['size'] > 4999]
    df = df[df['size'] < 10001]
    df_1 = df[df['instance'] == 1]
    df_2 = df[df['instance']  == 2]
    ufids = np.concatenate([df_1['ufid'].values,df_2['ufid'].values])
    return ufids


def load_params(file_path='/Users/claudiashi/data/small_sweep/params.csv'):
    df = pd.read_csv(file_path, header=0, sep=',')
    df = df[df['size'] > 4999]
    df = df[df['size'] < 10001]
    return df


def main():
    data_path = '../data/LIBDD'
    simulation_id = '43b75dcfc0fc49beb95a111098ae11b1'

    covariate_path = os.path.join(data_path, 'x.csv')
    all_covariates = load_and_format_covariates(covariate_path)
    simulation_file = os.path.join(data_path, 'scaling/factuals/', simulation_id + '.csv')
    t, y, sample_id, x = load_treatment_and_outcome(all_covariates, simulation_file)





if __name__ == '__main__':
    # main()
    pass