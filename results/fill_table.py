import os
import re

import numpy as np
import pandas as pd

if __name__ == '__main__':
    resulting_table = pd.DataFrame(index=['VAE', 'IWAE, K=10', 'IWAE, K=50',
                                          'AMCVAE, K=3, fixed', 'AMCVAE, K=3, sigmoidal', 'AMCVAE, K=3, all_learnable',
                                          'AMCVAE, K=5, fixed', 'AMCVAE, K=5, sigmoidal', 'AMCVAE, K=5, all_learnable',
                                          'LMCVAE, K=5, fixed', 'LMCVAE, K=5, sigmoidal', 'LMCVAE, K=5, all_learnable',
                                          'LMCVAE, K=10, fixed', 'LMCVAE, K=10, sigmoidal',
                                          'LMCVAE, K=10, all_learnable',
                                          ], columns=['e10', 'e30', 'e100', 'n10', 'n30', 'n100'])

    list_of_files = os.listdir(os.path.dirname(os.path.realpath(__file__)))
    cleaned_list = [el for el in list_of_files if el.startswith('run-default')]

    unique_ids = sorted(np.unique([int(re.search(r'(\d{2,4})', el).group(1)) for el in cleaned_list]))

    for i, ind in enumerate(unique_ids):
        current_df_name = sorted([el for el in cleaned_list if el.find(str(ind)) > -1])

        try:
            elbo_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), current_df_name[0]))
            try:
                resulting_table.loc[resulting_table.index[i], "e10"] = elbo_df.values[10, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "e10"] = np.nan
            try:
                resulting_table.loc[resulting_table.index[i], "e30"] = elbo_df.values[30, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "e30"] = np.nan

            try:
                resulting_table.loc[resulting_table.index[i], "e100"] = elbo_df.values[100, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "e100"] = np.nan
        except:
            resulting_table.loc[resulting_table.index[i], "e10"] = np.nan
            resulting_table.loc[resulting_table.index[i], "e30"] = np.nan
            resulting_table.loc[resulting_table.index[i], "e100"] = np.nan

        try:
            nll_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), current_df_name[1]))
            try:
                resulting_table.loc[resulting_table.index[i], "n10"] = nll_df.values[0, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "n10"] = np.nan

            try:
                resulting_table.loc[resulting_table.index[i], "n30"] = nll_df.values[2, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "n30"] = np.nan

            try:
                resulting_table.loc[resulting_table.index[i], "n100"] = nll_df.values[9, 2]
            except:
                resulting_table.loc[resulting_table.index[i], "n100"] = np.nan
        except:
            resulting_table.loc[resulting_table.index[i], "n10"] = np.nan
            resulting_table.loc[resulting_table.index[i], "n30"] = np.nan
            resulting_table.loc[resulting_table.index[i], "n100"] = np.nan

    resulting_table.to_csv('CELEBA_1.csv')
