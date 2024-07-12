"""
### Preprocessing scRNA-seq datasets for Scaden simulation
- https://figshare.com/articles/code/Publication_Figures/8234030/1
"""

import os
import pandas as pd
import scanpy as sc
from pathlib import Path
import numpy as np
from deside.utility import check_dir
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, color_map='viridis')

if __name__ == '__main__':
    out_path = "./R2Q6/scaden/scRNAseq_ds_read_counts"
    # num_top_genes = 150
    # ds_name = 'luad_kim_05_gse131907'
    file_dir = f'../datasets/single_cell/read_counts/'
    ds2file_path = {
        'Bi2021_Kidney_3CA': os.path.join(file_dir, 'Bi2021_Kidney_3CA', 'Bi2021_Kidney_3CA_single_cell_counts.txt'),
        'gbm_abdelfattah_12': os.path.join(file_dir, 'gbm_abdelfattah_12', 'gbm_abdelfattah_12_single_cell_counts.txt'),

        'HNSCC_Kürten2021_3CA': os.path.join(file_dir, 'HNSCC_Kürten2021_3CA',
                                              'HNSCC_Kürten2021_3CA_single_cell_counts_delete_SomeCellType.txt'),
        'Lee2020_Colorectal_3CA': os.path.join(file_dir, 'Lee2020_Colorectal_3CA',
                                               'Lee2020_Colorectal_3CA_single_cell_counts.txt'),
        'luad_kim_05_gse131907': os.path.join(file_dir, 'luad_kim_05_gse131907',
                                              'Lung_cancer_GSE131907_single_cell_counts.txt'),
        'prad_cheng_08': os.path.join(file_dir, 'prad_cheng_08', 'prad_cheng_08_single_cell_counts.txt'),
        'Qian2020_Breast_3CA': os.path.join(file_dir, 'Qian2020_Breast_3CA', 'Qian2020_Breast_3CA_single_cell_counts.txt'),
        'Sharma2020_Liver_3CA': os.path.join(file_dir, 'Sharma2020_Liver_3CA', 'Sharma2020_Liver_3CA_single_cell_counts.txt'),
        'Geistlinger2020_Ovarian_3CA': os.path.join(file_dir, 'Geistlinger2020_Ovarian_3CA.tar(1)',
                                                    'Geistlinger2020_Ovarian_3CA_single_cell_counts.txt'),

    }
    for ds_name, counts_file_path in ds2file_path.items():
        print('Dealing with:', ds_name)
        read_count_file_path = Path(counts_file_path)
        cell_type_file_path = Path(counts_file_path.replace('cell_counts', 'cell_annotation'))
        current_out_path = os.path.join(out_path, ds_name)
        check_dir(current_out_path)
        counts_out_path = os.path.join(current_out_path, f'{ds_name}_counts.txt')
        if not os.path.exists(counts_out_path):
            if ds_name in ['HNSCC_Kürten2021_3CA', 'prad_cheng_08']:
                adata = sc.read_text(read_count_file_path)
            else:
                adata = sc.read_text(read_count_file_path).T
            adata.obs_names_make_unique()
            adata.var_names_make_unique()
            # print(adata)
            print('The name of samples:', adata.obs_names)
            print('The name of genes:', adata.var_names)

            sc.pp.filter_cells(adata, min_genes=500)
            sc.pp.filter_genes(adata, min_cells=5)
            adata.obs['n_counts'] = adata.X.sum(axis=1)
            sc.pl.scatter(adata, x='n_counts', y='n_genes')

            # Filter cells
            adata = adata[adata.obs['n_genes'] < 7000, :]
            adata = adata[adata.obs['n_counts'] < 20000, :]
            adata.raw = sc.pp.log1p(adata, copy=True)

            # Normalize per cell
            sc.pp.normalize_per_cell(adata)

            # using the intersection samples of adata and cell_type_file
            cell_type_df = pd.read_csv(cell_type_file_path, sep='\t', index_col=0)
            print(cell_type_df)
            print(cell_type_df.shape)
            common_samples = list(set(adata.obs.index).intersection(cell_type_df.index))
            print(len(common_samples))
            cell_type_df = cell_type_df.loc[common_samples, :]
            cell_type_df.rename(columns={'cell_type': 'Celltype'}, inplace=True)
            new_samples = []
            for cell_type, group in cell_type_df.groupby('Celltype'):
                print(cell_type, group.shape)
                # sampling 1000 cells for each cell type if the number of cells is larger than 1000
                if group.shape[0] > 1000:
                    group = group.sample(n=1000, random_state=0)
                new_samples.append(group)
            cell_type_df = pd.concat(new_samples)
            print('After filtering: ', cell_type_df.shape)
            adata = adata[cell_type_df.index, :]
            assert adata.obs.index.tolist() == cell_type_df.index.tolist(), 'The order of samples are not the same.'

            # Save the complete matrix
            df = pd.DataFrame(adata.X)
            df.columns = adata.var.index
            df.index = adata.obs.index
            print(df.shape)
            print(df.head(2))

            df.to_csv(counts_out_path.replace('_counts.txt', '_with_sample_id.csv'))
            df.reset_index(drop=True).to_csv(counts_out_path, sep='\t')
            cell_type_df.to_csv(os.path.join(current_out_path, f'_celltypes.txt'), sep='\t')
        else:
            print(f'{current_out_path} exists.')
