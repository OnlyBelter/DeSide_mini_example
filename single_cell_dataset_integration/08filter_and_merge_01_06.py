import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from deside.plot import plot_marker_exp
from deside.single_cell import filter_cells_by_marker_gene
from deside.utility import check_dir, read_marker_gene
import anndata as an
from matplotlib import rcParams
from scipy.sparse import csr_matrix
sc.settings.set_figure_params(dpi=200)
sns.set()


if __name__ == '__main__':
    root_dir = r'../datasets/single_cell/merged_after_batch_correction'
    marker_gene_exp_dir = r'../datasets/single_cell/marker_gene_expression'
    result_dir_before_f = os.path.join(marker_gene_exp_dir, 'before_filtering')
    result_dir_after_f = os.path.join(marker_gene_exp_dir, 'after_filtering')
    single_cell_dir = r'../datasets/single_cell'
    result_dir_after_f_h5ad = os.path.join(single_cell_dir, 'filtered')

    all_filtering_info_file_path = os.path.join(result_dir_after_f, 'filtering_info.csv')
    sorted_cell_types = ['Epithelial Cells', 'T Cells', 'CD8 T', 'CD4 T', 'NK', 'B Cells', 'Endothelial Cells',
                         'Fibroblasts', 'Mast Cells', 'Macrophages', 'Neutrophils', 'DC']
    cell_type2markers = read_marker_gene('selected_marker_genes.csv', include_t_cell=True)
    if not os.path.exists(all_filtering_info_file_path):
        check_dir(result_dir_after_f_h5ad)
        dataset2file_name = {
            'hnscc_cillo_01': '01_all_HNSCC_TIL_Cillo_Immunity_2020_merged_after_batch_correction.h5ad',
            'pdac_pengj_02': '02_PDAC_Peng_J_Cell_Res_2019_merged_after_batch_correction.h5ad',
            'hnscc_puram_03': '03_all_HNSCC_Puram_Cell_2017_merged_after_batch_correction.h5ad',
            'pdac_steele_04': '04_PDAC_Steele_NCancer_2020_merged_after_batch_correction.h5ad',
            'luad_kim_05': '05_LUAD_primary_site_cancer_Kim_NCommun_2020_merged_after_batch_correction.h5ad',
            'nsclc_guo_06': '06_NSCLC_tumor_tissue_Guo_NMed_2018_merged_after_batch_correction.h5ad',
        }

        # read data
        dataset_quantile = []
        dataset_quantile_filtered = []
        filtering_info = []
        for dataset_id, file_name in dataset2file_name.items():
            print('--------------------Reading {}-----------------'.format(dataset_id))
            current_result_dir = os.path.join(result_dir_before_f, dataset_id)
            check_dir(current_result_dir)
            current_dataset = sc.read_h5ad(
                os.path.join(root_dir, file_name))
            print('>>> Plot expression profiles of marker genes before filtering...')
            print(current_dataset)
            quantile_df = plot_marker_exp(dataset_id=dataset_id, single_cell_dataset=current_dataset,
                                          cell_type2markers=cell_type2markers, max_exp=9600,
                                          result_dir=current_result_dir, cell_types=sorted_cell_types,
                                          groupby='leiden', exp_range=(50, 320), font_scale=1.2)
            dataset_quantile.append(quantile_df)

            # filter each sub-cluster
            print('>>> Start to filter this dataset...')
            current_result_dir_filtered = os.path.join(result_dir_after_f, dataset_id)
            check_dir(current_result_dir_filtered)
            _file_name = dataset_id + '_' + file_name.replace('merged_after_batch_correction',
                                                              'filtered_by_markers')
            current_result_after_f_h5ad = os.path.join(result_dir_after_f_h5ad, _file_name)
            filtering_info_file_path = os.path.join(current_result_dir_filtered, f'filtering_info_{dataset_id}.csv')
            if not os.path.exists(current_result_after_f_h5ad):
                filtered_result = filter_cells_by_marker_gene(single_cell_dataset=current_dataset,
                                                              dataset_id=dataset_id,
                                                              cell_types=sorted_cell_types,
                                                              cell_type2markers=cell_type2markers,
                                                              exp_range=(50, 320), groupby='leiden',
                                                              max_exp=9600)
                filtered_dataset = filtered_result[0]
                filtered_dataset.write(current_result_after_f_h5ad)
                current_filtering_info = filtered_result[1]
                current_filtering_info.to_csv(filtering_info_file_path)
            else:
                print(f'   # Using previous result: {current_result_after_f_h5ad}')
                filtered_dataset = an.read_h5ad(current_result_after_f_h5ad)
                current_filtering_info = pd.read_csv(filtering_info_file_path, index_col=0)
            filtering_info.append(current_filtering_info)

            # plot marker gene expression after filtering
            print('>>> Plot expression profiles of marker genes after filtering...')
            print(filtered_dataset)
            quantile_df_filtered = plot_marker_exp(dataset_id=dataset_id, single_cell_dataset=filtered_dataset,
                                                   cell_type2markers=cell_type2markers, max_exp=9600,
                                                   result_dir=current_result_dir_filtered, cell_types=sorted_cell_types,
                                                   groupby='leiden', font_scale=1.2)
            dataset_quantile_filtered.append(quantile_df)

        dataset2quantile_df = pd.concat(dataset_quantile)
        dataset2quantile_df.to_csv(os.path.join(result_dir_before_f, 'dataset2quantile.csv'))

        dataset2quantile_df_filtered = pd.concat(dataset_quantile_filtered)
        dataset2quantile_df_filtered.to_csv(os.path.join(result_dir_after_f, 'dataset2quantile.csv'))

        filtering_info_df = pd.concat(filtering_info)
        filtering_info_df.to_csv(all_filtering_info_file_path)
    else:
        print(f'   # Using previous results: {all_filtering_info_file_path}, '
              f'the directory of filtered .h5ad files: {result_dir_after_f_h5ad}')
        filtering_info_df = pd.read_csv(all_filtering_info_file_path, index_col=0)

    # merge all datasets together
    filtered_dataset2file = {
        'hnscc_cillo_01': {'file_name': 'hnscc_cillo_01_all_HNSCC_TIL_Cillo_Immunity_2020_filtered_by_markers.h5ad'},
        'pdac_pengj_02': {'file_name': 'pdac_pengj_02_PDAC_Peng_J_Cell_Res_2019_filtered_by_markers.h5ad'},
        'hnscc_puram_03': {'file_name': 'hnscc_puram_03_all_HNSCC_Puram_Cell_2017_filtered_by_markers.h5ad'},
        'pdac_steele_04': {'file_name': 'pdac_steele_04_PDAC_Steele_NCancer_2020_filtered_by_markers.h5ad'},
        'luad_kim_05': {'file_name': 'luad_kim_05_LUAD_primary_site_cancer_Kim_NCommun_2020_filtered_by_markers.h5ad'},
        'nsclc_guo_06': {'file_name': 'nsclc_guo_06_NSCLC_tumor_tissue_Guo_NMed_2018_filtered_by_markers.h5ad'},
    }
    all_dataset = {}
    merged_dataset_tmp_path = os.path.join(single_cell_dir, 'merged_6_sc_datasets_tmp.h5ad')
    merged_dataset_file_path = os.path.join(single_cell_dir, 'merged_6_sc_datasets.h5ad')
    if not os.path.exists(merged_dataset_tmp_path):
        # read data
        for dataset_id, file_name in filtered_dataset2file.items():
            print('--------------------Reading {}-----------------'.format(dataset_id))
            current_dataset = sc.read_h5ad(os.path.join(result_dir_after_f_h5ad, file_name["file_name"]))
            print(current_dataset)
            current_fil_info = filtering_info_df.loc[filtering_info_df['dataset_id'] == dataset_id, :].copy()
            current_dataset.obs['hit_cell_type'] = current_dataset.obs['leiden'].map(
                current_fil_info.to_dict()['hit_cell_type']
            )
            unanno_clusters = filtering_info_df.loc[(filtering_info_df['hit_cell_type'].isna()) &
                                                    (filtering_info_df['dataset_id'] == dataset_id), :].copy()
            if unanno_clusters.shape[0] > 0:
                unanno_cells = current_dataset.obs.loc[current_dataset.obs['leiden'].isin(unanno_clusters.index),
                                                       :].copy()
                _leiden_str = ', '.join(unanno_clusters.index.to_list())
                print(f'    # {unanno_cells.shape[0]} cells will be removed from cluster {_leiden_str}')
                current_dataset = current_dataset[~current_dataset.obs.index.isin(unanno_cells.index), :].copy()
            current_dataset.obs['dataset_id'] = dataset_id
            # leiden_1st represent the leiden label that obtained when each dataset was annotated
            current_dataset.obs['leiden_1st'] = current_dataset.obs['leiden']
            all_dataset[dataset_id] = current_dataset
            print(all_dataset[dataset_id])

        common_genes = []  # common genes for all dataset
        for di, dataset in all_dataset.items():
            if not common_genes:
                common_genes = dataset.var.index.to_list()
            else:
                common_genes = [i for i in dataset.var.index if i in common_genes]
        print(f'   # There are {len(common_genes)} common genes in {len(all_dataset)} datasets.')
        all_dataset = {k: ds[:, ds.var.index.isin(common_genes)].copy() for k, ds in all_dataset.items()}

        all_dataset_rescaled = {}  # rescaling to 1e6 since removed some genes
        for di, dataset in all_dataset.items():
            print(f'   # Rescaling dataset {di}')
            dataset.X = csr_matrix(np.power(2, dataset.X.A) - 1)  # non-log
            # merged_dataset_new.X = csr_matrix(dense_matrix)
            sc.pp.normalize_total(dataset, target_sum=1e6)
            sc.pp.log1p(dataset, base=2)
            all_dataset_rescaled[di] = dataset

        merged_dataset = None
        all_dataset_id = list(all_dataset.keys())
        for i in range(len(all_dataset_id)-1):
            print('   # Merging dataset {}'.format([all_dataset_id[i]]))
            next_dataset = all_dataset[all_dataset_id[i+1]]
            if merged_dataset is None:
                current_dataset = all_dataset[all_dataset_id[i]]
                merged_dataset = current_dataset.concatenate(next_dataset)
            else:
                merged_dataset = merged_dataset.concatenate(next_dataset)

        print('------------ The information of merged dataset --------------')
        merged_dataset.obs = merged_dataset.obs.loc[:, ['sample_id', 'leiden_1st', 'dataset_id',
                                                        'hit_cell_type', 'm_max_cd4/mean_cd8', 'm_cd4/m_cd8 group']]
        merged_dataset.var = merged_dataset.var.loc[:, ['gene_ids-0-0-0-0-0']]
        merged_dataset.var.rename(columns={'gene_ids-0-0-0-0-0': 'gene_ids'}, inplace=True)
        print(merged_dataset)

        # PCA
        sc.tl.pca(merged_dataset, svd_solver='arpack', n_comps=100, use_highly_variable=False, zero_center=False)
        # sc.pl.pca_variance_ratio(merged_dataset, n_pcs=30)

        # before removing batch effect
        sc.pp.neighbors(merged_dataset, n_neighbors=10, n_pcs=40)
        sc.tl.umap(merged_dataset)
        sc.tl.leiden(merged_dataset)
        rcParams['figure.figsize'] = 8, 8
        merged_dataset.obs.loc[merged_dataset.obs['m_max_cd4/mean_cd8'] > 2, 'm_max_cd4/mean_cd8'] = 2
        for c in ['leiden', 'leiden_1st', 'hit_cell_type', 'dataset_id', 'm_max_cd4/mean_cd8']:
            sc.pl.umap(merged_dataset, color=c, legend_loc='on data', legend_fontsize='small',
                       title='{}_{}'.format(c, 'merged_before_remove_batch_effect'), frameon=False,
                       save=f'_merged_before_remove_batch_effect_{c}.png'.replace('/', '_'), show=False)

        # Batch effect correction by BBKNN
        sc.external.pp.bbknn(merged_dataset, batch_key='dataset_id')
        sc.tl.umap(merged_dataset)
        sc.tl.leiden(merged_dataset)
        for c in ['leiden', 'leiden_1st', 'hit_cell_type', 'dataset_id', 'm_max_cd4/mean_cd8']:
            sc.pl.umap(merged_dataset, color=c, legend_loc='on data', legend_fontsize='small',
                       title='{}_{}'.format(c, 'merged_after_remove_batch_effect'), frameon=False,
                       save=f'_merged_after_remove_batch_effect_{c}.png'.replace('/', '_'), show=False)

        print('   Finding marker genes after batch effect correction...')
        sc.tl.rank_genes_groups(merged_dataset, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups(merged_dataset, n_genes=25, sharey=False)

        merged_dataset.write(merged_dataset_tmp_path)
    else:
        print(f'   Using previous merged dataset: {merged_dataset_tmp_path}...')
        merged_dataset = sc.read_h5ad(merged_dataset_tmp_path)

    if not os.path.exists(merged_dataset_file_path):
        top30 = pd.DataFrame(merged_dataset.uns['rank_genes_groups']['names']).head(30)
        common_marker_gene = pd.read_csv('selected_marker_genes.csv', index_col=0)
        for cluster_id in top30:
            print('Current cluster id: {}'.format(cluster_id))
            print(top30.loc[:, [cluster_id]].merge(common_marker_gene, left_on=cluster_id, right_index=True).iloc[:,
                  range(2)])

        # selected core marker genes
        marker_genes = [
            "EPCAM", "KRT18", "KRT19", 'KRT8',  # Epithelial cells
            "CD2", "CD3D", "CD3E",  # T Cell
            "CD8A", "CD8B",  # CD8 T
            "BATF", "ICOS", "IL7R", "CD4", "FOXP3", "TIGIT",  # CD4 T
            "GNLY", "NKG7", "KLRD1",  # NK
            "BANK1", "CD79A", "FCRL5", "MS4A1",  # B Cells
            "CLDN5", "ENG", "PLVAP", "VWF",  # Endothelial Cells
            "COL1A1", "COL1A2", "COL3A1", "MYL9",  # Fibroblasts
            'CPA3', 'HPGDS', 'GATA2',  # top 3 genes in cluster 12
            "AIF1", "CD14", "CD68", "MS4A7",  # Macrophages
            "CSF3R", "CXCR2", "FPR1", "SLC25A37",  # Neutrophils
            "GZMB", "CCR7", "LAMP3", "IRF8", 'IRF7'  # DC
        ]

        sc.pl.dotplot(merged_dataset, marker_genes, groupby='leiden',
                      save='merged_marker_genes_new.png', dendrogram=True)

        # annotating cell type by marker genes for each cluster
        new_cluster_names = {
            0: 'Epithelial Cells (1)',
            1: 'Macrophages (1)',
            2: 'CD4 T (1)',
            3: 'Macrophages (2)',
            4: 'CD8 T (1)',
            5: 'CD8 T (2)',
            6: 'CD4 T (2)',
            7: 'Epithelial Cells (2)',
            8: 'B Cells (1)',
            9: 'CD4 T (3)',
            10: 'Fibroblasts (1)',
            11: 'Fibroblasts (2)',
            12: 'CD4 T (4)',
            13: 'Endothelial Cells',
            14: 'Neutrophils',
            15: 'Mast Cells',
            16: 'B Cells (2)',
            17: 'NK',
            18: 'Epithelial Cells (3)',
            19: 'B Cells (3)',
            20: 'DC'
        }
        new_cluster_names_val = list(new_cluster_names.values())
        print(new_cluster_names_val)
        merged_dataset.rename_categories('leiden', new_cluster_names_val)
        merged_dataset.obs['cell_type'] = merged_dataset.obs['leiden'].map(lambda x: re.sub(r' \(\d\)', '', x))
        # only keep the cells which have the same cell type in
        # hit_cell_type (before merged) and cell_type (after merged)
        merged_dataset = merged_dataset[merged_dataset.obs['cell_type'] == merged_dataset.obs['hit_cell_type'],
                                        :].copy()
        print(merged_dataset)

        merged_dataset_group = merged_dataset.obs.groupby(['leiden', 'leiden_1st', 'dataset_id',
                                                           'hit_cell_type', 'm_cd4/m_cd8 group'])
        cells_in_small_group = []
        for i, group in merged_dataset_group:
            if (group.shape[0] > 0) and (group.shape[0] < 30):  # remove small cluster which contains less than 30 cells
                cells_in_small_group += group.index.to_list()
        merged_dataset = merged_dataset[~merged_dataset.obs.index.isin(cells_in_small_group), :].copy()

        merged_clusters_info = merged_dataset.obs.groupby(
            ['leiden', 'leiden_1st', 'dataset_id',
             'hit_cell_type', 'm_cd4/m_cd8 group']).size().reset_index(name='counts')
        merged_clusters_info.to_csv(
            os.path.join(single_cell_dir, '6_dataset_merged_clustering_info_after_removing_inconsistency.csv'))
        merged_dataset.obs.replace({'cell_type': {'Epithelial Cells': 'Cancer Cells', 'CD8 Tex': 'CD8 T'}},
                                   inplace=True)
        print(merged_dataset.obs['leiden'].value_counts())
        print(merged_dataset.obs['cell_type'].value_counts())
        print(merged_dataset.obs['dataset_id'].value_counts())
        merged_dataset.obs.to_csv(os.path.join(single_cell_dir, 'merged_6_sc_datasets_info.csv'))
        merged_dataset.write(merged_dataset_file_path)
        rcParams['figure.figsize'] = 8, 8
        for k in ['leiden', 'cell_type']:
            sc.pl.umap(merged_dataset, color=k, legend_loc='on data', legend_fontsize='small', title='', frameon=False,
                       save=f'_merged_filtered_{k}.png')
    else:
        print(f'   Previous result will be used: {merged_dataset_file_path}')
        merged_dataset = sc.read_h5ad(merged_dataset_file_path)
        print(merged_dataset)

    # plot the expression values of marker genes after merged
    quantile_df_filtered = plot_marker_exp(dataset_id='merged_6_sc_dataset', single_cell_dataset=merged_dataset,
                                           cell_type2markers=cell_type2markers, max_exp=9600,
                                           result_dir=os.path.join(marker_gene_exp_dir, 'merged_dataset'),
                                           cell_types=sorted_cell_types,
                                           groupby='leiden')
