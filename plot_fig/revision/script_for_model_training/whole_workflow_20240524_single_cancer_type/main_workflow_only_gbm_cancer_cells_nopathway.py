import os
import warnings
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
from deside.decon_cf import DeSide
from deside.workflow import run_step3, run_step4
from deside.utility.read_file import ReadH5AD, ReadExp, read_gene_set
from deside.plot import plot_pca, plot_pred_cell_prop_with_cpe
from deside.utility import (check_dir, print_msg, save_key_params, get_x_by_pathway_network,
                            sorted_cell_types, do_pca_analysis, do_umap_analysis, set_fig_style)
from deside.simulation import (BulkGEPGenerator, get_gene_list_for_filtering,
                               filtering_by_gene_list_and_pca_plot, SingleCellTypeGEPGenerator)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
set_fig_style(font_family='Arial', font_size=8)


if __name__ == '__main__':
    debug = False
    set_fig_style(font_family='Arial', font_size=8)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    physical_devices = tf.config.list_physical_devices('GPU')
    print('>>> Physical GPUs:', physical_devices)
    try:
        # Disable first GPU
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for the first GPU
        assert len(logical_devices) == len(physical_devices) - 2
        print(len(physical_devices), "Physical GPUs,", len(logical_devices), "Logical GPU")
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # all_cell_types = sorted_cell_types
    deside_data_dir = './datasets'
    subgroup_by = ['cell_prop']
    sc_dataset_ids = ['hnscc_cillo_01', 'pdac_peng_02', 'hnscc_puram_03', 'pdac_steele_04',
                      'luad_kim_05', 'nsclc_guo_06', 'prad_cheng_08',
                      'prad_dong_09', 'hcc_sun_10',
                      'gbm_neftel_11', 'gbm_abdelfattah_12',
                      ]
    cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PAAD', 'PRAD',
                    'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC']
    # CCC <= 0.5 when dataset D2 is used to train the model
    # cancer_types_for_filtering = ['KIRC', 'LIHC',
    #                               'PRAD', 'THCA', 'UCEC']
    cancer_types_for_filtering = cancer_types.copy()
    # coefficient to correct the difference of total RNA abundance in different cell types
    # alpha_total_rna_coefficient = {'B Cells': 1.0, 'CD4 T': 1.0, 'CD8 T': 1.0, 'DC': 1.0,
    #                                'Endothelial Cells': 9.566, 'Cancer Cells': 13.6, 'Fibroblasts': 9.369,
    #                                'Macrophages': 6.0, 'Mast Cells': 1.0, 'NK': 1.0, 'Neutrophils': 0.37}
    alpha_total_rna_coefficient = {'B Cells': 1.0, 'CD4 T': 1.0, 'CD8 T': 1.0, 'DC': 1.0,
                                   'Endothelial Cells': 1.0, 'Cancer Cells': 1.0, 'Fibroblasts': 1.0,
                                   'Macrophages': 1.0, 'Mast Cells': 1.0, 'NK': 1.0, 'Neutrophils': 1.0,
                                   'Double-neg-like T': 1.0, 'Monocytes': 1.0}
    # cell type to subtypes, if no subtypes, just use the cell type name as the subtype name
    cell_type2subtype_sct = {'B Cells': ['Non-plasma B cells', 'Plasma B cells'],
                             'CD4 T': ['CD4 T conv', 'CD4 Treg'], 'CD8 T': ['CD8 T (GZMK high)', 'CD8 T effector'],
                             'DC': ['mDC', 'pDC'], 'Endothelial Cells': ['Endothelial Cells'],
                             'Cancer Cells': ['Epithelial Cells', 'Glioma Cells'],
                             'Fibroblasts': ['CAFs', 'Myofibroblasts'], 'Macrophages': ['Macrophages'],
                             'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
                             'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}
    cell_type2subtypes = {'B Cells': ['Non-plasma B cells', 'Plasma B cells'],
                          'CD4 T': ['CD4 T'], 'CD8 T': ['CD8 T (GZMK high)', 'CD8 T effector'],
                          'DC': ['DC'], 'Endothelial Cells': ['Endothelial Cells'],
                          'Cancer Cells': ['Cancer Cells'],
                          'Fibroblasts': ['CAFs', 'Myofibroblasts'], 'Macrophages': ['Macrophages'],
                          'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
                          'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}

    # cell_type2subtypes = {'B Cells': ['B Cells'], 'CD4 T': ['CD4 T'], 'CD8 T': ['CD8 T'],
    #                       'DC': ['DC'], 'Endothelial Cells': ['Endothelial Cells'], 'Cancer Cells': ['Cancer Cells'],
    #                       'Fibroblasts': ['Fibroblasts'], 'Macrophages': ['Macrophages'],
    #                       'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
    #                       'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}

    all_cell_types = sorted([i for v in cell_type2subtypes.values() for i in v])
    all_cell_types = [i for i in sorted_cell_types if i in all_cell_types]
    subtype2type = {i: k for k, v in cell_type2subtypes.items() for i in v}

    # for gene-level filtering
    gene_list_type = 'high_corr_gene_and_quantile_range'
    remove_cancer_cell = True
    gene_quantile_range = [0.005, 0.5, 0.995]  # gene-level filtering
    gep_filtering_quantile = (0, 0.95)  # GEP-level filtering, L1-norm threshold
    filtering_in_pca_space = True
    pca_n_components = 0.9
    n_base = 100  # 100
    # optional, if set a prior cell proportion range, the GEP-filtering step will be faster, default is (0, 1)
    # cell_prop_prior = {'B Cells': (0, 0.25), 'CD4 T': (0, 0.5), 'CD8 T': (0, 0.5),
    #                    'DC': (0, 0.25), 'Mast Cells': (0, 0.25), 'NK': (0, 0.25), 'Neutrophils': (0, 0.25),
    #                    'Endothelial Cells': (0, 1), 'Fibroblasts': (0, 1), 'Macrophages': (0, 1),
    #                    'Cancer Cells': (0, 1), 'Double-neg-like T': (0, 0.5), 'Monocytes': (0, 0.25)}
    cell_prop_prior = {'B Cells': (0, 1), 'CD4 T': (0, 1), 'CD8 T': (0, 1),
                       'DC': (0, 1), 'Mast Cells': (0, 1), 'NK': (0, 1), 'Neutrophils': (0, 1),
                       'Endothelial Cells': (0, 1), 'Fibroblasts': (0, 1), 'Macrophages': (0, 1),
                       'Cancer Cells': (0, 1), 'Double-neg-like T': (0, 1), 'Monocytes': (0, 1)}
    subtype_prior = {i: cell_prop_prior[subtype2type[i]] for i in all_cell_types if i not in cell_prop_prior}
    cell_prop_prior.update(subtype_prior)

    # cell_prop_prior = None
    # read two gene sets as pathway mask
    gene_set_file_path1 = f'./datasets/gene_set/c2.cp.kegg.v2023.1.Hs.symbols.gmt'
    gene_set_file_path2 = f'./datasets/gene_set/c2.cp.reactome.v2023.1.Hs.symbols.gmt'
    all_pathway_files = [gene_set_file_path1, gene_set_file_path2]
    # pathway_mask = read_gene_set(all_pathway_files)  # genes by pathways
    pathway_mask = None
    method_adding_pathway = 'add_to_end'  # 'convert' / 'add_to_end'
    # for pathway profiles
    input_gene_list = 'filtered_genes'  # 'intersection_with_pathway_genes' / 'filtered_genes' / None (using all genes)
    cell_type_col = 'cell_type'  # the column name of cell type in the merged sc dataset
    cell_subtype_col = 'cell_subtype'  # the column name of cell subtype in the merged sc dataset
    single_cancer_type = 'gbm'
    simu_ds_dir = f'simulated_bulk_cell_only_{single_cancer_type}_cancer_cells'
    # simu_ds_dir = 'simulated_bulk_cell_dataset_subtypes3'

    dataset2parameters = {
        # 'HNSC': {'sc_dataset_id': ['hnscc_cillo_01', 'hnscc_puram_03'], 'n_per_gradient': {'all': 100}},
        # 'LUAD': {'sc_dataset_id': ['luad_kim_05'], 'n_per_gradient': 100},
        # 'PAAD': {'sc_dataset_id': ['pdac_pengj_02', 'pdac_steele_04'], 'n_per_gradient': 100},

        # 'SCT_POS_N100': {'n_each_cell_type': 100, 'cell_type2subtype': cell_type2subtype_sct},
        # # 'SCT_POS_N3K': {'n_each_cell_type': 3000, 'cell_type2subtype': cell_type2subtype_sct},
        # 'SCT_POS_N100_test': {'n_each_cell_type': 100, 'cell_type2subtype': cell_type2subtypes,
        #                       'test_set': True},
        f'SCT_POS_N10K_{single_cancer_type}': {'n_each_cell_type': 10000, 'cell_type2subtype': cell_type2subtype_sct},

        f'Mixed_N100K_segment_without_filtering_{single_cancer_type}':
            {'sc_dataset_ids': sc_dataset_ids,
             'cell_type2subtype': cell_type2subtypes,
             'n_samples': 100000,
             'sampling_method': 'segment',
             'filtering': False,
             },

        # 'Mixed_N100K_segment_luad': {'sc_dataset_ids': sc_dataset_ids,
        #                              'cell_type2subtype': cell_type2subtypes,
        #                              'n_samples': 100000,
        #                              'sampling_method': 'segment',
        #                              'filtering': True,
        #                              },

        # 'Mixed_N100K_random': {'sc_dataset_ids': sc_dataset_ids,
        #                        'cell_type2subtype': cell_type2subtypes,
        #                        'n_samples': 100000,
        #                        'sampling_method': 'random',
        #                        'filtering': False,
        #                        },
        # 'Test_set0': {'sc_dataset_ids': sc_dataset_ids,
        #               'cell_type2subtype': cell_type2subtypes,
        #               'n_samples': 3000,
        #               'sampling_method': 'random',
        #               'filtering': False,
        #               },
        # 'Test_set1': {'sc_dataset_ids': sc_dataset_ids,
        #               'cell_type2subtype': cell_type2subtypes,
        #               'n_samples': 3000,
        #               'sampling_method': 'segment',
        #               'filtering': True,
        #               },
        # 'Test_set2': {'sc_dataset_ids': sc_dataset_ids,
        #               'cell_type2subtype': cell_type2subtypes,
        #               'n_samples': 3000,
        #               'sampling_method': 'segment',
        #               'filtering': False,
        #               },
    }
    # params: dict[str | dict[str | Any, list[str] | Any] | list | int]
    params: 'Union[int, Dict[Union[str, Any], Union[List[str], Any, str, list]]]'
    for ds, params in dataset2parameters.items():
        if ('filtering' in params) and ('filtering_ref_types' not in params):
            if params['filtering']:
                params['filtering_ref_types'] = cancer_types_for_filtering
            else:
                params['filtering_ref_types'] = []
    merged_sc_dataset_file_path = os.path.join(deside_data_dir, 'single_cell',
                                               f'merged_12_sc_ds_only_{single_cancer_type}_cancer_cells.h5ad')

    tcga_data_dir = os.path.join(deside_data_dir, 'TCGA', 'tpm')   # input
    tcga_merged_tpm_file_path = os.path.join(tcga_data_dir, 'merged_tpm.csv')
    tcga2cancer_type_file_path = os.path.join(tcga_data_dir, 'tcga_sample_id2cancer_type.csv')
    # outlier_tcga_file_path = r'./datasets/TCGA/outliers_TCGA.csv'  #
    outlier_tcga_file_path = None  #
    cancer_purity_file_path = os.path.join(deside_data_dir, 'cancer_purity', 'cancer_purity.csv')  # input, CPE values
    marker_gene_file_path = os.path.join(deside_data_dir, 'single_cell', 'selected_marker_genes_subtypes.csv')

    n_sc_datasets = len(sc_dataset_ids)
    result_root_dir = os.path.join('results', f'whole_workflow_20240524_single_cancer_type')

    sc_dataset_dir = os.path.join(deside_data_dir,
                                  f'generated_sc_dataset_{n_sc_datasets}ds_n_base{n_base}_all_subtypes')
    sct_dataset_file_path = os.path.join(sc_dataset_dir,
                                         f'simu_bulk_exp_SCT_POS_N10K_{single_cancer_type}_log2cpm1p.h5ad')
    if debug:
        sct_dataset_file_path = os.path.join(sc_dataset_dir, f'simu_bulk_exp_SCT_POS_N100_log2cpm1p.h5ad')
    signature_score_method = 'mean_exp'

    # step1
    ll_ac = 'sigmoid' if remove_cancer_cell else 'softmax'
    current_project_id = f'DeSide_DiNonCancer_{single_cancer_type}_ds_{ll_ac}_nopathway'
    # current_project_id = f'DeSide_DiNonCancer_{single_cancer_type}_ds_softmax'
    log_file_path = os.path.join(result_root_dir, current_project_id, 'DeSide_running_log.txt')
    check_dir(os.path.dirname(log_file_path))

    deside_parameters = {'architecture': ([200, 2000, 2000, 2000, 50],
                                          [0.05, 0.05, 0.05, 0.2, 0]),
                         'architecture_for_pathway_network': ([50, 500, 500, 500, 50],
                                                              [0, 0, 0, 0, 0]),
                         'loss_function_alpha': 0.5,  # alpha*mae + (1-alpha)*rmse, mae means mean absolute error
                         'normalization': 'layer_normalization',  # batch_normalization / layer_normalization / None
                         # 1 means to add a normalization layer, input | the first hidden layer | ... | output
                         'normalization_layer': [0, 0, 1, 1, 1, 1],  # 1 more parameter than the number of hidden layers
                         'pathway_network': False,  # using an independent pathway network
                         'last_layer_activation': ll_ac,  # sigmoid / softmax
                         'learning_rate': 1e-4,
                         'batch_size': 128}

    if remove_cancer_cell:
        deside_parameters['last_layer_activation'] = 'sigmoid'

    sampling_method2dir = {
        'random': os.path.join(deside_data_dir, simu_ds_dir, '{}_{}ds_n_base{}'),
        'segment': os.path.join(deside_data_dir, simu_ds_dir, '{}_{}ds_{}_n_base{}_median_gep'),
    }

    dataset2path = {}
    _postfix_filtered_ds_naming = ''
    print_msg('Step1: simulating bulk cell expression profiles...', log_file_path=log_file_path)
    # naming the file of filtered bulk cell dataset
    q_names = ['q_' + str(int(q * 1000)/10) for q in gene_quantile_range]
    replace_by = f'_filtered_by_{gene_list_type}.h5ad'
    high_corr_gene_list = []
    if 'quantile_range' in gene_list_type:
        replace_by = f'_filtered_by_{gene_list_type}_{q_names[0]}_{q_names[2]}.h5ad'

    for dataset_name, params in dataset2parameters.items():
        print_msg(f'Generating dataset {dataset_name}...', log_file_path=log_file_path)
        if 'SCT' in dataset_name:
            b_gen_obj = SingleCellTypeGEPGenerator(simu_bulk_dir=sc_dataset_dir,
                                                   cell_type2subtype=params['cell_type2subtype'],
                                                   sc_dataset_ids=sc_dataset_ids, bulk_dataset_name=dataset_name,
                                                   merged_sc_dataset_file_path=merged_sc_dataset_file_path,
                                                   zero_ratio_threshold=0.97,
                                                   cell_type_col_name=cell_type_col,
                                                   subtype_col_name=cell_subtype_col)
            b_gen_obj.generate_samples(n_sample_each_cell_type=params['n_each_cell_type'], sample_type='positive',
                                       sep_by_patient=False, n_base_for_positive_samples=n_base)
        else:
            sampling_method = params['sampling_method']
            # the folder of simulated bulk cells
            simu_bulk_exp_dir = sampling_method2dir[sampling_method]
            if sampling_method in ['segment']:
                if params['filtering']:
                    _postfix_filtered_ds_naming = f'_{len(cancer_types_for_filtering)}cancer'
                    if filtering_in_pca_space:
                        _postfix_filtered_ds_naming += f'_pca_{pca_n_components}'
                    simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets,
                                                                 gep_filtering_quantile[1],
                                                                 str(n_base) + _postfix_filtered_ds_naming)
                else:
                    simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets, 'no_filtering', n_base)
            else:  # 'random'
                simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets, n_base)
            check_dir(simu_bulk_exp_dir)

            bulk_generator = BulkGEPGenerator(simu_bulk_dir=simu_bulk_exp_dir,
                                              merged_sc_dataset_file_path=None,
                                              cell_type2subtype=params['cell_type2subtype'],
                                              sc_dataset_ids=params['sc_dataset_ids'],
                                              bulk_dataset_name=dataset_name,
                                              sct_dataset_file_path=sct_dataset_file_path,
                                              check_basic_info=False,
                                              tcga2cancer_type_file_path=tcga2cancer_type_file_path,
                                              total_rna_coefficient=alpha_total_rna_coefficient,
                                              cell_type_col_name=cell_type_col,
                                              subtype_col_name=cell_type_col,)  # the class name is in the same column
            # bulk_generator.generate_single_cell_dataset(gen_sc_dataset_dir=generated_sc_dataset_dir)
            generated_bulk_gep_fp = bulk_generator.generated_bulk_gep_fp
            dataset2path[dataset_name] = generated_bulk_gep_fp
            if not os.path.exists(generated_bulk_gep_fp):
                bulk_generator.generate_gep(n_samples=params['n_samples'],
                                            simu_method='mul',
                                            sampling_method=params['sampling_method'],
                                            reference_file=tcga_merged_tpm_file_path,
                                            ref_exp_type='TPM',
                                            filtering=params['filtering'],
                                            filtering_ref_types=params['filtering_ref_types'],
                                            gep_filtering_quantile=gep_filtering_quantile,
                                            n_threads=5,
                                            log_file_path=log_file_path,
                                            show_filtering_info=True,
                                            # filtering_method='median_gep',
                                            filtering_method='median_gep',  # median_gep / mean_gep / linear_mmd
                                            cell_prop_prior=cell_prop_prior,
                                            filtering_in_pca_space=filtering_in_pca_space,
                                            norm_ord=1, pca_n_components=pca_n_components)

            # # get the highly correlated genes between the gene expression values
            # # and the cell proportions of each cell type in D2 and save the gene list
            # if dataset_name == f'Mixed_N100K_{sampling_method}_without_filtering':
            #     # the correlation between the gene expression values and the cell proportions of each cell type
            #     d2_dir = os.read_count_file_path.join(simu_bulk_exp_dir, 'D2')
            #     check_dir(d2_dir)
            #     corr_result_fp = os.read_count_file_path.join(d2_dir, f'corr_cell_frac_with_gene_exp_D2.csv')
            #     high_corr_gene_file_path = os.read_count_file_path.join(d2_dir, f'gene_list_filtered_by_high_corr_gene.csv')
            #     if not os.read_count_file_path.exists(high_corr_gene_file_path):
            #         print(f'High correlation gene list will be saved in: {high_corr_gene_file_path}')
            #         high_corr_gene_list = get_gene_list_for_filtering(bulk_exp_file=generated_bulk_gep_fp,
            #                                                           filtering_type='high_corr_gene',
            #                                                           corr_result_fp=corr_result_fp,
            #                                                           tcga_file=tcga_merged_tpm_file_path,
            #                                                           quantile_range=gene_quantile_range,
            #                                                           result_file_path=high_corr_gene_file_path,
            #                                                           corr_threshold=0.3, n_gene_max=1000)
            #     else:
            #         print(f'High correlation gene list file existed: {high_corr_gene_file_path}')
            #         high_corr_gene_list = pd.read_csv(high_corr_gene_file_path)
            #         high_corr_gene_list = high_corr_gene_list['gene_name'].to_list()
            # if gene_list_type == 'high_corr_gene_and_quantile_range':
            #     assert len(high_corr_gene_list) > 0, 'The high correlation gene list is empty!'
            # # gene-level filtering that depends on high correlation genes and quantile range (each dataset itself)
            # if params['filtering'] and 'Mixed' in dataset_name:
            #     filtered_file_path = generated_bulk_gep_fp.replace('.h5ad', replace_by)
            #     if not os.read_count_file_path.exists(filtered_file_path):
            #         gene_list = high_corr_gene_list.copy()
            #         # get gene list, filtering, PCA and plot
            #         current_result_dir = os.read_count_file_path.join(simu_bulk_exp_dir, dataset_name)
            #         check_dir(current_result_dir)
            #         # the gene list file for current dataset
            #         if 'quantile_range' in gene_list_type:
            #             gene_list_file_path = os.read_count_file_path.join(simu_bulk_exp_dir, dataset_name,
            #                                                f'gene_list_filtered_by_quantile_range.csv')
            #             gene_list_file_path = gene_list_file_path.replace('.csv', f'_{q_names[0]}_{q_names[2]}.csv')
            #             if not os.read_count_file_path.exists(gene_list_file_path):
            #                 print(f'Gene list of {dataset_name} will be saved in: {gene_list_file_path}')
            #                 quantile_gene_list = get_gene_list_for_filtering(bulk_exp_file=generated_bulk_gep_fp,
            #                                                                  filtering_type='quantile_range',
            #                                                                  tcga_file=tcga_merged_tpm_file_path,
            #                                                                  quantile_range=gene_quantile_range,
            #                                                                  result_file_path=gene_list_file_path,
            #                                                                  q_col_name=q_names)
            #             else:
            #                 print(f'Gene list file existed: {gene_list_file_path}')
            #                 quantile_gene_list = pd.read_csv(gene_list_file_path)
            #                 quantile_gene_list = quantile_gene_list['gene_name'].to_list()
            #             # get the intersection of the two gene lists (high correlation genes and within quantile range)
            #             gene_list = [gene for gene in gene_list if gene in quantile_gene_list]
            #         # save the filtered gene list
            #         gene_list_file_path = os.read_count_file_path.join(simu_bulk_exp_dir, dataset_name,
            #                                            f'gene_list_filtered_by_{gene_list_type}.csv')
            #         pd.DataFrame({'gene_name': gene_list}).to_csv(gene_list_file_path, index=False)
            #         bulk_exp_obj = ReadH5AD(generated_bulk_gep_fp)
            #         bulk_exp = bulk_exp_obj.get_df()
            #         bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction()
            #         tcga_exp = ReadExp(tcga_merged_tpm_file_path, exp_type='TPM').get_exp()
            #         pc_file_name = f'both_TCGA_and_simu_data_{dataset_name}'
            #         pca_model_file_path = os.read_count_file_path.join(current_result_dir,
            #                                            f'{pc_file_name}_PCA_{gene_list_type}.joblib')
            #         pca_data_file_path = os.read_count_file_path.join(current_result_dir,
            #                                           f'{dataset_name}_PCA_with_TCGA_{gene_list_type}.csv')
            #         # save GEPs data by filtered gene list
            #         print('Filtering by gene list and PCA plot')
            #         filtering_by_gene_list_and_pca_plot(bulk_exp=bulk_exp, tcga_exp=tcga_exp, gene_list=gene_list,
            #                                             result_dir=current_result_dir, n_components=2,
            #                                             simu_dataset_name=dataset_name,
            #                                             pca_model_name_postfix=gene_list_type,
            #                                             pca_model_file_path=pca_model_file_path,
            #                                             pca_data_file_path=pca_data_file_path,
            #                                             h5ad_file_path=filtered_file_path,
            #                                             cell_frac_file=bulk_exp_cell_frac)

    # Step2, training model
    training_set_list = [
        # 'Mixed_N100K_random',
        # 'Mixed_N100K_random_1',
        # 'Mixed_N100K_random_2',
        # 'Mixed_N100K_random_3',
        # 'Mixed_N100K_random_4',

        # 'Mixed_N100K_segment_filtered_luad-Mixed_N100K_segment_without_filtering_luad',
        # 'Mixed_N100K_segment_filtered',  # filtered by both cell-level and gene-level
        # 'Mixed_N100K_segment_filtered-SCT_POS_N100',  # filtered by both cell-level and gene-level

        f'Mixed_N100K_segment_without_filtering_{single_cancer_type}',  # not filtered by both GEP-level and gene-level
        # 'Mixed_N100K_segment_filtered-Mixed_N100K_segment_filtered_1',
        # 'Mixed_N100K_segment_without_filtering-Mixed_N100K_segment_without_filtering_1',
        # 'Mixed_N100K_segment_filtered_1-Mixed_N100K_segment_without_filtering_1',
        # 'Mixed_N100K_segment_filtered_2-Mixed_N100K_segment_without_filtering',
        # 'Mixed_N100K_segment_filtered_3-Mixed_N100K_segment_without_filtering',
        # 'Mixed_N100K_segment_filtered_4-Mixed_N100K_segment_without_filtering',

        ]
    test_set_list = ['Test_set0', 'Test_set1', 'Test_set2']
    sct_test_set = 'SCT_POS_N100_test'
    original_test_set_dir = os.path.join(deside_data_dir, 'simulated_bulk_cell_dataset_subtypes_all_range')
    dataset2path['Test_set0'] = os.path.join(original_test_set_dir, 'random_12ds_n_base100',
                                             'simu_bulk_exp_Test_set0_log2cpm1p.h5ad')
    dataset2path['Test_set1'] = os.path.join(original_test_set_dir,
                                             'segment_12ds_0.95_n_base100_19cancer_pca_0.9_median_gep',
                                             'simu_bulk_exp_Test_set1_log2cpm1p.h5ad')
    dataset2path['Test_set2'] = os.path.join(original_test_set_dir,
                                             'segment_12ds_no_filtering_n_base100_median_gep',
                                             'simu_bulk_exp_Test_set2_log2cpm1p.h5ad')
    evaluation_dataset2path = {k: v for k, v in dataset2path.items() if k in test_set_list}
    evaluation_dataset2path[sct_test_set] = os.path.join(sc_dataset_dir,
                                                         f'simu_bulk_exp_{sct_test_set}_log2cpm1p.h5ad')
    name_mapping = {'tcga': 'TCGA', 'Mixed_N100K_segment_filtered': 'D1',
                    'Mixed_N100K_segment_filtered_1': 'D1_1',
                    'Mixed_N100K_segment_filtered_2': 'D1_2',
                    'Mixed_N100K_segment_filtered_3': 'D1_3',
                    'Mixed_N100K_segment_filtered_4': 'D1_4',
                    'Mixed_N100K_segment_without_filtering': 'D2',
                    'Mixed_N100K_segment_without_filtering_1': 'D2_1',
                    'Mixed_N100K_segment_without_filtering_2': 'D2_2',
                    'Mixed_N100K_segment_without_filtering_3': 'D2_3',
                    'Mixed_N100K_segment_without_filtering_4': 'D2_4',
                    'Mixed_N30K_segment_without_filtering': 'Mix_N30K_SNF',
                    'Mixed_N30K_segment': 'Mix_N30K_SF',
                    'Mixed_N100K_random': 'D0_100K',
                    'Mixed_N100K_random_1': 'D0_1',
                    'Mixed_N100K_random_2': 'D0_2',
                    'Mixed_N100K_random_3': 'D0_3',
                    'Mixed_N100K_random_4': 'D0_4',
                    'SCT_POS_N100': 'T_SCT',
                    'Test_set0': 'T0', 'Test_set1': 'T1', 'Test_set2': 'T2', 'Test_set3': 'T3'}

    train_ds2path = {}
    for ds in training_set_list:
        print(f'>>> Adding {ds} to training set')
        for tsn in ds.split('-'):  # training set name
            if 'SCT' in tsn:  # a smaller SCT datasetc
                simu_bulk_exp_dir = sc_dataset_dir
                # tsfp = os.read_count_file_path.join(simu_bulk_exp_dir, f'simu_bulk_exp_{tsn}_log2cpm1p.h5ad')
            else:
                sampling_method = 'random'
                if 'segment' in tsn:
                    sampling_method = 'segment'
                simu_bulk_exp_dir = sampling_method2dir[sampling_method]
                if sampling_method in ['segment']:
                    if 'filtered' in tsn:
                        _postfix_filtered_ds_naming = f'_{len(cancer_types_for_filtering)}cancer'
                        if filtering_in_pca_space:
                            _postfix_filtered_ds_naming += f'_pca_{pca_n_components}'
                        simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets,
                                                                     gep_filtering_quantile[1],
                                                                     str(n_base) + _postfix_filtered_ds_naming)
                    else:  # without filtering
                        simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets,
                                                                     'no_filtering', n_base)
                else:
                    simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets, n_base)
            tsfp = os.path.join(simu_bulk_exp_dir, f'simu_bulk_exp_{tsn}_log2cpm1p.h5ad')
            if '_filtered' in tsn:
                _tsn = tsn.replace('_filtered', '')
                # using the dataset filtered by both gene-level and GEP-level
                tsfp = os.path.join(simu_bulk_exp_dir,
                                    f'simu_bulk_exp_{_tsn}_log2cpm1p.h5ad').replace('.h5ad', replace_by)
                # using the dataset filtered by only GEP-level
                # tsfp = os.read_count_file_path.join(simu_bulk_exp_dir, f'simu_bulk_exp_{_tsn}_log2cpm1p.h5ad')

            train_ds2path[tsn] = tsfp

    print(train_ds2path)
    # start to training DNN model
    for inx, training_set_name in enumerate(training_set_list):
        if len(training_set_name.split('-')) >= 3:
            training_set_name_abbr = '-'.join([name_mapping[i] for i in training_set_name.split('-')])
        else:
            training_set_name_abbr = training_set_name
        result_dir = os.path.join(result_root_dir, current_project_id, training_set_name_abbr)
        check_dir(result_dir)
        model_dir = os.path.join(result_dir, 'DeSide_model')
        all_vars = {**globals()}
        save_key_params(all_vars=all_vars)

        pred_cell_frac_tcga_dir = os.path.join(result_dir, 'predicted_cell_fraction_tcga')

        print_msg(f'Step2: Training model on dataset {training_set_name}...', log_file_path=log_file_path)
        training_set_file_path = [train_ds2path[tsn] for tsn in training_set_name.split('-')]
        filtered_gene_list = None

        if 'filtered' in training_set_name:
            filtering_method = training_set_name.split('_')[-2]
            if input_gene_list == 'filtered_genes' and f'{filtering_method}_filtered' in training_set_name:
                filtered_gene_file_path = os.path.join(os.path.dirname(training_set_file_path[0]),
                                                       f'Mixed_N100K_{filtering_method}',
                                                       'gene_list_filtered_by_high_corr_gene_and_quantile_range.csv')
                filtered_gene_list = pd.read_csv(filtered_gene_file_path, index_col=0).index.tolist()

        model_obj = DeSide(model_dir=model_dir, log_file_path=log_file_path)
        model_obj.train_model(training_set_file_path=training_set_file_path, hyper_params=deside_parameters,
                              cell_types=all_cell_types, scaling_by_constant=True, scaling_by_sample=False,
                              n_patience=100, remove_cancer_cell=remove_cancer_cell, n_epoch=3000,
                              pathway_mask=pathway_mask, method_adding_pathway=method_adding_pathway,
                              filtered_gene_list=filtered_gene_list, input_gene_list=input_gene_list)
        # Step3, evaluation on test set
        run_step3(evaluation_dataset2path=evaluation_dataset2path, log_file_path=log_file_path, result_dir=result_dir,
                  model_dir=model_dir, all_cell_types=all_cell_types,
                  pathway_mask=pathway_mask, method_adding_pathway=method_adding_pathway,
                  hyper_params=deside_parameters)

        # Step4, test on TCGA dataset
        run_step4(tcga_data_dir=tcga_data_dir, cancer_types=cancer_types, log_file_path=log_file_path,
                  model_dir=model_dir, marker_gene_file_path=marker_gene_file_path, result_dir=result_dir,
                  pred_cell_frac_tcga_dir=pred_cell_frac_tcga_dir, cancer_purity_file_path=cancer_purity_file_path,
                  all_cell_types=all_cell_types, model_names=['DeSide'], outlier_file_path=outlier_tcga_file_path,
                  signature_score_method=signature_score_method, update_figures=False,
                  pathway_mask=pathway_mask, method_adding_pathway=method_adding_pathway,
                  hyper_params=deside_parameters, cell_type2subtypes=cell_type2subtypes)
        # plot the correlation between predicted cancer cell proportions and CPE
        pred_cell_frac_file_path = \
            os.path.join(pred_cell_frac_tcga_dir, 'DeSide', f'all_predicted_cell_fraction_by_DeSide.csv')
        plot_pred_cell_prop_with_cpe(pred_cell_prop_file_path=pred_cell_frac_file_path,
                                     cpe_file_path=cancer_purity_file_path, result_dir=pred_cell_frac_tcga_dir)

        # plot training set and TCGA in latent space
        ds_l_first_file_path = os.path.join(result_dir, 'all_dataset_l_first_values.csv')
        ds_l_last_file_path = os.path.join(result_dir, 'all_dataset_l_last_values.csv')

        # plot the latent space of the first hidden layer and the last hidden layer
        if not os.path.exists(ds_l_first_file_path) or not os.path.exists(ds_l_last_file_path):
            training_set2path = dict(zip(training_set_name.split('-'), training_set_file_path))

            all_dataset2path = {**training_set2path, 'tcga': tcga_merged_tpm_file_path}
            deside_model = model_obj.get_model()
            # the first hidden layers of part one and part two
            if deside_parameters['pathway_network']:
                deside_model_dense_l_first = tf.keras.Model(inputs=deside_model.inputs,
                                                            outputs=[deside_model.layers[1].output,
                                                                     deside_model.layers[7].output])
            else:
                deside_model_dense_l_first = tf.keras.Model(inputs=deside_model.inputs,
                                                            outputs=[deside_model.layers[1].output])
            # deside_model_dense_l_first = tf.keras.Model(inputs=deside_model.inputs,
            deside_model_dense_l_last = tf.keras.Model(inputs=deside_model.inputs,
                                                       outputs=deside_model.layers[-2].output)  # the last hidden layer

            print(deside_model_dense_l_first.summary())
            print(deside_model_dense_l_last.summary())
            l_first, l_last = [], []
            for dataset_name, dataset_fp in all_dataset2path.items():
                print(f'Getting the latent space values for dataset {dataset_name}...')
                if dataset_name == 'tcga':
                    exp_type = 'TPM'
                else:
                    exp_type = 'log_space'
                x = model_obj.get_x_before_predict(input_file=dataset_fp, exp_type=exp_type,
                                                   scaling_by_constant=True, scaling_by_sample=False,
                                                   pathway_mask=pathway_mask,
                                                   method_adding_pathway=method_adding_pathway)
                if x.shape[0] > 100000:
                    x = x.sample(100000)
                x_index = x.index.copy()
                pathway_network = deside_parameters['pathway_network']
                x = get_x_by_pathway_network(x=x, pathway_network=pathway_network, pathway_mask=pathway_mask)
                x_dense_l_first = deside_model_dense_l_first.predict(x)
                x_dense_l_last = deside_model_dense_l_last.predict(x)
                if type(x_dense_l_first) is list and len(x_dense_l_first) == 2:
                    x_dense_l_first = np.concatenate(x_dense_l_first, axis=1)
                x_dense_l_first_df = pd.DataFrame(
                    x_dense_l_first, index=x_index,
                    columns=['x_dense_l_first_' + str(i) for i in range(x_dense_l_first.shape[1])]
                )
                x_dense_l_last_df = pd.DataFrame(
                    x_dense_l_last, index=x_index,
                    columns=['x_dense_l_last_' + str(i) for i in range(x_dense_l_last.shape[1])]
                )
                x_dense_l_first_df['dataset_name'] = dataset_name
                x_dense_l_last_df['dataset_name'] = dataset_name
                l_first.append(x_dense_l_first_df)
                l_last.append(x_dense_l_last_df)
            all_l_first = pd.concat(l_first, axis=0)
            all_l_last = pd.concat(l_last, axis=0)
            all_l_first.to_csv(ds_l_first_file_path, index_label='index')
            all_l_last.to_csv(ds_l_last_file_path, index_label='index')
        else:
            print(f'{ds_l_first_file_path} and {ds_l_last_file_path} already exist, skip this step.')
        all_l_first = pd.read_csv(ds_l_first_file_path, index_col=['index', 'dataset_name'])
        all_l_last = pd.read_csv(ds_l_last_file_path, index_col=['index', 'dataset_name'])
        n_components = 2

        # set_fig_style()
        for lx, all_l_df in [('l_first', all_l_first), ('l_last', all_l_last)]:
            for model_type in ['pca']:
                print(f'Plotting {lx} latent space for {model_type}...')
                model_file_path = os.path.join(result_dir, f'{lx}_{model_type}.joblib')
                lower_dim_file_path = os.path.join(result_dir, f'{lx}_{model_type}_lower_dim.csv')
                if model_type == 'pca':
                    model = do_pca_analysis(all_l_df.values, n_components=n_components,
                                            pca_result_fp=model_file_path)
                else:  # umap
                    model = do_umap_analysis(all_l_df.values, n_components=n_components,
                                             umap_model_result_fp=model_file_path, min_dist=0.1, n_neighbors=15)
                if not os.path.exists(lower_dim_file_path):
                    model.fit(all_l_df.values)
                    all_lx_lower_dim = model.transform(all_l_df.values)
                    label_name = 'PC' if model_type == 'pca' else 'UMAP'
                    all_lx_lower_dim_df = pd.DataFrame(all_lx_lower_dim, index=all_l_df.index,
                                                       columns=[f'{label_name}{i+1}' for i in range(n_components)])
                    all_lx_lower_dim_df['class'] = all_l_df.index.get_level_values('dataset_name')
                    all_lx_lower_dim_df['class'] = all_lx_lower_dim_df['class'].map(name_mapping).to_list()
                    all_lx_lower_dim_df.to_csv(lower_dim_file_path, index_label=['index', 'dataset_name'])
                else:
                    print(f'{lower_dim_file_path} already exists, skip this step.')
                    all_lx_lower_dim_df = pd.read_csv(lower_dim_file_path, index_col=['index', 'dataset_name'])
                fig_file_path = os.path.join(result_dir, f'all_dataset_{model_type}_{lx}_new.png')
                if model_type == 'pca':
                    plot_pca(all_lx_lower_dim_df, explained_variance_ratio=model.explained_variance_ratio_,
                             result_fp=fig_file_path, anno=lx, show_core_zone_of_tcga=True, figsize=(3.5, 3.5), s=2)
                elif model_type == 'umap':
                    plot_pca(all_lx_lower_dim_df, result_fp=fig_file_path, label_name='UMAP', anno=lx,
                             show_core_zone_of_tcga=True, figsize=(3.5, 3.5), s=2)
        print('Results are saved in: ' + result_dir)
    print_msg('All Done!', log_file_path=log_file_path)
