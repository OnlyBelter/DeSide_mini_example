import os
import shutil
import pandas as pd
from deside.utility import check_dir
from deside.utility.compare import read_and_merge_result
from deside.plot import compare_y_y_pred_plot_cpe, plot_pred_cell_prop_with_cpe

if __name__ == '__main__':
    # Read and merge results
    dataset_dir = '../datasets'
    merged_dir = os.path.join('./R2Q6', 'scaden', 'merged_pred_results')
    cancer_purity_file_path = os.path.join(dataset_dir, 'cancer_purity', 'cancer_purity.csv')
    tcga_sample_id2cancer_type_file_path = os.path.join(dataset_dir, 'TCGA', 'tpm', 'tcga_sample_id2cancer_type.csv')
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    check_dir(merged_dir)

    algos = ['Scaden_GBM', 'Scaden_LUAD', 'Scaden_PRAD', 'Scaden_OV']
    algo2merged_file_path = {i: os.path.join(merged_dir, f'{i}_pred_cell_prop.csv') for i in algos}
    print(algo2merged_file_path)

    pred_dir = './R2Q6/scaden/scRNAseq_ds_read_counts'
    algo2raw_result_dir = {
        'Scaden_GBM': os.path.join(pred_dir, 'gbm_abdelfattah_12', 'predicted_results'),
        'Scaden_LUAD': os.path.join(pred_dir, 'luad_kim_05_gse131907', 'predicted_results'),
        'Scaden_PRAD': os.path.join(pred_dir, 'prad_cheng_08', 'predicted_results'),
        'Scaden_OV': os.path.join(pred_dir, 'ascites_ov', 'predicted_results'),
    }
    ct_in_luad = ["T Cells", "Endothelial Cells", "B Cells", "Mast Cells",
                  "Myeloid cells", "Cancer Cells", "Fibroblasts"]
    ct_in_prad = ['Cancer Cells', 'Fibroblasts',
                  'Epithelial Cells', 'B Cells', 'Macrophages', 'Mast Cells',
                  'T Cells', 'Endothelial Cells']
    ct_in_gbm = ["Mast Cells", "Neutrophils", "NK", "CD4 T", "Cancer Cells", "Monocytes",
                 "B Cells", "Endothelial Cells", "Macrophages", "Double-neg-like T",
                 "Fibroblasts", "CD8 T", "DC"]
    algo2cell_type_name_mapping = {
        'Scaden_GBM': dict(zip(ct_in_gbm, ct_in_gbm)),
        'Scaden_LUAD': dict(zip(ct_in_luad, ct_in_luad)),
        'Scaden_PRAD': {'CD8Tcells': 'CD8 T', 'Monocytes': 'Monocytes', 'Malignant': 'Cancer Cells',
                        'CD4Tcells': 'CD4 T', 'Fibroblast': 'Fibroblasts',
                        'Epithelial': 'Epithelial Cells',
                        'B_cell': 'B Cells',
                        'Macrophage': 'Macrophages',
                        'Mast': 'Mast Cells',
                        'T_cell': 'T Cells',
                        'Endothelial': 'Endothelial Cells',
                        },
        'Scaden_OV': {'CD4Tcells': 'CD4 T', 'CD8Tcells': 'CD8 T', 'Carcinoma': 'Cancer Cells',
                      'DC': 'DC', 'Fibroblast': 'Fibroblasts', 'NK': 'NK',
                      },
    }

    for algo, m_fp in algo2merged_file_path.items():
        cancer_types = None
        print(f'Merge the results of {algo}...')
        if algo == 'Scaden_OV':
            cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PAAD', 'PRAD',
                            'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC', 'OV']
        algo_result_dir = algo2raw_result_dir[algo]
        # iterate all files in the directory
        current_results = []
        for file in os.listdir(algo_result_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(algo_result_dir, file)
                current_df = pd.read_csv(file_path, index_col=0, sep='\t')
                current_df['cancer_type'] = file.replace('.txt', '').split('_')[-1]
                current_results.append(current_df)
        merged_df = pd.concat(current_results, axis=0)
        merged_df['algo'] = algo
        merged_df['sample_id'] = merged_df.index
        merged_df.reset_index(drop=True, inplace=True)
        merged_df.rename(columns=algo2cell_type_name_mapping[algo], inplace=True)
        merged_df.to_csv(m_fp)
        result_dir = os.path.join(merged_dir, algo)
        check_dir(result_dir)
        plot_pred_cell_prop_with_cpe(pred_cell_prop_file_path=m_fp,
                                     cpe_file_path=cancer_purity_file_path, result_dir=result_dir,
                                     all_cancer_types=cancer_types)
