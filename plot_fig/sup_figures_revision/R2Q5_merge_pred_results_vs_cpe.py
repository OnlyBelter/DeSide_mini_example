import pandas as pd
import os
from deside.utility import check_dir
from deside.plot import plot_pred_cell_prop_with_cpe

if __name__ == '__main__':
    # Load the prediction results
    pred_results_dir = './Yerong/PNAS_3_algorithm_data_240616/3_algo_unified_predict-R2Q5'
    scaden_results_dir = './R2Q6/scaden/scRNAseq_ds_read_counts'
    cpe_file_path = '../datasets/cancer_purity/cancer_purity.csv'
    algos = [
        # 'CIBERSORTX',
        # 'MuSiC',
        # 'EPIC',
        'Scaden'
    ]
    datasets = ["Geistlinger2020_Ovarian_3CA",
                "gbm_abdelfattah_12",
                "prad_cheng_08",
                "HNSCC_Kurten2021_3CA",
                "luad_kim_05",
                "scaden_ascites_OV",
                "HNSCC_CIBERSORTX_example"]
    all_cell_types = {}
    cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PRAD',
                    'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC', 'OV']
    cell_type_name_mapping = {'Malignant': 'Cancer Cells', 'T cells CD8': 'CD8 T', 'T cells CD4': 'CD4 T',
                              'Fibroblast': 'Fibroblasts', 'Macrophage': 'Macrophages', 'B cell': 'B Cells',
                              'Mast': 'Mast Cells', 'Dendritic': 'DC', 'Endothelial': 'Endothelial Cells',
                              'cellFractions.B.Cells': 'B Cells',
                              'cellFractions.Endothelial.Cells': 'Endothelial Cells',
                              'cellFractions.Fibroblasts': 'Fibroblasts', 'cellFractions.Macrophages': 'Macrophages',
                              'cellFractions.NK': 'NK', 'cellFractions.T.Cells': 'T Cells',
                              'cellFractions.CD4.T': 'CD4 T',
                              'cellFractions.CD8.T': 'CD8 T', 'cellFractions.DC': 'DC',
                              'cellFractions.Mast.Cells': 'Mast Cells',
                              'cellFractions.Neutrophils': 'Neutrophils', 'cellFractions.otherCells': 'Cancer Cells',
                              'cellFractions.Monocytes': 'Monocytes', 'Carcinoma': 'Cancer Cells'}
    for algo in algos:
        for ds in datasets:
            print('Dealing with:', algo, ds)
            current_dir = os.path.join(pred_results_dir, algo, ds + '_reference_predicted_results')
            result_dir = os.path.join(pred_results_dir, 'merged_results')
            if algo == 'Scaden':
                current_dir = os.path.join(scaden_results_dir, ds, 'predicted_results')
                result_dir = os.path.join(scaden_results_dir, 'merged_pred_results')
            check_dir(result_dir)
            if os.path.exists(current_dir):
                # iterate all files in the directory
                current_results = []
                current_result_file_path = os.path.join(result_dir, algo + '_' + ds + '_pred_results.csv')
                for file in os.listdir(current_dir):
                    if file.endswith('.csv') or file.endswith('.txt'):
                        if algo == 'Scaden':
                            pred_results = pd.read_csv(os.path.join(current_dir, file), index_col=0, sep='\t')
                        else:
                            pred_results = pd.read_csv(os.path.join(current_dir, file), index_col=0)
                        if algo == 'CIBERSORTX':
                            pred_results.drop(columns=['P-value', 'Correlation', 'RMSE'], inplace=True)
                        if algo in ['MuSiC', 'EPIC']:
                            pred_results.index = pred_results.index.str.replace('.', '-', regex=False)
                        pred_results.rename(columns=cell_type_name_mapping, inplace=True)
                        for cell_type in pred_results.columns:
                            if cell_type not in all_cell_types:
                                all_cell_types[cell_type] = 1
                            else:
                                all_cell_types[cell_type] += 1
                        if algo == 'Scaden':
                            cancer_type = file.replace('.txt', '').split('_')[-1]
                        else:
                            cancer_type = file.replace('.csv', '')
                        cancer_type = cancer_type.strip()
                        pred_results['cancer_type'] = cancer_type
                        pred_results['algo'] = algo
                        pred_results['reference_dataset'] = ds
                        current_results.append(pred_results)
                current_results_df = pd.concat(current_results)
                current_results_df['sample_id'] = current_results_df.index
                current_results_df.reset_index(inplace=True, drop=True)
                current_results_df.to_csv(current_result_file_path)
                plot_pred_cell_prop_with_cpe(pred_cell_prop_file_path=current_result_file_path,
                                             cpe_file_path=cpe_file_path,
                                             result_dir=result_dir,
                                             all_cancer_types=cancer_types,
                                             algo=algo,
                                             dataset=ds)

    print(all_cell_types)
