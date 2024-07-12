import pandas as pd
import os

if __name__ == '__main__':
    # merged_tpm = pd.read_csv('./datasets/TCGA/tpm/merged_tpm.csv', index_col=0)  # sample x gene
    # print('the shape of merged_tpm:', merged_tpm.shape)
    ov_tpm = pd.read_csv('./datasets/TCGA/tpm/OV/OV_TPM.csv', index_col=0).T  # gene x sample -> sample x gene
    # print('the shape of ov_tpm:', ov_tpm.shape)
    # merged_tpm = pd.concat([merged_tpm, ov_tpm], axis=0)
    # print('the shape of merged_tpm:', merged_tpm.shape)
    # merged_tpm.to_csv('./datasets/TCGA/tpm/merged_tpm_with_ov.csv')
    sample2cancer_type = pd.read_csv('./datasets/TCGA/tpm/tcga_sample_id2cancer_type.csv', index_col=0)
    ov_tpm['cancer_type'] = 'OV'
    ov_tpm = ov_tpm.loc[:, ['cancer_type']].copy()
    merged_tpm = pd.concat([sample2cancer_type, ov_tpm], axis=0)
    merged_tpm.to_csv('./datasets/TCGA/tpm/tcga_sample_id2cancer_type_with_ov.csv')
