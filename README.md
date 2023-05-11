# DeSide_mini_example
Minimal examples demonstrating the usage of DeSide

#### Folder structure of `DeSide_mini_example`:
```text
DeSide_mini_example
|-- DeSide_model  # the pre-trained model
|-- E1 - Using pre-trained model.ipynb
|-- E2 - Training a model from scratch.ipynb
|-- E3 - Synthesizing bulk tumors.ipynb
|-- LICENSE
|-- README.md
|-- datasets  # please download the datasets from https://figshare.com/account/articles/22801268
`-- results
```

### Dependencies
- `DeSide` is needed to reproduce the results. Please find the installation instructions about [DeSide](https://github.com/OnlyBelter/DeSide).

- The `datasets` folder used in the examples are available at [here](https://figshare.com/account/articles/22801268).

#### Folder structure of `datasets`:
```text
datasets
|-- TCGA
|   `-- tpm
|       |-- HNSC
|       |   |-- HNSC_TPM.csv
|       |   `-- HNSC_TPM.txt
|       |-- LUAD
|       |   |-- LUAD_TPM.csv
|       |   `-- LUAD_TPM.txt
|       |-- merged_tpm.csv  # merged TPM of 19 cancer types
|       `-- tcga_sample_id2cancer_type.csv
|-- cancer_purity
|   `-- cancer_purity.csv  # downloaded from D. Aran, et al., Nat. Commun. (2015).
|-- generated_sc_dataset_7ds_n_base100
|   |-- generated_frac_SCT_POS_N100.csv
|   |-- generated_frac_SCT_POS_N10K.csv
|   |-- simu_bulk_exp_SCT_POS_N100_log2cpm1p.h5ad
|   `-- simu_bulk_exp_SCT_POS_N10K_log2cpm1p.h5ad  # Dataset S1
|-- simulated_bulk_cell_dataset
|   `-- segment_7ds_0.95_n_base100_median_gep
|       |-- D2
|       |   |-- corr_cell_frac_with_gene_exp_D2.csv
|       |   |-- corr_cell_frac_with_gene_exp_D2.xlsx
|       |   `-- gene_list_filtered_by_high_corr_gene.csv  # the list of high correlation genes
|       `-- simu_bulk_exp_Mixed_N100K_segment_log2cpm1p_filtered_by_high_corr_gene_and_quantile_range_q_5.0_q_95.0.h5ad  # dataset D2
`-- single_cell
    |-- gene_list_in_merged_7_sc_datasets.csv
    |-- merged_7_sc_datasets_log2cpm1p.h5ad  # merged 7 single-cell RNA-seq datasets
    `-- merged_7_sc_datasets_sample_info.csv
```



### Example 1: Using pre-trained model
Using the pre-trained model to predict cell type proportions in a new dataset.
- Jupyter notebook: [E1 - Using pre-trained model.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E1%20-%20Using%20pre-trained%20model.ipynb)

### Example 2: Training a model from scratch
Training a model from scratch using the `DeSide` package and the synthesized bulk GEP dataset.
- Jupyter notebook: [E2 - Training a model from scratch.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E2%20-%20Training%20a%20model%20from%20scratch.ipynb)

### Example 3: Synthesizing bulk tumors
Synthesizing bulk tumors using the `DeSide` package.
- Jupyter notebook: [E3 - Synthesizing bulk tumors.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E3%20-%20Synthesizing%20bulk%20tumors.ipynb)
