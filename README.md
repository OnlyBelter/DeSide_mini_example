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
|-- datasets  # three large files need to be downloaded separately
`-- results
```

### Dependencies
- `DeSide` is needed to reproduce the results. Please find the installation instructions about [DeSide](https://github.com/OnlyBelter/DeSide).

- Three files that are larger than 100MB in the folder `datasets` are not uploaded to GitHub. Please download them and unzip them to the right place.
  - `simu_bulk_exp_Mixed_N100K_D1.h5ad`: the synthesized bulk gene expression profile (GEP) dataset after filtering (Dataset D1), which is used in the `example 2` as the training dataset. [Download link](https://doi.org/10.6084/m9.figshare.23047391.v1) (~1.5G)
  - `simu_bulk_exp_SCT_N10K_S1.h5ad`: the synthesized single-cell GEP dataset (scGEP, Dataset S1), which is used in the `example 3` as the source of single-cell GEPs. [Download link](https://doi.org/10.6084/m9.figshare.23043560.v1) (~2.8G)
  - `merged_tpm.csv`: gene expression profiles of 19 cancer types in TCGA (TPM format), which is used as the reference dataset to guild the filtering steps in the `example 3`. [Download link](https://doi.org/10.6084/m9.figshare.23047547.v1) (~300M)

#### Folder structure of `datasets`:
```text
datasets
|-- TCGA
|   `-- tpm
|       |-- LUAD
|       |   `-- LUAD_TPM.csv
|       |-- merged_tpm.csv # merged TPM of 19 cancer types
|       `-- tcga_sample_id2cancer_type.csv
|-- simu_bulk_exp_SCT_N10K_S1.h5ad # Dataset S1
`-- simulated_bulk_cell_dataset
    |-- D2
    |   |-- corr_cell_frac_with_gene_exp_D2.csv
    |   `-- gene_list_filtered_by_high_corr_gene.csv # the list of high correlation genes
    `-- simu_bulk_exp_Mixed_N100K_D1.h5ad # Dataset D2
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
