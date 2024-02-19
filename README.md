# DeSide_mini_example
Minimal examples demonstrating the usage of DeSide

#### Folder structure of `DeSide_mini_example`:
```text
DeSide_mini_example
├── DeSide_model  # the pre-trained model, one large file need to be downloaded separately
├── E1 - Using pre-trained model.ipynb
├── E2 - Training a model from scratch.ipynb
├── E3 - Synthesizing bulk tumors.ipynb
├── LICENSE
├── README.md
├── datasets  # three large files need to be downloaded separately
├── results   # the results of the three examples
├── plot_fig  # the figures and relevant data in the manuscript
├── main_workflow_demo.py  # the main workflow of the manuscript, only for achieving the code
└── single_cell_dataset_integration  # the single-cell dataset used in the manuscript
```

### Dependencies
- `DeSide` is needed to reproduce the results. Please find the installation instructions about [DeSide](https://github.com/OnlyBelter/DeSide).

- Three files larger than 100MB in the `datasets` folder are not uploaded to GitHub. Please download and unzip them to the right place.
  - `simu_bulk_exp_Mixed_N100K_D1.h5ad`: the synthesized bulk gene expression profiles (GEPs) after filtering (Dataset D1), which is used in the `example 2` as the training dataset. [Download link](https://doi.org/10.6084/m9.figshare.23047391.v2) (~2.2G)
  - `simu_bulk_exp_SCT_N10K_S1_16sct.h5ad`: the synthesized single-cell-type GEPs (sctGEPs, Dataset S1), which is used in the `example 3` as the source of single-cell GEPs for simulation. [Download link](https://doi.org/10.6084/m9.figshare.23043560.v2) (~7G)
  - `merged_tpm.csv`: gene expression profiles of 19 cancer types in TCGA (TPM format), which is used as the reference dataset to guild the filtering steps in the `example 3`. [Download link](https://doi.org/10.6084/m9.figshare.23047547.v2) (~300M)

#### Folder structure of `datasets`:
```text
datasets
├── TCGA
│ ├── pca_model_0.9  # the PCA model fitted by the TCGA dataset for GEP-level filtering
│ │ ├── gene_list_for_pca.csv
│ │ ├── tcga_pca_model_for_gep_filtering.pkl  # generated during dataset generation
│ │ └── tcga_pca_ref.csv
│ └── tpm
│     ├── LUAD
│     │ └── LUAD_TPM.csv
│     ├── merged_tpm.csv # merged TPM of 19 cancer types (need to be downloaded separately)
│     └── tcga_sample_id2cancer_type.csv
├── gene_set  # used as the pathway profiles
│ ├── c2.cp.kegg.v2023.1.Hs.symbols.gmt
│ └── c2.cp.reactome.v2023.1.Hs.symbols.gmt
├── simu_bulk_exp_SCT_N10K_S1_16sct.h5ad # Dataset S1 (need to be downloaded separately)
└── simulated_bulk_cell_dataset
    ├── D1
    │ ├── gene_list_filtered_by_high_corr_gene_and_quantile_range.csv  # gene list after gene-level filtering (different datasets can generate this gene list slightly differently)
    │ ├── gene_list_filtered_by_quantile_range_q_0.5_q_99.5.csv
    │ └── simu_bulk_exp_Mixed_N100K_D1.h5ad # Dataset D1 (need to be downloaded separately)
    └── D2
        ├── corr_cell_frac_with_gene_exp_D2.csv
        └── gene_list_filtered_by_high_corr_gene.csv # the list of high correlation genes (the same one used for the filtering step in other datasets)
```

- The following file in the folder `DeSide_model` is larger than 100MB and has not been uploaded to GitHub. Please download and put it to the right place.
  - `model_DeSide.h5`: the pre-trained model, which is used in the `example 1`. [Download link](https://doi.org/10.6084/m9.figshare.25117862.v1) (~100M)

#### Folder structure of `DeSide_model`:
```text
DeSide_model
├── celltypes.txt
├── genes.txt
├── genes_for_gep.txt
├── genes_for_pathway_profile.txt
├── history_reg.csv
├── key_params.txt
├── loss.png
└── model_DeSide.h5 # the pre-trained model (need to be downloaded separately)
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
