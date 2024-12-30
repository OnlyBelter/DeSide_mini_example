# DeSide_mini_example
Minimal examples demonstrating the usage of DeSide

## Getting Started
### 1. Clone the Repository
```bash
# Using SSH
git clone git@github.com:OnlyBelter/DeSide_mini_example.git
# Or using HTTPS
git clone https://github.com/OnlyBelter/DeSide_mini_example.git

cd DeSide_mini_example
```

### 2. Environment Setup
```bash
# Create and activate a new conda environment (recommended)
conda create -n deside python=3.8
conda activate deside

# Install JupyterLab
pip install jupyterlab

# Install DeSide package
pip install deside

# Start JupyterLab
jupyter lab
```

### 3. Download Required Files
Before running the examples, download the following required files:

1. Pre-trained model (Example 1):
   - Download [`model_DeSide.h5`](https://doi.org/10.6084/m9.figshare.25117862.v1) (~100MB)
   - Place in `DeSide_model/` directory

2. Dataset files (Examples 2 and 3):
   - Download the following files and place them in appropriate locations under `datasets/`:
     - [`simu_bulk_exp_Mixed_N100K_D1.h5ad`](https://doi.org/10.6084/m9.figshare.23047391.v2) (~2.2GB) (Example 2)
     - [`simu_bulk_exp_SCT_N10K_S1_16sct.h5ad`](https://doi.org/10.6084/m9.figshare.23043560.v2) (~7GB) (Example 3)
     - [`merged_tpm.csv`](https://doi.org/10.6084/m9.figshare.23047547.v2) (~300MB) (Example 3)


### Project Structure
```text
DeSide_mini_example
├── DeSide_model  # Pre-trained model directory
├── E1 - Using pre-trained model.ipynb
├── E2 - Training a model from scratch.ipynb
├── E3 - Synthesizing bulk tumors.ipynb
├── LICENSE
├── README.md
├── datasets  # Dataset directory
├── results   # Results from examples
├── plot_fig  # Figures and relevant data in the manuscript
├── main_workflow_demo.py  # Main workflow code from manuscript
└── single_cell_dataset_integration  # Single-cell dataset used in the manuscript
```

## Example 1: Using Pre-trained model
This example demonstrates how to use the pre-trained model to predict cell type proportions in a new dataset.

### Required Files
- Pre-trained model file: `DeSide_model/model_DeSide.h5` (~100MB)
  - [Download Link](https://doi.org/10.6084/m9.figshare.25117862.v1)

### Model Directory Structure
```text
DeSide_model
├── celltypes.txt
├── genes.txt
├── genes_for_gep.txt
├── genes_for_pathway_profile.txt
├── history_reg.csv
├── key_params.txt
├── loss.png
└── model_DeSide.h5  # Download required
```

### Tutorial
- See [E1 - Using pre-trained model.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E1%20-%20Using%20pre-trained%20model.ipynb)


## Example 2: Training a Model from Scratch
Learn how to train a DeSide model using synthesized bulk GEP dataset.
### Tutorial
- See [E2 - Training a model from scratch.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E2%20-%20Training%20a%20model%20from%20scratch.ipynb)

### Required File
- Dataset D1: `simu_bulk_exp_Mixed_N100K_D1.h5ad` (~2.2GB)
   - Synthesized bulk gene expression profiles (Dataset D1)
   - Used in Example 2 as training dataset
   - [Download Link](https://doi.org/10.6084/m9.figshare.23047391.v2)
   - Place in `datasets/simulated_bulk_cell_dataset/` directory

## Example 3: Synthesizing Bulk Tumors
Demonstrate the process of synthesizing bulk tumors using DeSide.
### Tutorial
- See: [E3 - Synthesizing bulk tumors.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E3%20-%20Synthesizing%20bulk%20tumors.ipynb)

### Required Files
- Dataset S1: `simu_bulk_exp_SCT_N10K_S1_16sct.h5ad` (~7GB)
   - Synthesized single-cell-type GEPs (sctGEPs)
   - Used in Example 3 as the source of single-cell GEPs for simulation
   - [Download Link](https://doi.org/10.6084/m9.figshare.23043560.v2)
   - Place in `datasets/` directory
- Merged bulk tumor dataset: `merged_tpm.csv` (~300MB)
   - Merged TPM of 19 cancer types from TCGA
   - Used in Example 3 as the reference dataset to guild the filtering steps
   - [Download Link](https://doi.org/10.6084/m9.figshare.23047547.v2)
   - Place in `datasets/TCGA/tpm/` directory

## Dataset Directory Structure
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
│     ├── merged_tpm.csv # Download required
│     └── tcga_sample_id2cancer_type.csv
├── gene_set  # used as the pathway profiles
│ ├── c2.cp.kegg.v2023.1.Hs.symbols.gmt
│ └── c2.cp.reactome.v2023.1.Hs.symbols.gmt
├── simu_bulk_exp_SCT_N10K_S1_16sct.h5ad # Download required
└── simulated_bulk_cell_dataset
    ├── D1
    │ ├── gene_list_filtered_by_high_corr_gene_and_quantile_range.csv  # gene list after gene-level filtering (different datasets can generate this gene list slightly differently)
    │ ├── gene_list_filtered_by_quantile_range_q_0.5_q_99.5.csv
    │ └── simu_bulk_exp_Mixed_N100K_D1.h5ad # Download required
    └── D2
        ├── corr_cell_frac_with_gene_exp_D2.csv
        └── gene_list_filtered_by_high_corr_gene.csv # the list of high correlation genes (the same one used for the filtering step in other datasets)
```

