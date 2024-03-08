# README for Cancer Model Benchmark Results

This collection of directories and files contains the benchmarking results of three DL models applied to five different cancer datasets. The models' training and testing sets follow the 'GDSC2_EGFR_PI3K_MAPK_Top5cancers' data folder.

## Directory Structure

- {model_name} (e.g. DeepCDR)
  - breast.csv
  - COAD.csv
  - LUAD.csv
  - melanoma.csv
  - SCLC.csv


## Model Descriptions

1. GraphDRP: GNN & Mutation CNN

2. DeepCDR: UGCN & Mutation + CNA +  Methylation CNN

3. NeRD: GNN & Mutation CNN + CNA MLP
## Cancer Types

Each CSV file within the model directories contains data for a specific type of cancer:
- Breast Cancer (breast.csv)
- Colon Adenocarcinoma (COAD.csv)
- Lung Adenocarcinoma (LUAD.csv)
- Melanoma (melanoma.csv)
- Small Cell Lung Cancer (SCLC.csv)

## File Contents

Each CSV file contains structured data with benchmarking metrics mean absolute error (**test_mae**), and root mean square error (**test_rmse**) for model predictions on the respective cancer dataset. 

Six different numbers of training samples and six seeds are repeated.

Three metrics for drug response: 
- AUC
- Emax
- IC50

The columns '**metric_num_seed**' indicates the experiment information:
- e.g. AUC_8_1: AUC testing result with 8 training samples by seed 1.

Please select '**test_mae**' column for AUC and Emax while '**test_rmse**' for IC50.
