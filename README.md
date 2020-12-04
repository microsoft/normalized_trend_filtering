# Normalized Trend Filtering for Biomedical Datasets

# Requirements
Install required modules with:
```
bash
conda create -n NTF python=3.7 numpy=1.18.1 mkl_random=1.1.0
conda activate NTF
pip install -r requirements.txt
```

# Introduction
Associated paper can be found here: 

# Data
Paper and datasets used for empirical study:

* Harel_PD1: [DATA](https://www.cell.com/cms/10.1016/j.cell.2019.08.012/attachment/1ccc78b2-37c7-44f0-a829-cbd81455ea9b/mmc1.xlsx), [PAPER](https://www.cell.com/cell/fulltext/S0092-8674(19)30900-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867419309006%3Fshowall%3Dtrue)

* Harel_TIL: [DATA](https://www.cell.com/cms/10.1016/j.cell.2019.08.012/attachment/1ccc78b2-37c7-44f0-a829-cbd81455ea9b/mmc1.xlsx), [PAPER](https://www.cell.com/cell/fulltext/S0092-8674(19)30900-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867419309006%3Fshowall%3Dtrue)  

* VanAllen: [RNA-seq DATA](https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-019-0654-5/MediaObjects/41591_2019_654_MOESM3_ESM.txt), [RESPONSE](https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-019-0654-5/MediaObjects/41591_2019_654_MOESM4_ESM.xlsx), [PAPER](https://www.nature.com/articles/s41591-019-0654-5)  

* OpenPBTA: [DATA](https://github.com/AlexsLemonade/OpenPBTA-analysis#how-to-obtain-openpbta-data), [PAPER](https://alexslemonade.github.io/OpenPBTA-manuscript/)  
Follow data download instructions from OpenPBTA (ver. release-v9-20191105)  
RNA-seq data is found in "pbta-gene-expression-kallisto.stranded.rds"  
Response is found in "pbta-histologies.tsv"  

* inBiomap Interactome: [DATA](https://inbio-discover.intomics.com/api/data/map_public/2016_09_12/inBio_Map_core_2016_09_12.tar.gz), [PAPER](https://www.nature.com/articles/nmeth.4083)


# Notebooks
Notebooks produce the network figures and Tables 1. and 2. 

DATASET.ipynb contains analysis for Lasso, MCP, SCAD, Ridge, GraphTF, NTF-Lasso, NTF-MCP, NTF-SCAD, Laplacian Ridge, and Shuffled Experiments

DATASET_LapRidge.ipynb contains analysis for Lasso + Laplacian Ridge, MCP + Laplacian Ridge, SCAD + Laplacian Ridge



