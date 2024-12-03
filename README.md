# DKG
This study introduces a novel approach to measuring interdisciplinarity using keyword extraction and graph-based analysis. Keywords, extracted via KeyBERT, are mapped onto a Discipline Knowledge Graph as nodes, where interdisciplinarity scores are computed by using the BFS algorithm. Evaluation with a classification task using the SPECTER model reveals that zero-score papers encounter fewer fields, while interdisciplinary papers show lower classifier confidence, highlighting their complex, boundary-spanning nature. This method offers a content-driven perspective on interdisciplinarity, addressing limitations of journal-based analyses.

![node_edge](https://github.com/user-attachments/assets/058c1cf5-4b57-4353-808a-223e700a7f42)
![graph_search_strucutre](https://github.com/user-attachments/assets/ad0b978e-ab70-4b5d-afeb-71eb04a4b473)

## Paper

## Dataset
The title and abstract texts of papers from the OpenAlex data

## Code

* building_graph.ipynb : Building a knowledge graph
* en_only_df.csv :  the dataset used to build a knowledge graph -> too big to upload in github [link](https://drive.google.com/file/d/1-m9difRJAStlffAPlRwpey0AkhluLC5K/view?usp=sharing)
* en_inter_df.csv : the test dataset for the knowledge graph, and the score of this dataset is measured through the BFS algorithm. -> too big to upload in github [link](https://drive.google.com/file/d/1ealcKmlhqZ8Eg1ClwFJvHATFN3vQKfES/view?usp=sharing)
* data_process.py :  Data preprocessing to remove short texts and non-english texts
* final_test_data.csv : The dataset to predict the classification and model confidence
* final_train_df.csv :  The dataset used to fine-tune the SPECTER model
* specter_dataset.py :  The Dataset class to fine-tune the SPECTER model
* specter_main.py :  The main functions to fine-tune the SPECTER model
* specter_model.py :  The Classifier class to fine-tune the SPECTER model
