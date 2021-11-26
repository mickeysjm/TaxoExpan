# TaxoExpan

The source code used for self-supervised taxonomy expansion method [TaxoExpan](https://arxiv.org/abs/2001.09522), published in WWW 2020.

## Install Guide

### Install DGL 0.4.0 version with GPU suppert using Conda

From following page: [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

```
conda install -c dglteam dgl-cuda10.0
```

## Data Preparation

For dataset used in our WWW paper, you can directly download all input files from [Google Drive](https://drive.google.com/drive/folders/1-_yaDYDbivAW_ZA3em8WTbxDSnfIZfV9?usp=sharing) and skip this section.

For expanding new input taxonomies, you need to read this section and format your datasets accordingly.

### Step 0.a (Required): Organize your input taxonomy along with node features into the following 3 files

**1. <TAXONOMY_NAME>.terms**, each line represents one concept in the taxonomy, including its ID and surface name

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

**2. <TAXONOMY_NAME>.taxo**, each line represents one relation in the taxonomy, including the parent taxon ID and child taxon ID

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

**3. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.embed**, the first line indicates the vocabulary size and embedding dimension, each of the following line represents one taxon with its pretrained embedding

```
<VOCAB_SIZE> <EMBED_DIM>
taxon1_id taxon1_embedding
taxon2_id taxon2_embedding
taxon3_id taxon3_embedding
...
```

The embedding file follows the gensim word2vec format.

Notes:

1. Make sure the <TAXONOMY_NAME> is the same across all the 3 files.
2. The <EMBED_SUFFIX> is used to chooose what initial embedding you will use. You can leave it empty to load the file "<TAXONOMY_NAME>.terms.embed". **Make sure you can generate the embedding for a new given term.**

### Step 0.b (Optional): Generate train/validation/test partition files

You can generate your desired train/validation/test parition files by creating another 3 separated files (named <TAXONOMY_NAME>.terms.train, <TAXONOMY_NAME>.terms.validation, as well as <TAXONOMY_NAME>.terms.test) and puting them in the same directory as the above three required files.

These three partition files are of the same format -- each line includes one taxon_id that appears in the above <TAXONOMY_NAME>.terms file.

### Step 1: Generate the binary dataset file

1. create a folder "./data/{DATASET_NAME}"
2. put the above three required files (as well as three optional partition files) in "./data/{DATASET_NAME}"
3. under this root directory, run

```
python generate_dataset_binary.py \
    --taxon_name <TAXONOMY_NAME> \
    --data_dir <DATASET_NAME> \
    --embed_suffix <EMBED_SUFFIX> \
    --existing_partition 0
```

This script will first load the existing taxonomy (along with initial node features indicated by `embed_suffix`) from the previous three required files.
Then, if `existing_partition` is 0, it will generate a random train/validation/test partitions, otherwise, it will load the existing train/validation/test partition files.
Finally, it saves the generated dataset (along with all initial node features) in one pickle file for fast loading next time.

## Model Training

### Simplest training

Write all the parameters in **./config_files/config.universal.json** and start training.

```
python train.py --config config_files/config.universal.json
```

### Specifying parameters in training command

For example, you can indicate the architectures of graph propagation module, graph readout module, and matching module as follow:

```
python train.py --config config_files/config.universal.json --pm PGAT --rm WMR --mm LBM --device 0
```

Please check **./train.py** for all configurable parameters.

## Model Testing

### Test on default dataset using GPU 0

```
python test_fast.py --resume <MODEL_CHECKPOINT.pth> --device 0
```

### Test on a sperated dataset using GPU 0

```
python test_fast.py --resume <MODEL_CHECKPOINT.pth> --test_data <TEST_DATA.bin> --device 0
```

### Test with case study results saving using GPU 0

```
python test_fast.py --resume <MODEL_CHECKPOINT.pth> --case <OUTPUT_CASE_FILE.tsv> --device 0
```

Note: test with case study saving will almost double the running time, so if you don't really need to see the predicted parents, disable this functionality.

### Test on large dataset (e.g., MAG-Full) using GPU 0

```
python test_fast.py --resume <MODEL_CHECKPOINT.pth> --batch_size 30000 --device 0
```

## Model Inference

Predict on completely new taxons.

### Data Format

We assume the input taxon list is of the following format:

```
term1 \t embeddings of term1
term2 \t embeddings of term2
term3 \t embeddings of term3
...
```

The term can be either a unigram or a phrase, as long as it doesn't contain "\t" character.
The embedding is space-sperated and of the same dimension of trained model.
An input example is provided: **./data/mag_cs_new637.txt**.
An output example is provided: **./case_studies/infer_results_637_20191111.txt**.

### Infer

```
python infer.py --resume <MODEL_CHECKPOINT.pth> --taxon <INPUT_TAXON_LIST.txt> --save <OUTPUT_RESULT.tsv> --device 0
```

The model prediction results are saved in **OUTPUT_RESULT.tsv**.

## Pretrained Models

Pretained models for MAG-CS and MAG-Full datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1-fdImuJDFwJI_shgA91yM-68omZLn1_c?usp=sharing).

## Saved Case Stuides

1. MAG-CS: ./case_studies/mag_cs_case_study_20191111.tsv

2. MAG-Full: ./case_studies/mag_full_case_study_20191111.tsv

## Miscellaneous Notes & Details

### 1. Initial feature vectors for SemEval datasets

1. bert_base_uncased_defonly: the embedding obtained based on definition sentences based on BERT_base_uncased model
2. fasttext_mode1: average fasttext word embedding in definition sentences
3. fasttext_mode2: the average of (1) average fasttext word embedding in definition sentences, and (2) average fasttext word embedding in lemma names
4. fasttext_mode3: the weighted average of (1) average fasttext word embedding in definition sentences (0.75 weight), and (2) average fasttext word embedding in lemma names (0.25 weight)
5. fasttext_mode4: the average of (1) average fasttext word embedding in definition sentences, (2) average fasttext word embedding in lemma names, and (3) average fasttext word embedding of the first word in definition sentence that has the same part-of-speech tag as the lemma's part-of-speech tag. If this same POS word cannot be found, we simply double the weight of definition sentence embedding

The detailed preprocessing code please refer to **./data_preprocessing/semeval-task14.ipython**.

Other possible choices include: (1) general sentence embedding, (2) other bert/xlnet-based embedding, and (3) definition encoder.

### 2. Generate final SemEval evaluation file

For SemEval data, after training the model and running the test script with case study output, you need to read the script `/scripts/parse_to_semeval_format.py` to generate a tsv that satisfies the format requirements in original SemEval completition as follows:

```
python ./scripts/parse_to_semeval_format.py \
    --input <CASE_FILE.tsv> \
    --output <SEMEVAL_FILE.tsv>
```

### 3. Model Organization

For our own TaxoExpan implementation, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).
