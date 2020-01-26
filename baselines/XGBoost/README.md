# XGBoost model for self-supervised taxonomy generation

## Step 0: Preparation

First, you need to install the below dependencies.

```
xgboost
gensim=3.7.3
tqdm
torch>=1.0.0
```

Then, you need to obtain the binary taxonomy file with intended data partitions and make sure you have access to the file which contains the **MAGDataset** definition (which is currently under "../../data/dataset.py").

Change the system path at the begining of **self_supervision_generation.py** and **model_prediction.py** to make sure the MAGDataset class is successfully loaded.

## Step 1: Generate Self-supervision Data

```
python self_supervision_generation.py --data <INPUT_DATA_PATH> --mode <train/validation> --output <OUTPUT_DATA_PATH> --neg <NEGATIVE_SAMPLING_SIZE>
```

## Step 2.1: Tuning Model (optional)

You can indicate the initial combination of hyperparameters and a set of to be tuned hyperameters from the variable **default_param** and **tuned_param** in model_tuning.py, respectively.

The tuning process is based on hill-climbing.

After the tuning process, we will print out the best combination of hyperparameters in the command line and you need to manually copy them to model_training.py script for learning the final model.

Such tuning process may take several hours (depends on your data size and how many hyper-parameters you want to tune) and is optional.

```
python model_tuning.py --train <TRAIN_DATA_BUFFER> --validation <VALIDATION_DATA_BUFFER>
```

## Step 2.2: Training Model

You can indicate the model hyper-parameters from the variable **selected_param** in model_training.py.

```
python model_training.py --train <TRAIN_DATA_BUFFER> --validation <VALIDATION_DATA_BUFFER> --output <OUTPUT_MODEL_PATH>
```

## Step 3: Model Prediction

```
python model_prediction --model <MODEL_FILE_PATH> --test_data <TEST_DATA_FILE_PATH> --output_ranks <OUTPUT_RANKING_RESULTS> --retrieval_size <RETRIEVAL_SIZE>
```

Note: The parameter retrieval_size indicate the number of candidate positions to be tested. If this value equals to -1, we will test all positive candidate positions, which could be very slow. Otherwise, we test only top retrieval_size candidates based on the embedding similarities, which is much faster. 