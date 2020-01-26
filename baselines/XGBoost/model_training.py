"""
__author__: Jiaming Shen
__description__: Train XGBoost model using a pre-select combination of hyper-parameters
"""
import xgboost as xgb
import time
import argparse 
import pickle

def main(args):
    dtrain = xgb.DMatrix(args.train)
    dval = xgb.DMatrix(args.validation)
    # For MAG-CS data
    selected_param = {
        'max_depth': 4, 
        'objective': 'binary:logistic',
        'min_child_weight': 10,
        'gamma': 0.5,
        "subsample": 0.5,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'eta': 0.1,
        'importance_type': 'total_cover',
        'random_state': 940622,
        'nthread': 20,
        'tree_method': "hist",
        'eval_metric': ['error', 'auc']
    }
    # # For MAG-Full data
    # selected_param = {
    #     'max_depth': 7, 
    #     'objective': 'binary:logistic',
    #     'min_child_weight': 5,
    #     'gamma': 0.1,
    #     "subsample": 0.5,
    #     'colsample_bytree': 0.8,
    #     'reg_alpha': 1.0,
    #     'reg_lambda': 0.1,
    #     'eta': 0.1,
    #     'importance_type': 'total_cover',
    #     'random_state': 940622,
    #     'nthread': 20,
    #     'tree_method': "hist",
    #     'eval_metric': ['error', 'auc']
    # }
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(selected_param, dtrain=dtrain, num_boost_round=100, evals=evallist, early_stopping_rounds=10, verbose_eval=10)
    with open(args.output, 'wb') as f:
        pickle.dump(bst, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Example:
    #   --train: "/datadrive/structure_expan/data/MAG_FoS/mag_cs_train_1102.buffer"
    #   --validation: "/datadrive/structure_expan/data/MAG_FoS/mag_cs_validation_1102.buffer"
    #   --output: "/datadrive/structure_expan/saved/models/XGBoost/mag_cs_1102.pickle"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=str, help='training data file path')
    parser.add_argument('--validation', required=True, type=str, help='validation data file path')
    parser.add_argument('--output', required=True, type=str, help='output model pickle path')
    args = parser.parse_args()
    main(args)
