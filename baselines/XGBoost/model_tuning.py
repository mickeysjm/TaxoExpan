"""
__author__: Jiaming Shen
__description__: Tune XGBoost model to obtain a good combination of hyper-parameters
"""
import xgboost as xgb
import time
import argparse 

def main(args):
    dtrain = xgb.DMatrix(args.train)
    dval = xgb.DMatrix(args.validation)

    # default XGBoost hyperparameters, c.f.: https://xgboost.readthedocs.io/en/latest/parameter.html
    default_param = {
        'objective': 'binary:logistic',  # keep this fixed
        'max_depth': 6, 
        'min_child_weight': 5,
        'gamma': 0.0,
        "subsample": 0.5,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 0.1,
        'eta': 0.1,
        'importance_type': 'weight',
        'random_state': 0,
        'nthread': 20,
        'tree_method': "hist",
        'eval_metric': ['error', 'auc']
    }

    # indicate the hyper-parameters to be tuned
    # each row indicates a hyper-parameter name and its possible value range
    tuned_param = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [0, 3, 5, 10, 15, 20, 50],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10],
        'eta': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        'random_state': [0, 7, 19, 47, 147, 940622],
        'importance_type': ["gain", "weight", "cover", "total_gain", "total_cover"]
    }

    # start tuning by hill-climbing
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    start = time.time()
    cnt = 0
    num_round = 100
    early_stopping_rounds = 10
    verbose_eval = 10  # larger this value, less verbose the model is 
    for param_name in tuned_param:
        print(f"===================== Tunning param: {param_name} =====================")
        param_range = tuned_param[param_name]
        best_metric = -1e10  # assume we want to maximize a metric
        best_param_value = None
        for param_value in param_range:
            cnt += 1
            default_param[param_name] = param_value
            bst = xgb.train(default_param, dtrain=dtrain, num_boost_round=num_round, evals=evallist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
            cur_metric = bst.best_score
            if cur_metric >= best_metric:
                best_metric = cur_metric
                best_param_value = param_value
        default_param[param_name] = best_param_value
        print(f"!!! Best param_value: {best_param_value}")
    end = time.time()
    
    print(f"!!! Finish tuning {cnt} parameter combinations, using {end-start} seconds")
    print(f"Best score: {best_metric}")
    print(f"Best parameters:")
    for param_name in default_param:
        print(f"\t{param_name}: {default_param[param_name]}")
    print(f"Please save the above hyper-parameters and manually copy to model_training.py script for learning the final model")

if __name__ == "__main__":
    # Example:
    #   --train: "/datadrive/structure_expan/data/MAG_FoS/mag_cs_train_1102.buffer"
    #   --validation: "/datadrive/structure_expan/data/MAG_FoS/mag_cs_validation_1102.buffer"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=str, help='training data file path')
    parser.add_argument('--validation', required=True, type=str, help='validation data file path')
    args = parser.parse_args()
    main(args)
