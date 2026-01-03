import optuna
from optuna.integration import MLflowCallback
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config
from models.experiment_manager import ExperimentManager

class HyperparameterTuner:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, experiment_manager: ExperimentManager= None):
        self.config=get_config()
        self.exp_manager=experiment_manager
        self.X_train=X_train
        self.y_train=y_train
        self.X_val=X_val
        self.y_val=y_val
        self.best_params = None
        self.best_score = None
    def objective(self, trial: optuna.Trial) -> float:
        params={
            'n_estimators':trail.suggest_int('n_estimators',50,300),
            'max_depth':trail.suggest_int('max_depth',3,10),
            'learning_rate':trail.suggest_float('learning_rate',0.01,0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        model=XGBRegressor(**params)
        model.fit(self.X_train, self.y_train,
        eval_set=[(self.X_val, self.y_val)],
        verbose=False
        )
        y_pred = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        
        return rmse
    def run_optimization(self, n_trials: int=None, timeout: int=None) -> Dict:
        if n_trials is None:
            n_trials=self.self.config.model.tuning.n_trials
        if timeout is None:
            timeout-self.config.model.tuning.timeout
        study=optuna.create_study(direction="minimize", study_name="xgboost_tuning")
        mlflc=None
        if exp_manager:
            mlflc=MLflowCallback(tracking_uri=self.congif.mlflow.tracking_uri, metric_name="rmse")
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, callbacks=[mlflc] if mlflc else None, show_progress_bar=True)
        self.best_params=study.best_params
        self.best_score=study.best_value    
        #-----------------------------------------------------
        print(f"\n✅ Optimization complete!")
        print(f"   Best RMSE: {self.best_score:.2f}")
        print(f"   Best parameters:")
        for param, value in self.best_params.items():
            print(f"      {param}: {value}")
        #-----------------------------------------------------
        if exp_manager:
            exp_manager.log_params({
            f"best_{k}":v for k,v in self.best_params.items()
            }) 
            self.exp_manager.log_metrics({
                "best_val_rmse": self.best_score,
                "n_trials_run": len(study.trials)
            })
    
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials)
        }

    def get_best_params(self) -> Dict:
        return self.best_params

def tune_and_train(feature_version: str="v1") -> bool:
    try:
        config=get_config()
        if not congig.model.tuning.enabled:
            print("Hyperparameter tuning disabled in config. Using given parameters")
            return True
        exp_manager=ExperimentManager(experment_name="Hyperparameter_tuning")
        with exp_manager.start_experiment(run_name=f"tuning_{feature_version}"):
            exp_manager.set_tags({
                "pipeline": "hyperparameter_tuning",
                "feature_version": feature_version
            })
            feature_dir= Path("data") / "features" / feature_version 
            train_df=pd.read_parquet(feature_dir / "train.parquet")
            val_df=pd.read_parquet(feature_dir / "val.parquet")
            test_df=pd.read_parquet(feature_dir / "test.parquet")
            exclude_cols = ['date', 'sales', 'is_anomaly', 'store_id', 'item_id', 'is_outlier']
            feature_cols= [col for col in train_df.columns if col not in exclude_cols]
            X_train=train_df[feature_cols]
            y_train=train_df['sales']
            X_val=val_df[feature_cols]
            y_val=val_df['sales']

            tuner=HyperparameterTuner(X_train, y_train, X_val, y_val, exp_manager=exp_manager)
            results = tuner.run_optimization()
            #---------------------------------------------------------------
            print("\n" + "=" * 80)
            print("✅ HYPERPARAMETER TUNING COMPLETED!")
            print("=" * 80)
            
            print(f"\n📊 Results:")
            print(f"   Best Validation RMSE: {results['best_score']:.2f}")
            print(f"   Trials completed: {results['n_trials']}")
            
            print(f"\n🔧 Best Hyperparameters:")
            for param, value in results['best_params'].items():
                print(f"   {param}: {value}")
            
            print(f"\n💡 Update config.yaml with these parameters for better results!")
            print(f"🌐 View all trials in MLflow: http://localhost:5000")

            #---------------------------------------------------------------
            
        return True
    except Exception as e:
        print("Hyperparameter tuning failed ")
        print(f"Error: {str(e)}")
        import traceback 
        traceback.print_exc()
        return False

if __name__=="__main__":
    success=tune_and_train(feature_version="v1")
    sys.exit(0 if success else 1)   


