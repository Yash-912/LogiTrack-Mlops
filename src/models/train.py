import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config.config import get_config
from models.experiment_manager import ExperimentManager

class ModelTrainer:
    def __init__(self, experiment_manager: ExperimentManager):
        self.config=get_config()
        self.exp_manager=experiment_manager
        self.forecast_model=None
        self.anomaly_model=None
        self.feature_columns=None 

    def load_features(self, feature_version: str="v1") -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        feature_dir=Path("data") / ("features") / feature_version
        train_df=pd.read_parquet(feature_dir / "train.parquet")
        val_df=pd.read_parquet(feature_dir / "val.parquet")
        test_df=pd.read_parquet(feature_dir / "test.parquet")
        return train_df,val_df,test_df
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
        exclude_cols=[
            'date','sales','is_anomaly','store_id','item_id','is_outlier'
        ]
        if self.feature_columns is None:
            self.feature_columns =[col for col in df.columns if col not in exclude_cols]
        X=df[self.feature_columns]
        y=df['sales']
        return X,y 
    def train_forecast_model(self, X_train: pd.DataFrame, y_train: pd.Series,X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBRegressor:
        params=self.config.model.xgboost.dict()
        model=xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        if self.exp_manager:
            self.exp_manager.log_params({
                f"forecast_{k}":v for k,v in params.items()
            })
        self.forecast_model=model
        return model

    def evaluate_forecast_model(self, model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.DataFrame, dataset_name: str="test") -> Dict[str,float]:
        y_pred=model.predict(X)
        rmse=np.sqrt(mean_squared_error(y,y_pred))
        mae=mean_absolute_error(y,y_pred)
        r2=r2_score(y,y_pred)
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        metrics={
            f"{dataset_name}_rmse" :rmse,
            f"{dataset_name}_mae" :mae,
            f"{dataset_name}_r2" :r2,
            f"{dataset_name}_mape" :mape,
            
        }
        if self.exp_manager:
            self.exp_manager.log_metrics(metrics)
        return metrics

    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray,
                        title: str = "Predictions vs Actual",
                        save_path: str = None) -> plt.Figure:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Sales')
        axes[0, 0].set_ylabel('Predicted Sales')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Sales')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Time series sample (first 500 points)
        sample_size = min(500, len(y_true))
        axes[1, 1].plot(range(sample_size), y_true[:sample_size], 
                       label='Actual', alpha=0.7, linewidth=1)
        axes[1, 1].plot(range(sample_size), y_pred[:sample_size], 
                       label='Predicted', alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Sales')
        axes[1, 1].set_title('Sample Predictions (First 500)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"   💾 Saved plot to: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model: xgb.XGBRegressor, 
                               top_n: int = 20,
                               save_path: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            top_n: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get feature importance
        importance = model.feature_importances_
        feature_names = self.feature_columns
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"   💾 Saved feature importance to: {save_path}")
        
        return fig
    
    def train_anomaly_detector(self, X_train: pd.DataFrame, residuals_train: np.ndarray) -> IsolationForest:
        params=self.config.model.isolation_forest.dict()
        model=IsolationForest(**params)
        model.fit(residuals_train.reshape(-1,1))
        if self.exp_manager:
            self.exp_manager.log_params({
                f"anomaly_{k}": v for k, v in params.items()
            })
        self.anomaly_model=model
        return model
    def evaluate_anomaly_detector(self, model: IsolationForest, residuals: np.ndarray, true_anomalies: np.ndarray = None, dataset_name: str = "test") -> Dict[str, float]:
        predictions=model.predict(residuals.reshape(-1,1))
        anomaly_scores = model.score_samples(residuals.reshape(-1, 1))
        predicted_anomalies = (predictions == -1).astype(int)
        metrics = {
            f"{dataset_name}_anomaly_count": int(predicted_anomalies.sum()),
            f"{dataset_name}_anomaly_rate": float(predicted_anomalies.mean()),
        }
        if true_anomalies is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(true_anomalies, predicted_anomalies, zero_division=0)
            recall = recall_score(true_anomalies, predicted_anomalies, zero_division=0)
            f1 = f1_score(true_anomalies, predicted_anomalies, zero_division=0)
            
            metrics.update({
                f"{dataset_name}_anomaly_precision": precision,
                f"{dataset_name}_anomaly_recall": recall,
                f"{dataset_name}_anomaly_f1": f1
            })
            
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
    
        if self.exp_manager:
            self.exp_manager.log_metrics(metrics)
        
        return metrics
    def save_model(self, output_dir: str = "models/trained") -> Tuple[str,str]:
        output_path=Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        forecast_path=output_path / "forecast_model.joblib"
        anomaly_path = output_path / "anomaly_model.joblib"
        joblib.dump(self.forecast_model, forecast_path)
        joblib.dump(self.anomaly_model, anomaly_path)
        return str(forecast_path), str(anomaly_path)


def train_models(feature_version: str = "v1.0", experiment_name: str = "model_training") -> bool:
    """
    Train all models (forecast and anomaly detection).
    
    Args:
        feature_version: Version of features to use
        experiment_name: Name for the MLflow experiment
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config = get_config()
        exp_manager = ExperimentManager(experiment_name=experiment_name)
        trainer = ModelTrainer(exp_manager)
        
        with exp_manager.start_run(run_name=f"training_{feature_version}"):
            exp_manager.set_tags({
                "pipeline": "model_training",
                "feature_version": feature_version,
                "model_type": config.model.forecasting_model
            })
            
            train_df, val_df, test_df = trainer.load_features(feature_version)
            X_train, y_train = trainer.prepare_data(train_df)
            X_val, y_val = trainer.prepare_data(val_df)
            X_test, y_test = trainer.prepare_data(test_df)

            forecast_model = trainer.train_forecast_model(X_train, y_train, X_val, y_val)
            train_metrics = trainer.evaluate_forecast_model(forecast_model, X_train, y_train, dataset_name="train")
            val_metrics = trainer.evaluate_forecast_model(forecast_model, X_val, y_val, dataset_name="val")
            test_metrics = trainer.evaluate_forecast_model(forecast_model, X_test, y_test, dataset_name="test")

            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Predictions plot
            y_pred_test = forecast_model.predict(X_test)
            pred_fig = trainer.plot_predictions(
                y_test, y_pred_test,
                title="Test Set: Predictions vs Actual",
                save_path=str(plots_dir / "predictions.png")
            )
            exp_manager.log_figure(pred_fig, "predictions.png")
            plt.close()

            importance_fig = trainer.plot_feature_importance(
                forecast_model,
                top_n=20,
                save_path=str(plots_dir / "feature_importance.png")
            )
            exp_manager.log_figure(importance_fig, "feature_importance.png")
            plt.close()

            y_pred_train = forecast_model.predict(X_train)
            residuals_train = y_train.values - y_pred_train
            
            anomaly_model = trainer.train_anomaly_detector(X_train, residuals_train)
        
            residuals_test = y_test.values - y_pred_test
            true_anomalies = test_df['is_anomaly'] if 'is_anomaly' in test_df.columns else None
            
            anomaly_metrics = trainer.evaluate_anomaly_detector(
                anomaly_model, residuals_test, true_anomalies, "test"
            )
            forecast_path, anomaly_path = trainer.save_model()

            exp_manager.log_artifact(forecast_path, "models")
            exp_manager.log_artifact(anomaly_path, "models")
            exp_manager.log_model(
                model=forecast_model,
                artifact_path="forecast_model",
                registered_model_name="supply_chain_forecaster"
            )
            
            print(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print(f"\n📊 Test Set Results:")
            print(f"   RMSE: {test_metrics['test_rmse']:.2f}")
            print(f"   MAE:  {test_metrics['test_mae']:.2f}")
            print(f"   R²:   {test_metrics['test_r2']:.4f}")
            print(f"   MAPE: {test_metrics['test_mape']:.2f}%")
            
            print(f"\n🔍 Anomaly Detection:")
            print(f"   Detected: {anomaly_metrics['test_anomaly_count']} anomalies")
            print(f"   Rate: {anomaly_metrics['test_anomaly_rate']*100:.2f}%")
            
            print(f"\n💾 Models saved to: models/trained/")
            print(f"📊 Plots saved to: plots/")
            print(f"🌐 View in MLflow: http://localhost:5000")

            return True
    except Exception as e:
        print(f"\n❌ Model training failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    
if __name__ == "__main__":
    success = train_models()
    sys.exit(0 if success else 1)
