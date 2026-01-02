"""
MLflow Experiment Manager

Handles:
- Creating and managing experiments
- Logging parameters, metrics, and artifacts
- Model versioning
- Run context management
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from datetime import datetime
from pathlib import Path
import sys
from contextlib import contextmanager
from typing import Dict, Any, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


class ExperimentManager:
    """
    Manages MLflow experiments and run tracking.
    Provides a clean interface for logging all ML-related information.
    """
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize the experiment manager.
        
        Args:
            experiment_name: Name of the experiment (uses config default if None)
        """
        self.config = get_config()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        print("DEBUG mlflow.get_tracking_uri():", mlflow.get_tracking_uri())    
        
        # Use provided name or config default
        self.experiment_name = experiment_name or self.config.mlflow.experiment_name
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()
        
        print(f"✅ Experiment Manager initialized")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Experiment ID: {self.experiment_id}")
        print(f"   Tracking URI: {self.config.mlflow.tracking_uri}")
    
    def _get_or_create_experiment(self) -> str:
        """
        Get existing experiment or create a new one.
        
        Returns:
            Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.config.mlflow.artifact_location,
                tags={
                    "project": self.config.project.name,
                    "version": self.config.project.version,
                    "created_at": datetime.now().isoformat()
                }
            )
            print(f"✅ Created new experiment: {self.experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {self.experiment_name}")
        
        return experiment_id
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for MLflow runs.
        
        Args:
            run_name: Name for this run
            nested: Whether this is a nested run
            
        Yields:
            MLflow run object
            
        Example:
            with exp_manager.start_run("training_run"):
                mlflow.log_param("learning_rate", 0.1)
                mlflow.log_metric("accuracy", 0.95)
        """
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        ) as run:
            print(f"\n🏃 Started MLflow run: {run.info.run_id}")
            if run_name:
                print(f"   Run name: {run_name}")
            
            try:
                yield run
            finally:
                print(f"✅ Completed MLflow run: {run.info.run_id}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.
        
        Args:
            params: Dictionary of parameter names and values
            
        Example:
            exp_manager.log_params({
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            })
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        print(f"📊 Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking over time
            
        Example:
            exp_manager.log_metrics({
                "rmse": 15.3,
                "mae": 12.1,
                "r2": 0.85
            })
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        print(f"📈 Logged {len(metrics)} metrics")
    
    def log_model(self, 
                  model: Any, 
                  artifact_path: str,
                  registered_model_name: Optional[str] = None,
                  signature=None,
                  input_example=None) -> None:
        """
        Log a trained model.
        
        Args:
            model: The trained model object
            artifact_path: Path within the run's artifact URI
            registered_model_name: Name for model registry (optional)
            signature: Model signature (input/output schema)
            input_example: Example input for model
            
        Example:
            exp_manager.log_model(
                model=xgb_model,
                artifact_path="model",
                registered_model_name="supply_chain_forecaster"
            )
        """
        # Detect model type and log accordingly
        model_type = type(model).__name__
        
        if "XGB" in model_type or "xgboost" in str(type(model).__module__):
            mlflow.xgboost.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            # Generic sklearn-compatible model
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        
        print(f"💾 Logged model: {artifact_path}")
        if registered_model_name:
            print(f"   Registered as: {registered_model_name}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a file or directory as an artifact.
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Path within the run's artifact directory
            
        Example:
            exp_manager.log_artifact("plots/feature_importance.png", "plots")
        """
        mlflow.log_artifact(local_path, artifact_path)
        print(f"📎 Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, filename: str) -> None:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
            
        Example:
            exp_manager.log_dict({"config": "values"}, "config.json")
        """
        mlflow.log_dict(dictionary, filename)
        print(f"📋 Logged dictionary: {filename}")
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the saved figure
            
        Example:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            exp_manager.log_figure(fig, "plot.png")
        """
        mlflow.log_figure(figure, artifact_file)
        print(f"📊 Logged figure: {artifact_file}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set multiple tags for the current run.
        
        Args:
            tags: Dictionary of tag names and values
            
        Example:
            exp_manager.set_tags({
                "model_type": "xgboost",
                "data_version": "v1.0",
                "environment": "development"
            })
        """
        for key, value in tags.items():
            mlflow.set_tag(key, str(value))
        
        print(f"🏷️  Set {len(tags)} tags")
    
    def log_dataset_info(self, 
                         dataset_name: str,
                         num_rows: int,
                         num_features: int,
                         date_range: tuple = None) -> None:
        """
        Log information about the dataset used.
        
        Args:
            dataset_name: Name of the dataset
            num_rows: Number of rows
            num_features: Number of features
            date_range: Optional tuple of (start_date, end_date)
        """
        dataset_info = {
            "dataset_name": dataset_name,
            "num_rows": num_rows,
            "num_features": num_features
        }
        
        if date_range:
            dataset_info["start_date"] = str(date_range[0])
            dataset_info["end_date"] = str(date_range[1])
        
        # Log as params and as artifact
        self.log_params({
            "dataset_name": dataset_name,
            "num_rows": num_rows,
            "num_features": num_features
        })
        
        self.log_dict(dataset_info, "dataset_info.json")
        
        print(f"📊 Logged dataset info: {dataset_name}")
    
    def get_best_run(self, metric_name: str, ascending: bool = False):
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: If True, lower is better (e.g., RMSE)
            
        Returns:
            Best run information
            
        Example:
            best_run = exp_manager.get_best_run("rmse", ascending=True)
            print(f"Best RMSE: {best_run.data.metrics['rmse']}")
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            print(f"🏆 Best run: {best_run.info.run_id}")
            print(f"   {metric_name}: {best_run.data.metrics.get(metric_name)}")
            return best_run
        else:
            print("⚠️  No runs found")
            return None
    
    def compare_runs(self, run_ids: list, metrics: list = None):
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (None = all metrics)
            
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "status": run.info.status
            }
            
            # Add all metrics or specified ones
            if metrics:
                for metric in metrics:
                    run_data[metric] = run.data.metrics.get(metric, None)
            else:
                run_data.update(run.data.metrics)
            
            comparison_data.append(run_data)
        
        df = pd.DataFrame(comparison_data)
        print(f"\n📊 Run Comparison:")
        print(df.to_string(index=False))
        
        return df
    
    def end_run(self):
        """End the current run"""
        mlflow.end_run()
        print("🛑 Ended current run")


def test_experiment_manager():
    """
    Test the experiment manager with a simple example.
    """
    print("=" * 70)
    print("🧪 TESTING EXPERIMENT MANAGER")
    print("=" * 70)
    
    # Initialize
    exp_manager = ExperimentManager()
    
    # Start a test run
    with exp_manager.start_run(run_name="test_run"):
        # Log parameters
        exp_manager.log_params({
            "test_param_1": "value1",
            "test_param_2": 42,
            "test_param_3": 3.14
        })
        
        # Log metrics
        exp_manager.log_metrics({
            "test_metric_1": 0.95,
            "test_metric_2": 100.5
        })
        
        # Set tags
        exp_manager.set_tags({
            "test_type": "unit_test",
            "environment": "development"
        })
        
        # Log a dictionary
        exp_manager.log_dict(
            {"test": "data", "nested": {"key": "value"}},
            "test_config.json"
        )
    
    print("\n✅ Test completed!")
    print(f"\n🌐 View results at: {exp_manager.config.mlflow.tracking_uri}")
    print("   (Make sure MLflow server is running)")


if __name__ == "__main__":
    """Run test"""
    test_experiment_manager()