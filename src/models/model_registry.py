"""
Model Registry Module

Manages model versions and lifecycle in MLflow Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import sys
from typing import Optional, Dict, List
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


class ModelRegistry:
    """
    Manages models in MLflow Model Registry.
    """
    
    def __init__(self):
        """Initialize model registry"""
        self.config = get_config()
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        self.client = MlflowClient()
        
    def register_model(self, run_id: str, model_name: str,
                      artifact_path: str = "model") -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the model
            artifact_path: Path to model artifact in run
            
        Returns:
            Model version
        """
        print(f"\n📝 Registering model: {model_name}")
        print(f"   Run ID: {run_id}")
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        try:
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            
            print(f"   ✅ Registered as version {version}")
            
            return version
            
        except Exception as e:
            print(f"   ❌ Registration failed: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: str,
                              stage: str, archive_existing: bool = True) -> None:
        """
        Transition model to a different stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
        """
        print(f"\n🔄 Transitioning model to {stage}...")
        print(f"   Model: {model_name}")
        print(f"   Version: {version}")
        
        # Archive existing models in target stage if requested
        if archive_existing and stage in ["Staging", "Production"]:
            existing = self.client.get_latest_versions(model_name, stages=[stage])
            for model_version in existing:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Archived"
                )
                print(f"   📦 Archived version {model_version.version}")
        
        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"   ✅ Version {version} is now in {stage}")
    
    def load_model_from_registry(self, model_name: str,
                                 stage: Optional[str] = "Production",
                                 version: Optional[str] = None):
        """
        Load a model from registry.
        
        Args:
            model_name: Name of the model
            stage: Stage to load from (if version not specified)
            version: Specific version to load
            
        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
            print(f"\n📂 Loading model {model_name} version {version}")
        else:
            model_uri = f"models:/{model_name}/{stage}"
            print(f"\n📂 Loading model {model_name} from {stage} stage")
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"   ✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict:
        """
        Get metadata for a model version.
        
        Args:
            model_name: Name of the model
            version: Version (latest if None)
            
        Returns:
            Dictionary with metadata
        """
        if version is None:
            versions = self.client.get_latest_versions(model_name)
            if not versions:
                raise ValueError(f"No versions found for {model_name}")
            model_version = versions[0]
        else:
            model_version = self.client.get_model_version(model_name, version)
        
        # Get run info
        run = self.client.get_run(model_version.run_id)
        
        metadata = {
            'name': model_version.name,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'run_id': model_version.run_id,
            'creation_timestamp': model_version.creation_timestamp,
            'metrics': run.data.metrics,
            'params': run.data.params,
            'tags': run.data.tags
        }
        
        return metadata
    
    def list_models(self) -> pd.DataFrame:
        """
        List all registered models.
        
        Returns:
            DataFrame with model information
        """
        models = self.client.search_registered_models()
        
        if not models:
            print("No registered models found")
            return pd.DataFrame()
        
        model_list = []
        for model in models:
            for version in model.latest_versions:
                model_list.append({
                    'name': model.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id
                })
        
        return pd.DataFrame(model_list)
    
    def compare_models(self, model_name: str, 
                      versions: List[str],
                      metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple model versions.
        
        Args:
            model_name: Name of the model
            versions: List of version numbers to compare
            metrics: List of metrics to compare (None = all)
            
        Returns:
            DataFrame with comparison
        """
        print(f"\n📊 Comparing model versions...")
        
        comparison_data = []
        
        for version in versions:
            metadata = self.get_model_metadata(model_name, version)
            
            row = {
                'version': version,
                'stage': metadata['stage']
            }
            
            # Add metrics
            if metrics:
                for metric in metrics:
                    row[metric] = metadata['metrics'].get(metric, None)
            else:
                row.update(metadata['metrics'])
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        print(df.to_string(index=False))
        
        return df
    
    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
        """
        print(f"\n🗑️  Deleting model version...")
        print(f"   Model: {model_name}")
        print(f"   Version: {version}")
        
        self.client.delete_model_version(model_name, version)
        
        print(f"   ✅ Version {version} deleted")


def demo_model_registry():
    """Demo the model registry functionality"""
    print("=" * 70)
    print("🧪 MODEL REGISTRY DEMO")
    print("=" * 70)
    
    registry = ModelRegistry()
    
    # List all models
    print("\n📋 Registered Models:")
    models_df = registry.list_models()
    if not models_df.empty:
        print(models_df.to_string(index=False))
    else:
        print("   No models registered yet")
        print("\n💡 Run training pipeline first:")
        print("   python src/models/train.py")
    
    print("\n✅ Model registry demo complete!")


if __name__ == "__main__":
    demo_model_registry()