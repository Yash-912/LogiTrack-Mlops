"""
Feature Engineering Pipeline

Complete pipeline that:
1. Loads processed data
2. Engineers features
3. Validates features
4. Saves to feature store
5. Logs to MLflow
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import get_config
from features.feature_engineering import FeatureEngineer
from features.feature_store import FeatureStore
from models.experiment_manager import ExperimentManager


def run_feature_pipeline(version: str = "v1.0") -> bool:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        version: Version tag for the features
        
    Returns:
        True if successful, False otherwise
    """
    print("=" * 80)
    print(" " * 25 + "FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    try:
        # Step 1: Load configuration
        config = get_config()
        print(f"\n📋 Configuration loaded")
        
        # Step 2: Initialize MLflow
        exp_manager = ExperimentManager(experiment_name="feature_engineering")
        
        # Step 3: Load processed data
        print(f"\n📂 Loading processed data...")
        df = pd.read_parquet(config.data.processed_data)
        print(f"   ✅ Loaded: {df.shape}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Step 4: Feature engineering with MLflow tracking
        with exp_manager.start_run(run_name=f"feature_engineering_{version}"):
            
            # Log dataset info
            exp_manager.log_dataset_info(
                dataset_name="processed_supply_chain_data",
                num_rows=len(df),
                num_features=len(df.columns),
                date_range=(df['date'].min(), df['date'].max())
            )
            
            # Set tags
            exp_manager.set_tags({
                "pipeline": "feature_engineering",
                "version": version,
                "data_source": "processed_data"
            })
            
            # Initialize feature engineer
            feature_engineer = FeatureEngineer(mlflow_run=exp_manager)
            
            # Run feature engineering
            df_features = feature_engineer.run_pipeline(df, fit=True)
            
            # Log feature engineering parameters
            exp_manager.log_params({
                "lag_windows": str(config.features.lag_windows),
                "rolling_windows": str(config.features.rolling_windows),
                "rolling_stats": str(config.features.rolling_stats),
                "cyclical_encoding": config.features.cyclical_encoding,
                "num_input_rows": len(df),
                "num_output_rows": len(df_features),
                "num_input_features": len(df.columns),
                "num_output_features": len(df_features.columns)
            })
            
            # Log metrics
            rows_removed = len(df) - len(df_features)
            exp_manager.log_metrics({
                "rows_removed": rows_removed,
                "rows_removed_pct": (rows_removed / len(df)) * 100,
                "feature_count": len(df_features.columns),
                "null_count": df_features.isnull().sum().sum()
            })
            
            # Save encoders
            encoder_path = Path("models") / "encoders" / f"encoders_{version}.joblib"
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            feature_engineer.save_encoders(str(encoder_path))
            
            # Log encoder as artifact
            exp_manager.log_artifact(str(encoder_path), "encoders")
            
            # Step 5: Save to feature store
            print(f"\n💾 Saving to feature store...")
            feature_store = FeatureStore()
            
            feature_store.save_features(
                df=df_features,
                version=version,
                description=f"Engineered features for supply chain forecasting - {version}",
                config_dict={
                    "lag_windows": config.features.lag_windows,
                    "rolling_windows": config.features.rolling_windows,
                    "rolling_stats": config.features.rolling_stats,
                    "cyclical_encoding": config.features.cyclical_encoding
                }
            )
            
            # Log feature store metadata
            metadata = feature_store.get_feature_metadata(version)
            exp_manager.log_dict(metadata, "feature_metadata.json")
            
            # Step 6: Split into train/val/test
            print(f"\n✂️  Creating train/val/test splits...")
            train_df, val_df, test_df = feature_engineer.time_based_split(df_features)
            
            # Log split information
            exp_manager.log_params({
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "train_start": str(train_df['date'].min()),
                "train_end": str(train_df['date'].max()),
                "val_start": str(val_df['date'].min()),
                "val_end": str(val_df['date'].max()),
                "test_start": str(test_df['date'].min()),
                "test_end": str(test_df['date'].max())
            })
            
            # Save splits
            split_dir = Path("data") / "features" / version
            split_dir.mkdir(parents=True, exist_ok=True)
            
            train_df.to_parquet(split_dir / "train.parquet", index=False)
            val_df.to_parquet(split_dir / "val.parquet", index=False)
            test_df.to_parquet(split_dir / "test.parquet", index=False)
            
            print(f"   ✅ Saved splits to: {split_dir}")
            
            # Log splits as artifacts
            exp_manager.log_artifact(str(split_dir / "train.parquet"), "splits")
            exp_manager.log_artifact(str(split_dir / "val.parquet"), "splits")
            exp_manager.log_artifact(str(split_dir / "test.parquet"), "splits")
        
        # Step 7: Summary
        print("\n" + "=" * 80)
        print("📊 FEATURE ENGINEERING SUMMARY")
        print("=" * 80)
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\n📋 Results:")
        print(f"   Version: {version}")
        print(f"   Total features: {len(df_features.columns)}")
        print(f"   Total rows: {len(df_features):,}")
        print(f"\n📊 Split sizes:")
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")
        print(f"\n💾 Saved to:")
        print(f"   Feature store: {version}")
        print(f"   Splits: {split_dir}")
        print(f"   Encoders: {encoder_path}")
        print(f"\n🌐 View in MLflow: {config.mlflow.tracking_uri}")
        print("=" * 80)
        
        print("\n📝 Next steps:")
        print("   1. Review features in MLflow UI")
        print("   2. Proceed to model training")
        print("   3. Run: python pipelines/training_pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Feature pipeline failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Version tag for features (default: v1.0)"
    )
    
    args = parser.parse_args()
    
    success = run_feature_pipeline(version=args.version)
    sys.exit(0 if success else 1)