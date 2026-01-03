import sys 
from pathlib import Path 
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import get_config
from models.model_registry import ModelRegistry
from models.train import train_models
from models.hyperparameter_tuning import tune_and_train

def run_training_pipeline(feature_version : str="v1", run_tuning: bool = False, register_model: bool = False) -> bool:
    config=get_config()
    try:
        if run_tuning and config.model.tuning.enabled:
            success= tune_and_train(feature_version)
            if not success:
                return False
        else:
            print("Skipping parameter tuning")
        
        success=train_models(feature_version)
        if not success:
            print("Training failed")
            return False
        if register_model:
            print("Registering model to MLFlow")
            registry=ModelRegistry()
            models_df=registry.list_models()
        #-------------------------------------------------------------------
         # Success summary
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n📊 What was created:")
        print("   ✅ Trained forecasting model (XGBoost)")
        print("   ✅ Trained anomaly detection model (Isolation Forest)")
        print("   ✅ Model evaluation metrics")
        print("   ✅ Visualization plots")
        print("   ✅ Model registered in MLflow")
        
        print("\n💾 Saved artifacts:")
        print("   • models/trained/forecast_model.joblib")
        print("   • models/trained/anomaly_model.joblib")
        print("   • plots/predictions.png")
        print("   • plots/feature_importance.png")
        
        print("\n🌐 View results:")
        print(f"   MLflow UI: {config.mlflow.tracking_uri}")
        print("   Experiment: model_training")
        
        print("\n📝 Next steps:")
        print("   1. Review model performance in MLflow UI")
        print("   2. Analyze prediction plots")
        print("   3. Check feature importance")
        print("   4. Proceed to Phase VI: Testing")
        print("   5. Then: Phase VII: API Development")
        #------------------------------------------------------------------- 
                
        return True
    except Exception as e:
        print("training pipeline failed")
        print(f"Error: {str(e)}")            
        import traceback 
        traceback.print_exc()
        return False

def main():
    parser=argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1.0",
        help="Version of features to use (default: v1.0)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training"
    )
    
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip model registration"
    )
    args=parser.parse_args()
    success=run_training_pipeline(
        feature_version=args.feature_version,
        run_tuning=args.tune,
        register_model=not args.no_register
    )
    sys.exit(0 if success else 1)

if __name__=="__main__":
    main()
