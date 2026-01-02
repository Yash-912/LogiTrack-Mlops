"""
Feature Engineering Module

Creates time-series features for supply chain forecasting:
- Lag features (past values)
- Rolling window statistics
- Date features
- Cyclical encodings
- Categorical encodings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple
import json
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


class FeatureEngineer:
    """
    Feature engineering pipeline for time-series forecasting.
    """
    
    def __init__(self, mlflow_run=None):
        """
        Initialize feature engineer.
        
        Args:
            mlflow_run: Optional MLflow run context for logging
        """
        self.config = get_config()
        self.mlflow_run = mlflow_run
        self.feature_names = []
        self.encoders = {}
        
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive date features.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            DataFrame with additional date features
        """
        print("\n📅 Extracting date features...")
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Basic date components (if not already present)
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['date'].dt.day
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['date'].dt.isocalendar().week
        if 'quarter' not in df.columns:
            df['quarter'] = df['date'].dt.quarter
        
        # Days in month
        df['days_in_month'] = df['date'].dt.days_in_month
        
        # Week of month (approximate)
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        
        date_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                        'week_of_year', 'quarter', 'days_in_month', 'week_of_month']
        
        print(f"   ✅ Created {len(date_features)} date features")
        
        return df
    
    def encode_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical features using sin/cos transformation.
        This preserves the cyclical nature (e.g., December is close to January).
        
        Args:
            df: DataFrame with cyclical columns
            
        Returns:
            DataFrame with sin/cos encoded features
        """
        if not self.config.features.cyclical_encoding:
            print("\n⚠️  Cyclical encoding disabled in config")
            return df
        
        print("\n🔄 Encoding cyclical features...")
        
        # Month (1-12)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of week (0-6)
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of year (1-365)
        if 'day_of_year' in df.columns:
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        print("   ✅ Created 6 cyclical features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features (past values).
        
        Args:
            df: DataFrame with time-series data
            
        Returns:
            DataFrame with lag features
        """
        print(f"\n⏮️  Creating lag features...")
        
        lag_windows = self.config.features.lag_windows
        print(f"   Lag windows: {lag_windows}")
        
        # Sort by date within each store-item group
        df = df.sort_values(['store_id', 'item_id', 'date'])
        
        # Create lag features for each window
        for lag in lag_windows:
            col_name = f'sales_lag_{lag}'
            df[col_name] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
            self.feature_names.append(col_name)
        
        print(f"   ✅ Created {len(lag_windows)} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with time-series data
            
        Returns:
            DataFrame with rolling features
        """
        print(f"\n📊 Creating rolling window features...")
        
        windows = self.config.features.rolling_windows
        stats = self.config.features.rolling_stats
        
        print(f"   Windows: {windows}")
        print(f"   Statistics: {stats}")
        
        # Sort data
        df = df.sort_values(['store_id', 'item_id', 'date'])
        
        # Create rolling features for each window and statistic
        for window in windows:
            for stat in stats:
                col_name = f'sales_rolling_{stat}_{window}'
                
                if stat == 'mean':
                    df[col_name] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif stat == 'std':
                    df[col_name] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif stat == 'min':
                    df[col_name] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif stat == 'max':
                    df[col_name] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                
                self.feature_names.append(col_name)
        
        num_features = len(windows) * len(stats)
        print(f"   ✅ Created {num_features} rolling features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        print(f"\n🔗 Creating interaction features...")
        
        # Store-Item interaction (already categorical, will be encoded later)
        # df['store_item'] = df['store_id'].astype(str) + '_' + df['item_id'].astype(str)
        
        # Weekend × Holiday interaction
        if 'is_weekend' in df.columns and 'is_holiday' in df.columns:
            df['weekend_holiday'] = (df['is_weekend'] & df['is_holiday']).astype(int)
            print("   ✅ Created weekend_holiday interaction")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical columns
            fit: If True, fit encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded features
        """
        print(f"\n🏷️  Encoding categorical features...")
        
        categorical_cols = ['store_id', 'item_id']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    # Fit and transform
                    encoder = LabelEncoder()
                    df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[col] = encoder
                    print(f"   ✅ Encoded {col} ({df[col].nunique()} categories)")
                else:
                    # Transform only using existing encoder
                    if col in self.encoders:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] 
                            if x in self.encoders[col].classes_ else -1
                        )
                    else:
                        print(f"   ⚠️  Encoder for {col} not found, skipping")
        
        return df
    
    def time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets based on time.
        
        Args:
            df: DataFrame sorted by date
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n✂️  Splitting data by time...")
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate split indices
        n = len(df)
        train_size = int(n * self.config.features.train_size)
        val_size = int(n * self.config.features.val_size)
        
        # Split
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        
        print(f"   Train: {len(train_df):,} rows ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"   Val:   {len(val_df):,} rows ({val_df['date'].min()} to {val_df['date'].max()})")
        print(f"   Test:  {len(test_df):,} rows ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def remove_initial_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with NaN values created by lag/rolling features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame without initial NaN rows
        """
        print(f"\n🧹 Removing rows with NaN from lag/rolling features...")
        
        initial_rows = len(df)
        
        # Get lag and rolling feature columns
        lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        
        if lag_cols:
            # Remove rows where any lag/rolling feature is NaN
            df = df.dropna(subset=lag_cols)
            
            removed_rows = initial_rows - len(df)
            print(f"   ✅ Removed {removed_rows:,} rows ({removed_rows/initial_rows*100:.2f}%)")
            print(f"   Remaining: {len(df):,} rows")
        else:
            print("   ⚠️  No lag/rolling features found")
        
        return df
    
    def run_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            fit: If True, fit encoders and scalers
            
        Returns:
            DataFrame with engineered features
        """
        print("=" * 70)
        print("🔧 FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        
        print(f"\n📊 Input shape: {df.shape}")
        
        # Step 1: Extract date features
        df = self.extract_date_features(df)
        
        # Step 2: Encode cyclical features
        df = self.encode_cyclical_features(df)
        
        # Step 3: Create lag features
        df = self.create_lag_features(df)
        
        # Step 4: Create rolling features
        df = self.create_rolling_features(df)
        
        # Step 5: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 6: Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Step 7: Remove rows with NaN from feature engineering
        df = self.remove_initial_nulls(df)
        
        print(f"\n📊 Output shape: {df.shape}")
        print(f"✅ Feature engineering complete!")
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of all features for modeling.
        
        Returns:
            List of feature column names
        """
        return self.feature_names
    
    def save_encoders(self, path: str) -> None:
        """
        Save encoders for inference time.
        
        Args:
            path: Path to save encoders
        """
        import joblib
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.encoders, output_path)
        print(f"💾 Saved encoders to: {output_path}")
    
    def load_encoders(self, path: str) -> None:
        """
        Load encoders for inference.
        
        Args:
            path: Path to load encoders from
        """
        import joblib
        
        self.encoders = joblib.load(path)
        print(f"📂 Loaded encoders from: {path}")


if __name__ == "__main__":
    """Test feature engineering on processed data"""
    
    print("=" * 70)
    print("🧪 TESTING FEATURE ENGINEERING")
    print("=" * 70)
    
    # Load processed data
    config = get_config()
    df = pd.read_parquet(config.data.processed_data)
    
    print(f"\n📊 Loaded data: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Run pipeline
    df_features = feature_engineer.run_pipeline(df, fit=True)
    
    # Print sample
    print(f"\n🔍 Sample of engineered features (first 5 rows):")
    print(df_features.head().to_string())
    
    print(f"\n📋 Feature columns created:")
    feature_cols = [col for col in df_features.columns if 'lag_' in col or 'rolling_' in col or '_sin' in col or '_cos' in col]
    for col in feature_cols[:10]:  # Show first 10
        print(f"   • {col}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")
    
    print("\n✅ Feature engineering test complete!")