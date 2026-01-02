"""
Feature Store Module

A simple feature store implementation using SQLite.
Stores engineered features with versioning for reproducibility.
"""

import pandas as pd
import sqlite3
from pathlib import Path
import sys
from datetime import datetime
import json
from typing import Optional, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


class FeatureStore:
    """
    Simple feature store for storing and retrieving engineered features.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize feature store.
        
        Args:
            db_path: Path to SQLite database (uses config default if None)
        """
        self.config = get_config()
        
        if db_path is None:
            db_path = self.config.database.sqlite.path
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables if they don't exist
        self._create_tables()
        
        print(f"✅ Feature store initialized")
        print(f"   Database: {self.db_path}")
    
    def _create_tables(self) -> None:
        """Create necessary tables in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feature metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    version TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    num_features INTEGER NOT NULL,
                    num_rows INTEGER NOT NULL,
                    feature_names TEXT NOT NULL,
                    config TEXT,
                    description TEXT
                )
            """)
            
            # Feature versions table (stores actual feature data references)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (version) REFERENCES feature_metadata(version)
                )
            """)
            
            conn.commit()
    
    def save_features(self, 
                     df: pd.DataFrame,
                     version: str,
                     description: str = "",
                     config_dict: Optional[Dict] = None) -> None:
        """
        Save engineered features to the store.
        
        Args:
            df: DataFrame with features
            version: Version tag for these features
            description: Optional description
            config_dict: Optional configuration used to generate features
        """
        print(f"\n💾 Saving features to store...")
        print(f"   Version: {version}")
        print(f"   Shape: {df.shape}")
        
        # Save DataFrame to parquet
        feature_dir = self.db_path.parent / "features"
        feature_dir.mkdir(exist_ok=True)
        
        feature_path = feature_dir / f"features_{version}.parquet"
        df.to_parquet(feature_path, index=False)
        
        # Save metadata to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert metadata
            cursor.execute("""
                INSERT OR REPLACE INTO feature_metadata 
                (version, created_at, num_features, num_rows, feature_names, config, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                version,
                datetime.now().isoformat(),
                len(df.columns),
                len(df),
                json.dumps(list(df.columns)),
                json.dumps(config_dict) if config_dict else None,
                description
            ))
            
            # Insert feature version records
            for col in df.columns:
                cursor.execute("""
                    INSERT INTO feature_versions 
                    (version, feature_name, data_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    version,
                    col,
                    str(df[col].dtype),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
        
        print(f"   ✅ Saved to: {feature_path}")
        print(f"   ✅ Metadata saved to database")
    
    def load_features(self, version: Optional[str] = None) -> pd.DataFrame:
        """
        Load features from the store.
        
        Args:
            version: Version to load (loads latest if None)
            
        Returns:
            DataFrame with features
        """
        # Get version to load
        if version is None:
            version = self.get_latest_version()
            if version is None:
                raise ValueError("No features found in store")
        
        print(f"\n📂 Loading features from store...")
        print(f"   Version: {version}")
        
        # Load from parquet
        feature_dir = self.db_path.parent / "features"
        feature_path = feature_dir / f"features_{version}.parquet"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found: {feature_path}")
        
        df = pd.read_parquet(feature_path)
        
        print(f"   ✅ Loaded: {df.shape}")
        
        return df
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest feature version.
        
        Returns:
            Latest version string or None if no versions exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version FROM feature_metadata 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            return result[0] if result else None
    
    def get_feature_metadata(self, version: Optional[str] = None) -> Dict:
        """
        Get metadata for a feature version.
        
        Args:
            version: Version to get metadata for (latest if None)
            
        Returns:
            Dictionary with metadata
        """
        if version is None:
            version = self.get_latest_version()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM feature_metadata WHERE version = ?
            """, (version,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'version': result[0],
                    'created_at': result[1],
                    'num_features': result[2],
                    'num_rows': result[3],
                    'feature_names': json.loads(result[4]),
                    'config': json.loads(result[5]) if result[5] else None,
                    'description': result[6]
                }
            else:
                return {}
    
    def list_versions(self) -> pd.DataFrame:
        """
        List all feature versions in the store.
        
        Returns:
            DataFrame with version information
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("""
                SELECT version, created_at, num_features, num_rows, description
                FROM feature_metadata
                ORDER BY created_at DESC
            """, conn)
        
        return df
    
    def delete_version(self, version: str) -> None:
        """
        Delete a feature version.
        
        Args:
            version: Version to delete
        """
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feature_versions WHERE version = ?", (version,))
            cursor.execute("DELETE FROM feature_metadata WHERE version = ?", (version,))
            conn.commit()
        
        # Delete parquet file
        feature_dir = self.db_path.parent / "features"
        feature_path = feature_dir / f"features_{version}.parquet"
        
        if feature_path.exists():
            feature_path.unlink()
        
        print(f"✅ Deleted version: {version}")


def test_feature_store():
    """Test the feature store"""
    print("=" * 70)
    print("🧪 TESTING FEATURE STORE")
    print("=" * 70)
    
    # Initialize
    store = FeatureStore()
    
    # Create sample data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'feature1': range(100),
        'feature2': range(100, 200),
        'target': range(200, 300)
    })
    
    # Save features
    store.save_features(
        df=df,
        version="test_v1",
        description="Test feature set",
        config_dict={"test": "config"}
    )
    
    # Load features
    loaded_df = store.load_features(version="test_v1")
    print(f"\n✅ Loaded features: {loaded_df.shape}")
    
    # Get metadata
    metadata = store.get_feature_metadata(version="test_v1")
    print(f"\n📋 Metadata:")
    for key, value in metadata.items():
        if key != 'feature_names':
            print(f"   {key}: {value}")
    
    # List versions
    versions = store.list_versions()
    print(f"\n📚 Available versions:")
    print(versions.to_string(index=False))
    
    print("\n✅ Feature store test complete!")


if __name__ == "__main__":
    test_feature_store()