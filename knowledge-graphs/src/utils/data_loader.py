"""
Data Loading Utilities

Helpers for loading and preprocessing data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class DataLoader:
    """
    Utilities for loading course and learner data
    """
    
    @staticmethod
    def load_course_structure(filepath: str) -> Dict[str, Any]:
        """
        Load course structure from JSON file
        
        Args:
            filepath: Path to course structure JSON
            
        Returns:
            Course structure dictionary
        """
        logger.info(f"Loading course structure from {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded course: {data.get('course_name', 'Unknown')}")
        return data
    
    @staticmethod
    def load_learner_trajectories(filepath: str) -> pd.DataFrame:
        """
        Load learner trajectories from CSV
        
        Args:
            filepath: Path to trajectories CSV
            
        Returns:
            DataFrame with trajectory data
        """
        logger.info(f"Loading learner trajectories from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} trajectory steps for {df['learner_id'].nunique()} learners")
        return df
    
    @staticmethod
    def validate_course_structure(data: Dict[str, Any]) -> bool:
        """
        Validate course structure data
        
        Args:
            data: Course structure dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['course_id', 'modules']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate modules
        for module in data.get('modules', []):
            if 'module_id' not in module or 'name' not in module:
                logger.error(f"Invalid module structure: {module}")
                return False
        
        logger.info("Course structure validation passed")
        return True
    
    @staticmethod
    def validate_learner_data(df: pd.DataFrame) -> bool:
        """
        Validate learner trajectory data
        
        Args:
            df: Learner trajectory DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['learner_id', 'module_id', 'timestamp']
        
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for missing values in critical columns
        if df[required_columns].isnull().any().any():
            logger.warning("Found missing values in critical columns")
        
        logger.info("Learner data validation passed")
        return True
    
    @staticmethod
    def export_predictions(
        predictions: Dict[str, float],
        output_path: str,
        include_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Export predictions to file
        
        Args:
            predictions: Dictionary of learner_id -> risk_score
            output_path: Output file path
            include_metadata: Optional metadata to include
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            output_data = {
                "predictions": predictions,
                "metadata": include_metadata or {}
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        elif output_path.suffix == '.csv':
            df = pd.DataFrame([
                {"learner_id": k, "risk_score": v}
                for k, v in predictions.items()
            ])
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported predictions to {output_path}")


