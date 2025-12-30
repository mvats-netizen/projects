"""
Generate Sample Learner Trajectory Data

Creates synthetic learner data for testing and development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_trajectories(
    num_learners: int = 500,
    course_id: str = "ml-fundamentals-101",
    num_modules: int = 5,
    dropout_rate: float = 0.25,
    output_path: str = "data/sample_learner_trajectories.csv"
):
    """
    Generate synthetic learner trajectory data
    
    Args:
        num_learners: Number of learners to generate
        course_id: Course identifier
        num_modules: Number of modules in course
        dropout_rate: Overall dropout probability
        output_path: Output CSV file path
    """
    print(f"Generating {num_learners} learner trajectories...")
    
    np.random.seed(42)  # For reproducibility
    data = []
    
    for learner_idx in range(num_learners):
        learner_id = f"L{learner_idx:04d}"
        
        # Random start date
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 60))
        
        # Learner ability (affects scores)
        learner_ability = np.random.normal(75, 15)
        learner_ability = np.clip(learner_ability, 40, 100)
        
        # Engagement level (affects time spent and dropout)
        engagement = np.random.beta(5, 2)  # Skewed towards higher engagement
        
        dropped_out = False
        
        for module_idx in range(1, num_modules + 1):
            module_id = f"module_{module_idx}"
            
            # Progressive dropout probability (increases with modules)
            module_dropout_prob = dropout_rate * (1 + 0.1 * module_idx)
            
            # Lower engagement increases dropout risk
            adjusted_dropout_prob = module_dropout_prob * (1.5 - engagement)
            
            # Check if learner drops out
            if np.random.random() < adjusted_dropout_prob:
                dropped_out = True
                # Add final entry showing dropout
                data.append({
                    "learner_id": learner_id,
                    "course_id": course_id,
                    "module_id": module_id,
                    "timestamp": start_date + timedelta(days=module_idx * 3, hours=np.random.randint(0, 24)),
                    "score": None,  # Didn't complete
                    "time_spent": np.random.normal(1800, 600),  # Less time before dropout
                    "attempts": 1,
                    "completed": False,
                    "dropped_out": True
                })
                break
            
            # Module difficulty increases
            module_difficulty = 0.5 + (module_idx * 0.1)
            
            # Score based on ability and difficulty
            score = learner_ability - (module_difficulty * 20) + np.random.normal(0, 10)
            score = np.clip(score, 0, 100)
            
            # Time spent based on engagement and difficulty
            base_time = 3600  # 1 hour
            time_spent = base_time * module_difficulty * (2 - engagement) + np.random.normal(0, 600)
            time_spent = max(600, time_spent)  # At least 10 minutes
            
            # Number of attempts (lower scores = more attempts)
            if score < 60:
                attempts = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
            elif score < 75:
                attempts = np.random.choice([1, 2], p=[0.7, 0.3])
            else:
                attempts = 1
            
            # Add successful completion entry
            data.append({
                "learner_id": learner_id,
                "course_id": course_id,
                "module_id": module_id,
                "timestamp": start_date + timedelta(
                    days=module_idx * 3 + np.random.randint(0, 3),
                    hours=np.random.randint(8, 22)
                ),
                "score": round(score, 1),
                "time_spent": round(time_spent, 0),
                "attempts": attempts,
                "completed": True,
                "dropped_out": False
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by learner and timestamp
    df = df.sort_values(["learner_id", "timestamp"])
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"\nGenerated data saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  - Total entries: {len(df)}")
    print(f"  - Unique learners: {df['learner_id'].nunique()}")
    print(f"  - Dropout rate: {df['dropped_out'].mean():.1%}")
    print(f"  - Average score: {df['score'].mean():.1f}")
    print(f"  - Average time spent: {df['time_spent'].mean()/3600:.1f} hours")
    print(f"  - Completion rate: {df['completed'].mean():.1%}")
    
    return df


if __name__ == "__main__":
    generate_sample_trajectories(
        num_learners=500,
        dropout_rate=0.25
    )


