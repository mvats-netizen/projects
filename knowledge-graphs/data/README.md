# Data Directory

This directory contains sample data and schemas for the Graph-Native Learner Risk & Intervention Engine.

## Files

### `sample_course_structure.json`
Example course structure with modules, skills, assessments, and relationships.

**Schema:**
```json
{
  "course_id": "string",
  "course_name": "string",
  "modules": [
    {
      "module_id": "string",
      "name": "string",
      "order": "integer",
      "difficulty": "float (1-10)",
      "estimated_hours": "float",
      "skills": ["skill_id"],
      "prerequisites": ["skill_id"]
    }
  ],
  "skills": [...],
  "assessments": [...],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id",
      "type": "prerequisite|teaches|follows|assesses",
      "weight": "float"
    }
  ]
}
```

### `sample_learner_trajectories.csv` (to be generated)
Example learner trajectory data.

**Schema:**
```csv
learner_id,course_id,module_id,timestamp,score,time_spent,attempts,completed,dropped_out
L001,ml-fundamentals-101,module_1,2024-01-01 10:00:00,85.5,3600,1,true,false
L001,ml-fundamentals-101,module_2,2024-01-03 14:30:00,72.0,4200,2,true,false
```

**Columns:**
- `learner_id`: Unique learner identifier
- `course_id`: Course identifier
- `module_id`: Module identifier
- `timestamp`: ISO 8601 timestamp
- `score`: Performance score (0-100)
- `time_spent`: Time spent in seconds
- `attempts`: Number of attempts
- `completed`: Boolean (true/false)
- `dropped_out`: Boolean (true/false)

## Generating Sample Data

You can generate sample learner trajectory data using:

```python
from src.utils.data_loader import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic learner data
num_learners = 1000
num_modules = 5

data = []
for learner_id in range(num_learners):
    start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 30))
    
    for module_idx in range(1, num_modules + 1):
        # Simulate dropout (20% chance)
        if np.random.random() < 0.2:
            dropped_out = True
            break
        else:
            dropped_out = False
        
        data.append({
            "learner_id": f"L{learner_id:04d}",
            "course_id": "ml-fundamentals-101",
            "module_id": f"module_{module_idx}",
            "timestamp": start_date + timedelta(days=module_idx * 3),
            "score": np.random.normal(75, 15),
            "time_spent": np.random.normal(3600, 1200),
            "attempts": np.random.choice([1, 1, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.07, 0.03]),
            "completed": True,
            "dropped_out": dropped_out
        })

df = pd.DataFrame(data)
df.to_csv("data/sample_learner_trajectories.csv", index=False)
```

## Data Privacy

⚠️ **Important**: This directory should contain only sample/synthetic data. Never commit real learner data to version control.

- Real data should be stored in secure, access-controlled locations
- Use `.gitignore` to exclude sensitive data files
- Follow your organization's data privacy and security policies


