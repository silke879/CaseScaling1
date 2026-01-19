# CaseScaling ML Solution

This folder provide a simple miniature “production” for a Case Scaling ML solution.

## Project Layout

```
CaseScaling/
├─ README.md                         # this guide
├─ requirements.txt                  # runtime dependencies
├─ data/                            # downloaded Data set from Kaggle
│   └── TestDataExtraSmall          # extra small data sets for code-testing
│   └── TestDataExtraSmall          # small data sets for model-testing
├─ prompts/                          # AI prompts used
│   └── frontend_prompt.txt
├─ public/
│   ├── index.html                  # Interface
│   └── style.css                   # Styles
├─ index.py                         # FastAPI backend with OpenAI
├─ testModels.py                    # Verschillende modellen testen op latency
```