---
title: FMCG Inventory Intelligence
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.0.0"
app_file: app.py
pinned: false
---



# FMCG Inventory Intelligence Platform — Version 3

A connected, synthetic-first FMCG Machine Learning platform with three separate business dashboards under one shared Unicorn Enterprise UI.

## Streamlit deployment readiness

This repository is configured for a standard Streamlit deployment and runs from the root `app.py` entrypoint.

Use the deploy command:

```bash
streamlit run app.py
```

Recommended production settings:
- keep `app.py` as the entrypoint
- keep `requirements.txt` installed in the runtime environment
- serve on port `8501` via `.streamlit/config.toml`
- ensure the environment has Python 3.10+ and all dependencies from `requirements.txt`

## Implemented Modules
