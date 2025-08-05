#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# First, run the data generation script
echo "--- Running daily analysis script ---"
python daily_runner.py

# Then, start the streamlit dashboard
echo "--- Starting Streamlit dashboard ---"
streamlit run dashboard_app.py