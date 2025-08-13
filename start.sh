#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# First, run the data generation script
echo "--- Running daily analysis script ---"
python daily_runner.py

# Then, start the streamlit dashboard on all network interfaces
echo "--- Starting Streamlit dashboard ---"
streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0