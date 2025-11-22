"""
Deployment Metrics Dashboard for Streamlit ECG App
Displays response time and user feedback metrics in a separate dashboard.
"""

import streamlit as st
import pandas as pd
import os

st.title("Deployment Metrics Dashboard")

metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'deployment_metrics.csv')
feedback_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'user_feedback.csv')

st.header("Response Time Metrics")
if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    st.dataframe(metrics_df)
    if 'response_time' in metrics_df.columns:
        st.line_chart(metrics_df['response_time'], use_container_width=True)
    else:
        # If other numeric columns exist, plot them
        numeric_cols = metrics_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.line_chart(metrics_df[numeric_cols], use_container_width=True)
        else:
            st.info("No numeric metrics available for visualization.")
else:
    st.info("No deployment metrics found. Metrics will appear here once logged.")

st.header("User Feedback")
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
    st.dataframe(feedback_df)
else:
    st.info("No user feedback found. Feedback will appear here once collected.")