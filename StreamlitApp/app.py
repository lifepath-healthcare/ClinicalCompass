# app.py
import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np 

# --- Configuration (Centralized) ---
LLM_RESULTS_FILE = "multi_model_evaluation.json" 

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Clinical Context Hub",
    page_icon="💡",
    initial_sidebar_state="collapsed" 
)

# --- Helper Functions ---
@st.cache_data # Cache the data loading to speed up app
def load_and_prepare_data_llm(file_path):
    """
    Loads LLM evaluation results from a JSON file and transforms it into a pandas DataFrame.
    The DataFrame will be in a "long" format suitable for plotting.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Results file '{file_path}' not found. Please ensure it exists.")
        return pd.DataFrame(), []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Please check file integrity.")
        return pd.DataFrame(), []

    all_rows = []
    metric_names = set()

    for prompt_entry in data:
        sno = prompt_entry.get("SNO")
        prompt_text = prompt_entry.get("prompt")
        
        models_data = prompt_entry.get("models", {})
        for model_name, metrics in models_data.items():
            for metric_name, score in metrics.items():
                all_rows.append({
                    "SNO": sno,
                    "Prompt": prompt_text,
                    "Model Name": model_name,
                    "Metric": metric_name,
                    "Score": score
                })
                metric_names.add(metric_name) # Collect all unique metric names

    df = pd.DataFrame(all_rows)
    df["Score"] = pd.to_numeric(df["Score"], errors='coerce') # Convert score column to numeric
    return df, sorted(list(metric_names))

def create_radial_chart(metrics_dict, title="Model Metrics"):
    """
    Generates and displays a radial (radar) chart for given metrics.
    """
    if not metrics_dict:
        st.info(f"No metrics available to generate {title} chart.")
        return

    # Filter for numerical metrics only
    chart_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    if not chart_metrics:
        st.info(f"No numerical metrics found for {title} chart visualization.")
        return

    categories = list(chart_metrics.keys())
    values = list(chart_metrics.values())

    # Add the first value to the end to close the loop for the radar chart
    if categories:
        values.append(values[0])
        categories.append(categories[0])
    else:
        st.info(f"Not enough numerical metrics to create a radar chart for {title}.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line_color='darkslateblue',
        fillcolor='rgba(72, 61, 139, 0.5)'
    ))

    max_value_in_data = max(values) if values else 0
    radial_range_max = max(1.0, max_value_in_data * 1.1) # Ensure range goes up to at least 1.0

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, radial_range_max],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], # Explicit ticks for 0-1 range
                tickmode='array',
                showline=True,
                linecolor='gray'
            )
        ),
        showlegend=False,
        title=title,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# Store helper functions in session state so other pages can access them
st.session_state['load_data_llm'] = load_and_prepare_data_llm
st.session_state['create_radial_chart'] = create_radial_chart

# --- Import Page Functions ---
# These imports must come AFTER the helper functions are defined and stored in session_state
from landing_page import landing_page
from llm_overall_analysis_page import run_llm_mean_scores_dashboard
from llm_detailed_prompt_analysis_page import run_llm_detailed_analysis_dashboard


# --- Main Application Router ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'landing_page' # Start with the landing page

# Initialize session state for selected prompt in detailed view
if 'selected_prompt_sno' not in st.session_state:
    st.session_state['selected_prompt_sno'] = None # To pass selected prompt to detailed view

# Conditional rendering of pages based on 'current_page' session state
if st.session_state['current_page'] == 'landing_page':
    landing_page()
elif st.session_state['current_page'] == 'llm_overall_analysis_dashboard':
    run_llm_mean_scores_dashboard(LLM_RESULTS_FILE)
elif st.session_state['current_page'] == 'llm_detailed_analysis_dashboard':
    # Pass the selected prompt SNO to the detailed analysis page
    run_llm_detailed_analysis_dashboard(LLM_RESULTS_FILE, st.session_state['selected_prompt_sno'])