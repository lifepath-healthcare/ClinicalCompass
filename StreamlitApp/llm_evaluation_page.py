# llm_evaluation_page.py
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
        st.error(f"Error: Results file '{file_path}' not found. Please run the evaluation script first to generate it.")
        return pd.DataFrame(), [] # Return empty DataFrame and empty metric list
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
    
    # Convert score column to numeric, handling errors
    df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
    
    return df, sorted(list(metric_names))

# --- Streamlit App Function for LLM Dashboard ---
def run_llm_dashboard_view(results_file_path):
    # REMOVED: st.set_page_config() from here!
    # The configuration set in main_dashboard_tree.py applies globally.
    """
    Renders the LLM Dashboard page.
    Includes navigation buttons back to the main dashboard and the home screen.
    """
    col_nav1, col_nav2 = st.columns([1, 5]) 

    with col_nav1:
        # Button to go back to the Main Dashboard (Tree View), as per your example
        if st.button("⬅️ Back to Main Dashboard", key="back_to_main_dashboard_llm"):
            st.session_state['current_page'] = 'dashboard_page' # Set to the page name for main dashboard
            st.session_state['selected_tree_node_value'] = None # Clear selection for tree view
            # You might want to reset other main dashboard states here too
            st.rerun()
    with col_nav2:
        # Button to go directly to the Home Screen (Landing Page)
        if st.button("🏠 Go to Home Screen", key="go_to_home_from_llm"):
            st.session_state['current_page'] = 'landing_page'
            # Reset all relevant session states for a clean start on landing page
            st.session_state['selected_tree_node_value'] = None 
            st.session_state['tree_expanded_state'] = [] 
            st.session_state['tree_rerender_key'] = 0
            st.session_state.current_prompt_context_json = None 
            st.session_state.show_json_preview = False
            st.rerun()
    st.title("📊 LLM Evaluation Metrics Dashboard")

    # Add a back button to return to the main dashboard
    

    st.markdown("""
    This dashboard visualizes the performance of different LLMs based on various NLP and DeepEval metrics.
    Use the sidebar to filter and select views.
    """)

    # Load and prepare data
    df, all_metrics = load_and_prepare_data_llm(results_file_path)

    if df.empty:
        st.info("No data available for visualization. Please ensure your evaluation script runs successfully and creates the JSON file.")
    else:
        # --- Sidebar Filters ---
        st.sidebar.header("Filter and View Options")

        # Multiselect for models
        all_models = df["Model Name"].unique().tolist()
        selected_models = st.sidebar.multiselect(
            "Select Models to Compare",
            all_models,
            default=all_models
        )

        # Multiselect for metrics
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics to Display",
            all_metrics,
            default=all_metrics
        )

        # Apply filters
        filtered_df = df[
            (df["Model Name"].isin(selected_models)) & 
            (df["Metric"].isin(selected_metrics))
        ]

        if filtered_df.empty:
            st.warning("No data to display for the selected models/metrics. Please adjust your filters.")
        else:
            st.header("Overall Performance Overview")

            # Calculate average scores for selected metrics for each model
            avg_scores_df = filtered_df.groupby(["Model Name", "Metric"])["Score"].mean().unstack(level='Metric').reset_index()
            
            st.subheader("Average Scores by Model and Metric")
            st.dataframe(avg_scores_df.set_index("Model Name"))

            st.markdown("---")

            # --- Bar chart: Metrics vs Models (Average Scores) ---
            st.header("Metrics vs. Models Bar Chart (Average Scores)")
            st.markdown("This chart shows the average score for each selected metric, broken down by model.")

            if not selected_metrics:
                st.info("Please select at least one metric in the sidebar to see the bar chart.")
            elif not selected_models:
                st.info("Please select at least one model in the sidebar to see the bar chart.")
            else:
                avg_scores_melted = filtered_df.groupby(["Model Name", "Metric"])["Score"].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.barplot(x="Metric", y="Score", hue="Model Name", data=avg_scores_melted, palette="viridis", ax=ax)
                
                ax.set_title("Average Metric Scores by Model", fontsize=16)
                ax.set_ylabel("Average Score", fontsize=12)
                ax.set_xlabel("Metric", fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("---")

            # --- Detailed Prompt View (Optional) ---
            st.header("Detailed Prompt Analysis")
            all_prompt_texts = df["Prompt"].unique().tolist()
            selected_prompt_text = st.selectbox(
                "Select a Prompt for Detailed View",
                all_prompt_texts
            )

            if selected_prompt_text:
                st.subheader(f"Metrics for Prompt: '{selected_prompt_text}'")
                
                prompt_detail_df = filtered_df[filtered_df["Prompt"] == selected_prompt_text]

                if not prompt_detail_df.empty:
                    st.write("\n**Model-wise Scores for this Prompt:**")
                    pivot_prompt_detail = prompt_detail_df.pivot_table(
                                index="Model Name",
                                columns="Metric",
                                values="Score"
                            )
                    st.dataframe(pivot_prompt_detail[[m for m in selected_metrics if m in pivot_prompt_detail.columns]]) 
                    
                    fig_prompt_bar, ax_prompt_bar = plt.subplots(figsize=(14, 8))
                    sns.barplot(x="Metric", y="Score", hue="Model Name", data=prompt_detail_df, palette="tab10", ax=ax_prompt_bar)
                    ax_prompt_bar.set_title(f"Metric Scores for Prompt: '{selected_prompt_text}'", fontsize=16)
                    ax_prompt_bar.set_ylabel("Score", fontsize=12)
                    ax_prompt_bar.set_xlabel("Metric", fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_prompt_bar)
                    
                else:
                    st.warning("No data for selected models/metrics on this prompt. Please check your sidebar filters.")