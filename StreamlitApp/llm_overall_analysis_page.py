import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- Streamlit App Function for LLM Mean Scores Dashboard ---
def run_llm_mean_scores_dashboard(results_file_path):
    # --- Access helper functions from st.session_state ---
    load_and_prepare_data_llm = st.session_state['load_data_llm']
    create_radial_chart = st.session_state['create_radial_chart']
    
    st.title("📊 LLM Overall Prompt Analysis")

    # --- Back button to Home Screen ---
    if st.button("🏠 Back to Home", key="back_to_home_from_overall_llm"):
        st.session_state['current_page'] = 'landing_page'
        st.session_state['selected_prompt_sno'] = None
        st.rerun()

    st.markdown("---") # Separator

    st.markdown("""
    This dashboard provides an overall view of LLM performance across all evaluated prompts and metrics.
    Use the sidebar filters to customize your view.
    """)

    # Load and prepare data
    df, all_metrics = load_and_prepare_data_llm(results_file_path)

    if df.empty:
        st.info("No data available for visualization. Please ensure your evaluation script runs successfully and creates the JSON file.")
        return # Exit the function if no data

    # --- Sidebar Filters ---
    st.sidebar.header("Filter and View Options")

    all_models = df["Model Name"].unique().tolist()
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        all_models,
        default=all_models
    )

    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        all_metrics,
        default=[m for m in all_metrics if m != "generated_text"]
    )

    # Apply filters
    filtered_df = df[
        (df["Model Name"].isin(selected_models)) & 
        (df["Metric"].isin(selected_metrics))
    ].dropna(subset=["Score"])

    if filtered_df.empty:
        st.warning("No data to display for the selected models/metrics. Please adjust your filters.")
        return # Exit the function if no data after filtering

    st.header("Overall Performance Overview")

    # Calculate average scores for selected metrics for each model
    avg_scores_df = filtered_df.groupby(["Model Name", "Metric"])["Score"].mean().unstack(level='Metric').reset_index()
    
    st.subheader("Average Scores by Model and Metric")
    st.dataframe(avg_scores_df.set_index("Model Name"))

    st.markdown("---")

    # --- Bar chart: Metrics vs Models (Average Scores) - MOVED HERE ---
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
        st.pyplot(fig, use_container_width=True) 

    st.markdown("---")

    # --- Winner Determination (Overall) - MOVED HERE ---
    st.subheader("🏆 Overall Winning Model")
    if not avg_scores_df.empty:
        numeric_cols = avg_scores_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            avg_scores_df['Overall Mean Score'] = avg_scores_df[numeric_cols].mean(axis=1)
            winner_model_row = avg_scores_df.loc[avg_scores_df['Overall Mean Score'].idxmax()]
            winner_model_name = winner_model_row["Model Name"]
            winner_overall_mean_score = winner_model_row['Overall Mean Score']
            
            winner_metrics = winner_model_row[numeric_cols].to_dict()
            create_radial_chart(winner_metrics, title=f"{winner_model_name} Overall Metrics")
            st.success(f"The overall winning model is **{winner_model_name}** with an average score of **{winner_overall_mean_score:.4f}** across selected metrics!")

        else:
            st.info("No numerical metrics available to determine an overall winner.")
    else:
        st.info("No data available to determine an overall winner.")
    
    st.markdown("---")


    # --- Go to Detailed Prompt View Button (at the end) ---
    st.subheader("Explore Detailed Prompt Analysis")
    if st.button("➡️ Go to LLM Detailed Prompt Analysis", key="go_to_detailed_prompt_from_overall_llm"):
        st.session_state['current_page'] = 'llm_detailed_analysis_dashboard' 
        st.session_state['selected_prompt_sno'] = None
        st.rerun()