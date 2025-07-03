# llm_detailed_prompt_analysis_page.py 
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- Streamlit App Function for LLM Detailed Analysis Dashboard ---
def run_llm_detailed_analysis_dashboard(results_file_path, selected_prompt_sno=None):
    
    # --- Access helper functions from st.session_state ---
    load_and_prepare_data_llm = st.session_state['load_data_llm']
    create_radial_chart = st.session_state['create_radial_chart']
    
    st.title("🔍 LLM Detailed Prompt Analysis")

    # --- Back button to LLM Mean Scores Dashboard ---
    if st.button("⬅️ Back to Overall LLM Analysis", key="back_to_overall_llm_from_detailed"):
        st.session_state['current_page'] = 'llm_overall_analysis_dashboard'
        st.session_state['selected_prompt_sno'] = None
        st.rerun()

    st.markdown("---") # Separator

    st.markdown("""
    Dive deeper into the performance of LLMs for specific prompts.
    """)

    df, all_metrics = load_and_prepare_data_llm(results_file_path)

    if df.empty:
        st.info("No data available for visualization. Please ensure your evaluation script runs successfully and creates the JSON file.")
        return

    # Prepare display options for selectbox - REMOVED SNO
    all_prompt_sno_texts = df[['SNO', 'Prompt']].drop_duplicates().sort_values(by='SNO')
    # Display only the prompt text (truncated)
    display_options = [f"{row['Prompt'][:70]}..." for index, row in all_prompt_sno_texts.iterrows()]
    # Keep mapping from displayed text back to SNO for internal logic
    sno_to_prompt_map = {f"{row['Prompt'][:70]}..." : row['SNO'] for index, row in all_prompt_sno_texts.iterrows()}

    # Determine initial index for selectbox
    initial_index = 0
    if selected_prompt_sno is not None:
        try:
            # Find the display option that matches the selected_prompt_sno
            # Need to reconstruct the display option without SNO to find it
            matching_display_option = next(
                f"{row['Prompt'][:70]}..." for index, row in all_prompt_sno_texts.iterrows() if row['SNO'] == selected_prompt_sno
            )
            initial_index = display_options.index(matching_display_option)
        except (StopIteration, ValueError):
            initial_index = 0 
    elif display_options:
        initial_index = 0
    else:
        initial_index = None

    if display_options:
        selected_display_option = st.selectbox(
            "Select a Prompt for Detailed View",
            options=display_options,
            index=initial_index,
            key="detailed_prompt_select_box"
        )
    else:
        st.warning("No prompts available for detailed analysis.")
        selected_display_option = None

    selected_prompt_sno_current = None
    if selected_display_option:
        selected_prompt_sno_current = sno_to_prompt_map[selected_display_option]
        st.session_state['selected_prompt_sno'] = selected_prompt_sno_current

    if selected_prompt_sno_current is not None:
        full_prompt_text = df[df['SNO'] == selected_prompt_sno_current]['Prompt'].iloc[0]
        
        # REMOVED SNO from subheader and markdown
        st.subheader("Metrics for Selected Prompt") 
        st.markdown(f"**Prompt:** *{full_prompt_text}*") # Keeps just the prompt text
        
        prompt_detail_df = df[df["SNO"] == selected_prompt_sno_current].dropna(subset=["Score"])

        if not prompt_detail_df.empty:
            st.write("\n**Model-wise Scores for this Prompt:**")
            pivot_prompt_detail = prompt_detail_df.pivot_table(
                                        index="Model Name",
                                        columns="Metric",
                                        values="Score"
                                    )
            
            display_metrics = [m for m in all_metrics if m != "generated_text" and m in pivot_prompt_detail.columns]
            st.dataframe(pivot_prompt_detail[display_metrics]) 
            
            st.markdown("---")

            # --- Bar Chart: Prompt-specific Scores ---
            fig_prompt_bar, ax_prompt_bar = plt.subplots(figsize=(14, 8))
            sns.barplot(x="Metric", y="Score", hue="Model Name", data=prompt_detail_df[prompt_detail_df["Metric"].isin(display_metrics)], palette="tab10", ax=ax_prompt_bar)
            
            # REMOVED SNO from chart title
            ax_prompt_bar.set_title(f"Metric Scores for Selected Prompt", fontsize=16) 
            ax_prompt_bar.set_ylabel("Score", fontsize=12)
            ax_prompt_bar.set_xlabel("Metric", fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_prompt_bar, use_container_width=True) 
            
            st.markdown("---")

            # --- Winner Determination (for this specific prompt) ---
            st.subheader(f"🏆 Winning Model for this Prompt")
            if not pivot_prompt_detail.empty:
                numeric_cols = pivot_prompt_detail.select_dtypes(include=np.number).columns
                if not numeric_cols.empty:
                    pivot_prompt_detail['Prompt Mean Score'] = pivot_prompt_detail[numeric_cols].mean(axis=1)
                    winner_model_row_prompt = pivot_prompt_detail.loc[pivot_prompt_detail['Prompt Mean Score'].idxmax()]
                    winner_model_name_prompt = winner_model_row_prompt.name 
                    winner_prompt_mean_score = winner_model_row_prompt['Prompt Mean Score']
                    
                    winner_metrics_prompt = winner_model_row_prompt[numeric_cols].to_dict()
                    create_radial_chart(winner_metrics_prompt, title=f"Metrics for {winner_model_name_prompt} on this Prompt")
                    st.success(f"For this prompt, the winning model is **{winner_model_name_prompt}** with an average score of **{winner_prompt_mean_score:.4f}** across selected metrics!")

                else:
                    st.info("No numerical metrics available to determine a winner for this prompt.")
            else:
                st.info("No data available to determine a winner for this prompt.")
            
            st.markdown("---")
            
        else:
            st.warning("No data for selected models/metrics on this prompt. Please check your sidebar filters (if any apply here).")
    else:
        if display_options: 
            st.info("Please select a prompt from the dropdown above to view its detailed analysis.")
        else: 
            st.info("No prompts found in the data for detailed analysis.")