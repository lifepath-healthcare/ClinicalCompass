import streamlit as st
from streamlit_tree_select import tree_select
import json
import os
import pandas as pd # Ensure this is imported if you use dataframes
import matplotlib.pyplot as plt # Ensure these are imported if you use them
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

# --- Import page functions ---
from landing_page import landing_page
from llm_evaluation_page import run_llm_dashboard_view 

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Clinical Context Hub", page_icon="💡")

# --- Configuration ---
# Ensure these paths are correct for your system
TRADITIONAL_RESULTS_FILE = "traditional_model_metrics.json"
LLM_RESULTS_FILE = "multi_model_evaluation.json"
PROMPT_CONTEXT_FILE = "prompt_context.json" 

# --- Custom CSS for better visibility of disabled text inputs ---
st.markdown(
    """
    <style>
    /* Target disabled text inputs */
    div.stTextInput > div > div > input[disabled] {
        color: #333333; /* Darker text color (e.g., dark grey) */
        background-color: #f0f2f6; /* Slightly off-white/light grey background to differentiate */
        -webkit-text-fill-color: #333333; /* For Webkit browsers like Chrome, Safari */
        opacity: 1; /* Ensure no opacity is applied by default disabled styling */
    }

    /* Make labels for text inputs darker and bolder */
    div.stTextInput > label {
        color: #1a1a1a; /* Even darker for labels */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions for Data Loading and Structuring ---
# (These remain the same as in your previous main_app.py code)

@st.cache_data
def load_json_data(file_path):
    """Loads JSON data from a specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None # Return None if file not found, to handle gracefully
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Check file integrity.")
        return None

@st.cache_data
def load_and_structure_traditional_data(file_path):
    """
    Loads traditional model data and structures it hierarchically:
    {algorithm: {model: {use_case_value: {"label": use_case_text, "metrics": {...}}}}}
    """
    raw_data = load_json_data(file_path)
    if not raw_data:
        return {} # Return empty dict if file not found or empty

    structured_data = {}
    for entry in raw_data:
        algorithm = entry.get("algorithm", "Unknown Algorithm")
        model = entry.get("model", "Unknown Model")
        use_case = entry.get("use_case", "Unknown Use Case")
        metrics = entry.get("metrics", {})

        if algorithm not in structured_data:
            structured_data[algorithm] = {}
        if model not in structured_data[algorithm]:
            structured_data[algorithm][model] = {}
        
        # Create a unique value for the use case node in the tree
        use_case_value = f"traditional_{algorithm.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}_{use_case.lower().replace(' ', '_')}"
        structured_data[algorithm][model][use_case_value] = {
            "label": use_case,
            "metrics": metrics
        }
    return structured_data

@st.cache_data
def load_and_structure_llm_data(file_path):
    """
    Loads LLM evaluation data and structures it hierarchically:
    {model_name: {prompt_value: {"label": prompt_display_text, "full_prompt": "...", "metrics": {...}}}}
    """
    raw_data = load_json_data(file_path)
    if not raw_data:
        return {} # Return empty dict if file not found or empty

    structured_data = {}

    for entry in raw_data:
        prompt_sno = entry.get("SNO", "Unknown SNO")
        prompt_text = entry.get("prompt", "Unknown Prompt")
        
        models_data = entry.get("models", {})
        for model_name, metrics in models_data.items():
            if model_name not in structured_data:
                structured_data[model_name] = {}
            
            prompt_value = f"llm_{model_name.lower().replace(' ', '_')}_prompt_{prompt_sno}"
            structured_data[model_name][prompt_value] = {
                "label": f"Prompt {prompt_sno}: {prompt_text[:70]}...", # Truncate for display
                "full_prompt": prompt_text,
                "metrics": metrics
            }
    return structured_data

def create_tree_nodes(traditional_structured, llm_structured):
    """
    Constructs the tree nodes for streamlit_tree_select based on structured data.
    """
    nodes = []

    # Add Traditional Models
    for algorithm, models_data in traditional_structured.items():
        algorithm_children = []
        for model, use_cases_data in models_data.items():
            model_children = []
            for use_case_value, uc_info in use_cases_data.items():
                model_children.append({
                    "label": f"Use Case: {uc_info['label']}",
                    "value": use_case_value
                })
            algorithm_children.append({
                "label": model,
                "value": f"model_{algorithm.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}",
                "children": model_children
            })
        nodes.append({
            "label": algorithm,
            "value": f"algorithm_{algorithm.lower().replace(' ', '_')}",
            "children": algorithm_children
        })

    # Add LLM Models
    if llm_structured:
        llm_models_children = []
        for model_name, prompts_data in llm_structured.items():
            prompt_children = []
            for prompt_value, prompt_info in prompts_data.items():
                prompt_children.append({
                    "label": prompt_info["label"],
                    "value": prompt_value
                })
            llm_models_children.append({
                "label": model_name,
                "value": f"llm_model_{model_name.lower().replace(' ', '_')}",
                "children": prompt_children
            })
        nodes.append({
            "label": "LLM Prediction Algorithms", 
            "value": "llm_prediction_algorithms_root",
            "children": llm_models_children
        })
    else:
        nodes.append({
            "label": "LLM Prediction Algorithms (No Data)",
            "value": "llm_prediction_algorithms_root_no_data",
            "children": []
        })

    return nodes

def get_path_to_node(nodes, target_value, current_path=[]):
    """Recursively finds the path (list of values) from root to target_value."""
    for node in nodes:
        new_path = current_path + [node['value']]
        if node['value'] == target_value:
            return new_path 
        if 'children' in node:
            res = get_path_to_node(node['children'], target_value, new_path) 
            if res:
                return res
    return None

# --- Function to create a Radial Chart ---
def create_radial_chart(metrics_dict):
    """
    Creates a radial (radar) chart from a dictionary of numerical metrics.
    Uses darker colors for better visibility.
    Assumes metrics are scaled between 0 and 1.
    """
    if not metrics_dict:
        st.info("No metrics available to generate a chart.")
        return

    # Filter out metrics that are not numerical or suitable for radar chart
    chart_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    if not chart_metrics:
        st.info("No numerical metrics found for chart visualization.")
        return

    categories = list(chart_metrics.keys())
    values = list(chart_metrics.values())

    # To close the circle on the radar chart, repeat the first value and category
    if categories:
        values.append(values[0])
        categories.append(categories[0])
    else:
        st.info("Not enough numerical metrics to create a radar chart.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line_color='darkslateblue', # Darker blue for the line
        fillcolor='rgba(72, 61, 139, 0.5)' # Semi-transparent darker blue for the fill
    ))

    # Determine the maximum range for the radial axis.
    max_value_in_data = max(values) if values else 0
    radial_range_max = max(1.0, max_value_in_data * 1.1)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, radial_range_max],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                tickmode='array',
                showline=True,
                linecolor='gray'
            )
        ),
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Modified display_metrics_in_column function ---
def display_metrics_in_column(content_type, identifier, traditional_structured_data, llm_structured_data):
    """
    Displays the metrics in a column based on content type and identifier,
    using st.text_input with disabled=True, and includes a radial chart.
    Now with only one main heading and no separate subheaders for chart/inputs.
    """
    st.subheader("Metric Details") # This is the only main heading for the section
    st.markdown("---")

    metrics_to_display = {}
    context_info = {} 

    if content_type == 'traditional_use_case':
        found_data = False
        for algo_name, models in traditional_structured_data.items():
            for model_name, use_cases in models.items():
                if identifier in use_cases:
                    use_case_info = use_cases[identifier]
                    context_info = {
                        "Algorithm": algo_name,
                        "Model": model_name,
                        "Use Case": use_case_info['label']
                    }
                    metrics_to_display = use_case_info['metrics']
                    found_data = True
                    break
            if found_data:
                break
        
        if not found_data:
            st.error("Could not find traditional use case data.")
            return

    elif content_type == 'llm_prompt':
        found_data = False
        for model_name, prompts_in_model in llm_structured_data.items():
            if identifier in prompts_in_model:
                prompt_info = prompts_in_model[identifier]
                context_info = {
                    "Model": model_name,
                    "Prompt Text": prompt_info['full_prompt'] 
                }
                # Filter out 'generated_text' from metrics for display and chart
                metrics_to_display = {k: v for k, v in prompt_info['metrics'].items() if k != 'generated_text'} 
                found_data = True
                break
        
        if not found_data:
            st.error("Could not find LLM prompt data.")
            return
    else:
        return 

    # --- Display Context Info ---
    if context_info:
        for label, value in context_info.items():
            if label == "Prompt Text":
                st.write(f"**Prompt:** {value}")
            else:
                st.write(f"**{label}:** {value}")
    
    st.markdown("---") 

    # --- Display Radial Chart ---
    if metrics_to_display:
        create_radial_chart(metrics_to_display)
        st.markdown("---") # Separator after chart
    
    # --- Display Individual Metric Inputs ---
    if metrics_to_display:
        for metric_name, score in metrics_to_display.items():
            value_to_display = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
            st.text_input(
                label=metric_name,
                value=value_to_display,
                disabled=True,
                key=f"metric_input_{identifier}_{metric_name.replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '')}"
            )
    else:
        st.info("No metrics available for this selection.")


# --- Function for the Main Dashboard Page ---
# This function is now the 'tree_view' page mentioned in your LLM dashboard example
def run_dashboard_view():
    """Renders the main dashboard page."""

    # --- Home button at the very top of the main content area ---
    if st.button("🏠 Go to Home Screen", key="home_button_top_dashboard"):
        st.session_state['current_page'] = 'landing_page'
        # Reset relevant session states for a clean start on landing page
        st.session_state['selected_tree_node_value'] = None 
        st.session_state['tree_expanded_state'] = [] 
        st.session_state['tree_rerender_key'] = 0
        st.session_state.current_prompt_context_json = None 
        st.session_state.show_json_preview = False
        st.rerun()

    st.markdown("---") # Separator after the home button

    # --- JSON Input Prompt Context JSON Viewer ---
    st.subheader("📊 Input Prompt Context JSON Viewer")
    st.markdown("This section displays the content of the input JSON file. It will initially load `prompt_context.json`. You can also upload a custom file below.")
    
    # --- Progress bar for default prompt_context.json loading ---
    if st.session_state.current_prompt_context_json is None and os.path.exists(PROMPT_CONTEXT_FILE):
        st.write("### Loading Default Input Data...")
        progress_bar_default = st.progress(0, text=f"Loading default '{PROMPT_CONTEXT_FILE}'...")
        import time
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar_default.progress(percent_complete + 1, text=f"Loading default '{PROMPT_CONTEXT_FILE}'... {percent_complete + 1}%")
        progress_bar_default.progress(100, text=f"Default '{PROMPT_CONTEXT_FILE}' loaded successfully!") 

        loaded_default_data = load_json_data(PROMPT_CONTEXT_FILE)
        if loaded_default_data:
            st.session_state.current_prompt_context_json = loaded_default_data
            st.info(f"Default '{PROMPT_CONTEXT_FILE}' loaded.")
        else:
            st.warning(f"Could not load default '{PROMPT_CONTEXT_FILE}'. It might be empty or corrupted.")
    elif st.session_state.current_prompt_context_json is None and not os.path.exists(PROMPT_CONTEXT_FILE):
        st.info(f"Default '{PROMPT_CONTEXT_FILE}' not found. Please upload a file to preview.")


    # --- File uploader for custom JSON (with its own progress bar) ---
    st.markdown("---")
   
    col_json_label, col_json_uploader = st.columns([0.4, 0.6]) # Adjust ratio as needed

    with col_json_label:
        st.markdown("##### Upload Custom JSON") # A smaller, inline heading
        st.write("Upload an alternative Input Prompt JSON File:")

    with col_json_uploader:
        uploaded_file = st.file_uploader(
            "", # Set label to empty string as the label is handled by st.write in the other column
            type="json",
            key="json_uploader"
        )
    
    if uploaded_file is not None:
        st.write("### Loading Uploaded Data...")
        progress_bar_upload = st.progress(0, text="Loading uploaded JSON file...")
        import time
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar_upload.progress(percent_complete + 1, text=f"Loading uploaded JSON file... {percent_complete + 1}%")
        progress_bar_upload.progress(100, text="Uploaded JSON file loaded successfully!") 

        try:
            uploaded_json_data = json.load(uploaded_file)
            st.session_state.current_prompt_context_json = uploaded_json_data
            st.success("Uploaded JSON file loaded successfully!")
            st.session_state.show_json_preview = True
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON.")
            st.session_state.current_prompt_context_json = None
            st.session_state.show_json_preview = False
        except Exception as e:
            st.error(f"An error occurred with the uploaded file: {e}")
            st.session_state.current_prompt_context_json = None
            st.session_state.show_json_preview = False
    
    # --- Button to show/hide the JSON content ---
    if st.session_state.current_prompt_context_json is not None:
        if st.button("👁️ Preview JSON", key="preview_json_button"):
            st.session_state.show_json_preview = not st.session_state.show_json_preview
            st.rerun()
    else:
        st.info("Upload a file or ensure `prompt_context.json` exists to enable JSON preview.")


    # --- Conditionally display the JSON expander based on button click ---
    if st.session_state.get('show_json_preview', False) and st.session_state.current_prompt_context_json is not None:
        with st.expander("JSON Content Preview", expanded=True):
            st.json(st.session_state.current_prompt_context_json)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Hierarchy")
        st.markdown("Explore different AI models, traditional algorithms, and Large Language Models here. Click on a **Use Case** or **LLM Prompt** to view its performance metrics and details appear on the right side. Selecting a new item will automatically replace any previously displayed information.")
        
        traditional_structured_data = load_and_structure_traditional_data(TRADITIONAL_RESULTS_FILE)
        llm_structured_data = load_and_structure_llm_data(LLM_RESULTS_FILE)
        nodes_data = create_tree_nodes(traditional_structured_data, llm_structured_data)

        previous_selected_node_value = st.session_state['selected_tree_node_value']
        
        selected_tree_output = tree_select(
            nodes_data,
            checked=[st.session_state['selected_tree_node_value']] if st.session_state['selected_tree_node_value'] else None,
            expanded=st.session_state['tree_expanded_state'],
            check_model='leaf',
            expand_on_click=True,
            key=f'main_tree_select_{st.session_state["tree_rerender_key"]}'
        )

        should_rerun = False
        newly_checked_from_tree = selected_tree_output.get('checked', [])

        if newly_checked_from_tree:
            current_candidate_value = None
            if len(newly_checked_from_tree) == 1:
                current_candidate_value = newly_checked_from_tree[0]
            elif previous_selected_node_value in newly_checked_from_tree:
                other_items = [item for item in newly_checked_from_tree if item != previous_selected_node_value]
                if other_items:
                    current_candidate_value = other_items[0]
                else: 
                    current_candidate_value = previous_selected_node_value
            else: 
                current_candidate_value = newly_checked_from_tree[0]

            if current_candidate_value and current_candidate_value != st.session_state['selected_tree_node_value']:
                st.session_state['selected_tree_node_value'] = current_candidate_value
                st.session_state['tree_rerender_key'] += 1 
                should_rerun = True

                is_traditional_use_case = False
                for algo_data in traditional_structured_data.values():
                    for model_data in algo_data.values():
                        if current_candidate_value in model_data:
                            is_traditional_use_case = True
                            break
                    if is_traditional_use_case:
                        break
                
                is_llm_prompt = False
                for model_data in llm_structured_data.values():
                    if current_candidate_value in model_data:
                        is_llm_prompt = True
                        break

                if is_traditional_use_case or is_llm_prompt:
                    st.session_state['show_metrics_column'] = True
                    st.session_state['current_metrics_type'] = 'traditional_use_case' if is_traditional_use_case else 'llm_prompt'
                    st.session_state['current_metrics_id'] = current_candidate_value
                else: 
                    st.session_state['show_metrics_column'] = False
                    st.session_state['current_metrics_type'] = None
                    st.session_state['current_metrics_id'] = None
                    st.session_state['selected_tree_node_value'] = None 
                    
                path_to_selected_node = get_path_to_node(nodes_data, current_candidate_value)
                if path_to_selected_node:
                    new_expanded_state = []
                    for node_val in path_to_selected_node:
                        if not (node_val.startswith('traditional_') and 'use_case' in node_val) and \
                           not (node_val.startswith('llm_') and 'prompt' in node_val):
                            new_expanded_state.append(node_val)
                    
                    if current_candidate_value and current_candidate_value.startswith('llm_') and 'llm_prediction_algorithms_root' not in new_expanded_state:
                        new_expanded_state.insert(0, 'llm_prediction_algorithms_root')

                    st.session_state['tree_expanded_state'] = list(set(new_expanded_state))
                else: 
                     st.session_state['tree_expanded_state'] = []

        elif not newly_checked_from_tree and st.session_state['selected_tree_node_value'] is not None:
            st.session_state['show_metrics_column'] = False
            st.session_state['current_metrics_type'] = None
            st.session_state['current_metrics_id'] = None
            st.session_state['selected_tree_node_value'] = None 
            st.session_state['tree_expanded_state'] = [] 
            st.session_state['tree_rerender_key'] += 1 
            should_rerun = True

        if should_rerun:
            st.rerun()

        st.markdown("---") 
        
        st.subheader("LLM Dashboard")
        # Button to navigate to LLM dashboard
        if st.button("📊 Go to LLM Dashboard", key="llm_dashboard_button"):
            st.session_state['current_page'] = 'llm_dashboard'
            # Reset main dashboard specific states if you want a clean slate when returning
            st.session_state['selected_tree_node_value'] = None 
            st.session_state['tree_expanded_state'] = [] 
            st.session_state['tree_rerender_key'] += 1 
            st.rerun()

    with col2:
        # Metric details column content is now only rendered if a model is selected
        if st.session_state['show_metrics_column']:
            display_metrics_in_column(
                st.session_state['current_metrics_type'],
                st.session_state['current_metrics_id'],
                traditional_structured_data,
                llm_structured_data
            )
        # If show_metrics_column is False, this column will simply be empty,
        # providing the desired "not visible" effect.

# --- Main Application Entry Point (Router) ---
# Initialize session state for page control if not already set
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'landing_page' # Start with the landing page

# Initialize other session states needed across pages
if 'selected_tree_node_value' not in st.session_state:
    st.session_state['selected_tree_node_value'] = None
if 'tree_expanded_state' not in st.session_state:
    st.session_state['tree_expanded_state'] = []
if 'tree_rerender_key' not in st.session_state:
    st.session_state['tree_rerender_key'] = 0
if 'current_prompt_context_json' not in st.session_state:
    st.session_state.current_prompt_context_json = None
if 'show_json_preview' not in st.session_state:
    st.session_state.show_json_preview = False
if 'show_metrics_column' not in st.session_state:
    st.session_state['show_metrics_column'] = False
if 'current_metrics_type' not in st.session_state:
    st.session_state['current_metrics_type'] = None
if 'current_metrics_id' not in st.session_state:
    st.session_state['current_metrics_id'] = None


# Conditional rendering of pages based on 'current_page' session state
if st.session_state['current_page'] == 'landing_page':
    landing_page() # Call the function from landing_page_app.py
elif st.session_state['current_page'] == 'dashboard_page': # This now maps to 'tree_view'
    run_dashboard_view()
elif st.session_state['current_page'] == 'llm_dashboard':
    run_llm_dashboard_view(LLM_RESULTS_FILE)