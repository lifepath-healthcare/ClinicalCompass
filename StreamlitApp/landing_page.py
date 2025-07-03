import streamlit as st

def landing_page():
    """Renders the initial landing screen with image upload and navigation to dashboard."""
    st.title("💡 Clinical Context")

    st.markdown("---") # Separator

    st.markdown("""
    Welcome to the **AI Model Evaluation Hub for Clinical Context**!

    This platform is designed to provide a comprehensive overview of various LLM model performances,
    including traditional machine learning algorithms and advanced Large Language Models (LLMs),
    specifically tailored for clinical applications and healthcare scenarios.

    **What you can do here:**
    - **Explore Model Performance:** Navigate a structured hierarchy of LLM models and delve into their performance metrics for specific use cases or prompts.
    - **Analyze Key Metrics:** View detailed metrics for each model, often visualized with radial charts for quick and intuitive understanding of strengths and weaknesses.
   
    Our mission is to foster a deeper understanding and transparent evaluation of AI capabilities in healthcare, empowering researchers, clinicians, and data scientists to make informed decisions and advance the integration of AI responsibly.
    """)
    st.markdown("---")
    
    if st.button("🚀 Go to LLM Overall Prompts Analysis", key="go_to_dashboard_button_landing"):
        st.session_state['current_page'] = 'llm_overall_analysis_dashboard'
        st.rerun()