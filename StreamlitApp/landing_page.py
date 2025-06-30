# landing_page_app.py
import streamlit as st
import base64 # To store image in session state as base64 string

def landing_page():
    """Renders the initial landing screen with image upload and navigation to dashboard."""
    st.title("💡 Clinical Context: AI Model Evaluation Hub")

    st.markdown("---") # Separator

    # Initialize session state for storing the image data
    if 'project_image_data' not in st.session_state:
        st.session_state['project_image_data'] = None
    if 'project_image_filename' not in st.session_state:
        st.session_state['project_image_filename'] = None
    if 'show_image_uploader' not in st.session_state:
        st.session_state['show_image_uploader'] = True # Control visibility of uploader

    # --- Image Upload/Display Section ---
    # Determine what to display based on whether an image is present in session state
    if st.session_state['project_image_data'] is None:
        # If no image is uploaded, show the file uploader
        st.info("Upload a project image to personalize this page.")
        uploaded_file = st.file_uploader(
            "Upload an image (e.g., logo, banner)", # Simpler prompt
            type=["png", "jpg", "jpeg"],
            key="initial_image_uploader"
        )
        if uploaded_file is not None:
            # Read image as bytes, then encode to base64 for session state storage
            image_bytes = uploaded_file.getvalue()
            st.session_state['project_image_data'] = base64.b64encode(image_bytes).decode('utf-8')
            st.session_state['project_image_filename'] = uploaded_file.name
            st.session_state['show_image_uploader'] = False # Hide uploader after successful upload
            st.rerun() # Rerun to update the display

    else:
        # If image data exists in session state, display the image and the replace button
        image_bytes = base64.b64decode(st.session_state['project_image_data'])
        
        # Display the image covering the section
        # --- CORRECTED LINE HERE ---
        st.image(image_bytes, 
                 caption=f"Current Image: {st.session_state['project_image_filename']}", 
                 use_container_width=True) # Changed from use_column_width to use_container_width

        # Provide a "Replace Image" button
        if st.button("✏️ Replace Image", key="replace_image_button"):
            st.session_state['project_image_data'] = None # Clear image data
            st.session_state['project_image_filename'] = None
            st.session_state['show_image_uploader'] = True # Show uploader again
            st.rerun() # Rerun to switch back to the uploader view
        
        st.markdown(
            """
            <style>
            .stFileUploader {
                display: none; /* Hide the file uploader widget if it's rendered by default */
            }
            </style>
            """,
            unsafe_allow_html=True # This CSS might be useful if the uploader flickers or is stubbornly visible
        )
        st.info("To replace the image, click the '✏️ Replace Image' button.")


    st.markdown("---") # Separator

    st.markdown("""
    Welcome to the **AI Model Evaluation Hub for Clinical Context**!

    This platform is designed to provide a comprehensive overview of various AI model performances,
    including traditional machine learning algorithms and advanced Large Language Models (LLMs),
    specifically tailored for clinical applications and healthcare scenarios.

    **What you can do here:**
    - **Explore Model Performance:** Navigate a structured hierarchy of AI models (both traditional and LLM-based) and delve into their performance metrics for specific use cases or prompts.
    - **Analyze Key Metrics:** View detailed metrics for each model, often visualized with radial charts for quick and intuitive understanding of strengths and weaknesses.
    - **Manage Prompt Contexts:** Easily upload and preview JSON files that define the prompt contexts used in LLM evaluations, ensuring transparency and reproducibility.

    Our mission is to foster a deeper understanding and transparent evaluation of AI capabilities in healthcare, empowering researchers, clinicians, and data scientists to make informed decisions and advance the integration of AI responsibly.
    """)
    st.markdown("---")
    
    if st.button("🚀 Go to Dashboard", key="go_to_dashboard_button_landing"):
        st.session_state['current_page'] = 'dashboard_page'
        st.rerun()