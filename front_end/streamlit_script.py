import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd

from llm_integration.recommender_system_flow import RecommenderWizard, WizardState, S3Cache, Step

# ──────────────────────────────────────────
# Initialize session state
# ──────────────────────────────────────────
if "wizard_state" not in st.session_state:
    st.session_state.wizard_state = WizardState()

if "cache" not in st.session_state:
    st.session_state.cache = S3Cache()


def get_wizard() -> RecommenderWizard:
    """Get wizard instance with current state"""
    return RecommenderWizard(
        state=st.session_state.wizard_state,
        cache=st.session_state.cache
    )


def save_state(wizard: RecommenderWizard):
    """Save wizard state back to session"""
    st.session_state.wizard_state = wizard.state


# ──────────────────────────────────────────
# Process compute steps automatically
# ──────────────────────────────────────────
def run_compute_steps():
    """Run all pending compute steps"""
    wizard = get_wizard()
    
    while wizard.is_compute_step() and not wizard.is_done():
        wizard.process_compute_step()
        save_state(wizard)
    
    return wizard


# ──────────────────────────────────────────
# Main App
# ──────────────────────────────────────────
st.title("League Champion Recommender")

# Run any pending compute steps
wizard = run_compute_steps()

# Check if done
if wizard.is_done():
    if wizard.state.current_step == Step.SHOW_RECOMMENDATIONS:
        st.header("Your Personalized Champion Recommendations")
        
        recs = wizard.state.recommendations
        if recs:
            for key, category in recs.items():
                st.subheader(category.get("title", key))
                
                champions = category.get("champions", [])
                descriptions = category.get("descriptions", {})
                
                for champ in champions:
                    desc = descriptions.get(champ, "")
                    st.markdown(f"**• {champ}** — {desc}")
                
                st.divider()
        
        if st.button("Start Over"):
            st.session_state.wizard_state = WizardState()
            st.session_state.cache = S3Cache()
            st.rerun()
        
        st.stop()
    
    elif wizard.state.current_step == Step.ERROR:
        st.error(wizard.state.error_message or "An error occurred.")
        
        if st.button("Start Over"):
            st.session_state.wizard_state = WizardState()
            st.rerun()
        
        st.stop()

# Get UI config for current step
ui_config = wizard.get_ui_config()

# Show loading state
if ui_config.is_loading:
    with st.spinner(ui_config.question or "Processing..."):
        wizard.process_compute_step()
        save_state(wizard)
        st.rerun()

# Show error if any
if ui_config.error_message and ui_config.input_type == "none":
    st.error(ui_config.error_message)
    st.stop()

# Display question
if ui_config.question:
    st.subheader(ui_config.question)

# Handle different input types
selected = None

# ──────────────────────────────────────────
# Button Input
# ──────────────────────────────────────────
if ui_config.input_type == "button":
    cols = st.columns(len(ui_config.choices)) if len(ui_config.choices) <= 5 else [st.container()]
    
    if len(ui_config.choices) <= 5:
        for i, choice in enumerate(ui_config.choices):
            with cols[i]:
                if st.button(choice, key=f"btn_{choice}", use_container_width=True):
                    selected = choice
    else:
        for choice in ui_config.choices:
            if st.button(choice, key=f"btn_{choice}"):
                selected = choice

# ──────────────────────────────────────────
# Text Input
# ──────────────────────────────────────────
elif ui_config.input_type == "text":
    user_input = st.text_input(
        "Enter value:",
        placeholder=ui_config.placeholder or "",
        key="text_input",
        label_visibility="collapsed"
    )
    
    if st.button("Submit", key="submit_btn"):
        if user_input:
            selected = user_input
        else:
            st.warning("Please enter a value.")

# ──────────────────────────────────────────
# Dropdown Input
# ──────────────────────────────────────────
elif ui_config.input_type == "dropdown":
    dropdown_value = st.selectbox(
        "Select:",
        options=ui_config.choices,
        index=None,
        placeholder=ui_config.placeholder or "Choose an option",
        key="dropdown_input",
        label_visibility="collapsed"
    )
    
    if st.button("Confirm", key="confirm_dropdown"):
        if dropdown_value:
            selected = dropdown_value
        else:
            st.warning("Please select an option.")

# ──────────────────────────────────────────
# Selectbox with Dataframe
# ──────────────────────────────────────────
elif ui_config.input_type == "selectbox":
    if ui_config.dataframe:
        st.dataframe(pd.DataFrame(ui_config.dataframe), use_container_width=True)
    
    selectbox_value = st.selectbox(
        "Select:",
        options=ui_config.choices,
        index=None,
        key="selectbox_input",
        label_visibility="collapsed"
    )
    
    if st.button("Confirm Selection", key="confirm_selectbox"):
        if selectbox_value:
            selected = selectbox_value
        else:
            st.warning("Please select an option.")

# ──────────────────────────────────────────
# Continue Button
# ──────────────────────────────────────────
elif ui_config.input_type == "continue":
    if st.button("Continue →", key="continue_btn"):
        selected = "__continue__"

# ──────────────────────────────────────────
# Handle Selection
# ──────────────────────────────────────────
if selected:
    wizard.submit_input(selected)
    save_state(wizard)
    st.rerun()

# ──────────────────────────────────────────
# Progress indicator
# ──────────────────────────────────────────
st.divider()

# Show current step for debugging (optional - remove in production)
with st.expander("Debug Info"):
    st.write(f"Current Step: {wizard.state.current_step.name}")
    st.write(f"Role: {wizard.state.role}")
    st.write(f"Use Own Data: {wizard.state.use_own_data}")
    st.write(f"User Champion: {wizard.state.user_champion}")
    st.write(f"Decision Method: {wizard.state.decision_making_method}")