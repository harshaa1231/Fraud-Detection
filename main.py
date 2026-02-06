"""
Main entry point - Streamlit Cloud will run this file.
This redirects to the actual Streamlit app in app.py
"""
import sys
import os

# Ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The actual Streamlit app code would be here or imported from app.py
# For Streamlit Cloud, it's better to use app.py as the main file directly
# But if Streamlit Cloud is pointing to main.py, we need the actual app code here

# Import the main function/code from app.py would go here
# But since app.py is a complete Streamlit app, it's best to point Streamlit to it directly
