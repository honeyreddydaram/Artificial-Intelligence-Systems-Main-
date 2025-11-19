# streamlit_app/pages/__init__.py
# Make the page modules importable
import sys
import os

# Add the parent directory to Python's module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules directly
from streamlit_app.pages.home import show
from streamlit_app.pages.analysis import show
from streamlit_app.pages.classification import show
from streamlit_app.pages.visualization import show

# For easier access, create aliases
home = sys.modules['streamlit_app.pages.home']
analysis = sys.modules['streamlit_app.pages.analysis']
visualization = sys.modules['streamlit_app.pages.visualization']
classification = sys.modules['streamlit_app.pages.classification']