import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import utils, eda, build_model, fe, about # import your pages here

# Create an instance of the app
app = MultiPage()

# Add all your applications (pages) here
app.add_page("Exploratory Data Analysis", eda.app)
app.add_page("Feture Engineering",fe.app)
app.add_page("Machine learning Model", build_model.app)
app.add_page("About",about.app)

# The main app
app.run()
