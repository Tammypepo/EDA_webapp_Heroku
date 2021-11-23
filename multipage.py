import streamlit as st
from PIL import Image

# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        self.pages = []


    def add_page(self, title, func) -> None:

        self.pages.append({

                "title": title,
                "function": func
            })


    def run(self):
        # Drodown to select the page to run
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 400px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 500px;
                margin-left: -500px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Tool for Data science</h1>", unsafe_allow_html=True)

        image = Image.open("img/meowdata.png")
        st.sidebar.image(image, use_column_width= True)
        page = st.sidebar.selectbox(
            'Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )

        # run the app function
        page['function']()
