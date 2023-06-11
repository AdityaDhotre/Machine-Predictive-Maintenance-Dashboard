import streamlit as st
import page1
import predictive_maintenance_demo


# Page navigation
PAGES = {
    "Data Visualization": page1,
    "Maintenance Predictor": predictive_maintenance_demo
}


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Display the selected page with the help of submodules
    page = PAGES[selection]
    page.main()


if __name__ == '__main__':
    main()
