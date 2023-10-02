import streamlit as st
import langchain_helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="youtube_assistant"):
        youtube_url = st.sidebar.text_input(
            label="What is the YouTube video URL?",
            key="youtube_url",
        )
        query = st.sidebar.text_area(
            label="Ask about the video?",
            max_chars=100,
            key="query",
        )
        submit_btn = st.form_submit_button(label="Submit")

if query and youtube_url and submit_btn:
    db = lch.create_vectordb_from_youtube(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
