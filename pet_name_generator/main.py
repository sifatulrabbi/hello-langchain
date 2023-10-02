import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")

with st.sidebar:
    with st.form(key="pet_name_generator"):
        animal_type = st.sidebar.selectbox(
            "What is your pet?",
            ("Cat", "Dog", "Rabbit", "Hamster"),
        )
        pet_color = st.sidebar.text_area(
            f"What is your {animal_type}'s color?",
            max_chars=100,
        )
        submit_btn = st.form_submit_button(label="Submit", type="primary")

if animal_type and pet_color and submit_btn:
    response = lch.generate_pet_name(animal_type, pet_color)
    st.subheader("Names for your Pet:")
    st.text(response.get("pet_name"))
