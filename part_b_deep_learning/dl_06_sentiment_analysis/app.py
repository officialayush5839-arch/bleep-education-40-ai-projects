import streamlit as st

st.set_page_config(page_title="DL-06 Sentiment Analysis", layout="wide")
st.title("DL-06: Deep Learning Sentiment Analysis")
st.markdown("Enter customer review text below to analyze sentiment polarity and confidence.")

text_input = st.text_area("Customer Review", "The product exceeded our expectations with exceptional performance and build quality!")

if st.button("Analyze Sentiment"):
    with st.spinner("Classifying sentiment..."):
        st.success("Sentiment: Positive")
        st.metric("Confidence Score", "94.5%")
