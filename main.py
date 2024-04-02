import streamlit as st
# Install Modules
# !pip install transformers==2.8.0
# !pip install torch==1.4.0
# !pip install datasets transformers[sentencepiece]
# !pip install sentencepiece

st.set_page_config(layout="wide")

# Import Module
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from PyPDF2 import PdfReader

# initialize the pretrained model
# model = T5ForConditionalGeneration.from_pretrained('t5-base')
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')



def preprocess_text(text):
    # Replace the special characters with the correct quotation marks
    text = text.replace("?", '"').replace("?", "'")

    # Remove the extra spaces and newlines
    text = text.strip().replace("\n", " ")

    # Add a period at the end of the text if it is missing
    if not text.endswith("."):
        text = text + "."

    # Prepend the text with the prefix "summarize: " to indicate the task to the model
    t5_prepared_Text = "summarize: " + text

    return t5_prepared_Text


@st.cache_resource
def text_summary(text):
    t5_prepared_Text = preprocess_text(text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=3,
                                 min_length=30,
                                 max_length=200,
                                 length_penalty=2.0,
                                 temperature=0.8)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output





def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text


choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using T5")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document using T5")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
