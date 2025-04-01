import faiss
import json
import os
os.environ['USER_AGENT'] = "my_agent"  # fix environmental variable warning from langchain
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # fix path warning from streamlit
import streamlit as st
from io import BytesIO
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from PyPDF2 import PdfReader


### parameters ###
# api token
with open("api_tokens.json", "r") as file:
    api_tokens = json.load(file)
api_key = api_tokens["HUGGINGFACEHUB_API_TOKEN"]

# model parameters
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
generation_model_name = "mistralai/Mistral-Nemo-Instruct-2407"
max_new_tokens = 512
temperature = 0.8


### functions ###
# preprocess input data
def textPreprocessing(input_type, input_data):
    # gather text from the provided source
    documents = ""
    if input_type == "Text":
        documents = input_data
    elif input_type == "PDF":
        for file in input_data:
            pdf_reader = PdfReader(BytesIO(file.read()))
            for page in pdf_reader.pages:
                documents += page.extract_text()
    elif input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        documents = str(documents[0].page_content)
    else:
        raise ValueError("Unsupported input type.")

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(documents)

    # return preprocessed texts
    return texts


# initialize the vectorstore according to hugging face embeddings
def initVectorstore():
    # initialize hugging face embedding function
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.stop()

    # create faiss index and vectorstore
    index = faiss.IndexFlatL2(len(hf_embeddings.embed_query("sample text")))
    vectorstore = FAISS(
        embedding_function=hf_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # return vectorstore
    return vectorstore


# query llm with prompt
def queryLLM(vectorstore, query):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=generation_model_name,
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = qa.invoke(query)
        try:
            return result["result"]
        except:
            st.error(f"An error occurred while querying: {str(result)}")
            st.stop()
    except Exception as e:
        st.error(f"An error occurred while querying: {str(e)}")
        st.stop()


### chatbot ###
# streamlit interface
st.set_page_config(page_title="Q&A System")
st.title("Q&A System")
st.write("Provide a text, a link, or PDF documents to ask questions based on their content. All processed input data are stored in the knowledge base until you press reset.")

# initialize variables
input_data = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "vectorstore" not in st.session_state:
    with st.spinner("Downloading the embedding model and preparing the vector database..."):
        st.session_state["vectorstore"] = initVectorstore()
        st.rerun()

# container for consistens layout
with st.container(border=True):
    # selection box for different inputs
    input_type = st.selectbox(label="Input Type", options=["Text", "PDF", "Link"])

    # input form for different types
    with st.form(key="input_data", clear_on_submit=True, border=False):
        if input_type == "Text":
            input_data = st.text_area("Enter Text")
        elif input_type == "PDF":
            input_data = st.file_uploader("Select PDF File(s)", type=["pdf"], accept_multiple_files=True)
        elif input_type == "Link":
            input_data = st.text_input("Enter Link").strip()

        # process the input data and append it to the vectorstore
        process_button = st.form_submit_button("Process Input")
        if process_button:
            if input_type == "Text" and not input_data:
                st.error("Please enter a text.")
            elif input_type == "PDF" and not input_data:
                st.error("Please provide at least one file.")
            elif input_type == "Link" and not input_data:
                st.error("Please enter a valid link.")
            else:
                try:
                    texts = textPreprocessing(input_type, input_data)
                    st.session_state["vectorstore"].add_texts(texts)
                    st.success("Input data processed successfully and added to the knowledge base!")
                except Exception as e:
                    st.error(f"An error occurred while processing the input data: {str(e)}")

# query form
with st.form(key="input_question", clear_on_submit=True, enter_to_submit=True):
    # get question from user
    query = st.text_input("Enter Question")

    # query answer from llm
    submit_button = st.form_submit_button("Submit")
    if submit_button:
        if query.strip():
            answer = queryLLM(st.session_state["vectorstore"], query)
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {answer}")
            st.session_state["history"].append((query, answer))
        else:
            st.error("Please enter a question.")

# q&a history
with st.expander(label="Q&A History"):
    if len(st.session_state["history"]) == 0:
        st.markdown("")
    else:
        for i, (q, a) in enumerate(st.session_state["history"], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

# reset the vectorstore and history if reset button pressed
if st.button("Reset"):
    st.session_state["vectorstore"] = initVectorstore()
    st.session_state["history"] = []
    st.rerun()
