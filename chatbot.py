import json
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # fix path warning from streamlit
import streamlit as st
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


### parameters ###
# api token
with open('api_tokens.json', 'r') as file:
    api_tokens = json.load(file)
api_key = api_tokens['HUGGINGFACEHUB_API_TOKEN']

# define whether to run the model locally or online via the hugging face api
local = False

# model parameters
if local:
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens = 256
    temperature = 0.8
    task = "text-generation"
    device = torch.device("cpu")
else:
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    max_new_tokens = 512
    temperature = 0.8

# base prompt and messages
prompt_base = "You are a helpful assistant and your name is Chatty. Please answer all questions as clearly as possible."
message_intro = "Hello there! My name is Chatty. How can I help you?"


### chatbot ###
# streamlit interface
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# initialize model or client
if local:
    # download and prepare models
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        # show message that model is loading
        with st.spinner("Downloading and preparing the local models..."):
            # download models
            try:
                st.session_state["model"] = AutoModelForCausalLM.from_pretrained(model_name, token=api_key)
                st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_name, token=api_key)
            except Exception as e:
                st.error(f"An error occurred while loading the model: {str(e)}")
                st.stop()

    # initialize llm pipeline
    llm = pipeline(
        task=task,
        model=st.session_state["model"],
        tokenizer=st.session_state["tokenizer"],
        device=device
    )
else:
    # initialize llm client
    client = InferenceClient(
        api_key=api_key
    )

# initialize chat history including the base prompt and other state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "system", "content": prompt_base})
    st.session_state.chat_history.append({"role": "assistant", "content": message_intro})
    st.session_state.is_processing = False
    st.session_state.prompt = None

    # refresh page to clear downloading message
    st.rerun()

# display chat messages from the whole history
for message in st.session_state.chat_history:
    if message["role"]=="user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"]=="assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# create bar for the prompt with example input
if not st.session_state.is_processing:
    st.session_state.prompt = st.chat_input("Type your question here ...", disabled=False)
else:
    st.chat_input("Type your question here ...", disabled=True)

# query llm if new prompt from user entered
if st.session_state.prompt:
    # reload page to disable prompt bar
    if not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.rerun()

    # show message from user and append it to the chat history
    with st.chat_message("user"):
        st.markdown(st.session_state.prompt)
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.prompt})

    # query answer from local or online llm
    if local:
        response = llm(
            text_inputs=st.session_state.chat_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        response = response[0]["generated_text"][-1]["content"]
    else:
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=st.session_state.chat_history,
                max_tokens=max_new_tokens,
            )
            try:
                response = response["choices"][0]["message"]["content"]
            except:
                st.error(f"An error occurred while querying: {str(response)}")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred while querying: {str(e)}")
            st.stop()

    # show message from assistant and append it to the chat history
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # reload page to enable prompt bar
    st.session_state.is_processing = False
    st.session_state.prompt = None
    st.rerun()
