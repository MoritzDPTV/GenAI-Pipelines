import json
import requests
import streamlit as st


### parameters ###
# api token
with open('api_tokens.json', 'r') as file:
    api_tokens = json.load(file)
HUGGINGFACEHUB_API_TOKEN = api_tokens['HUGGINGFACEHUB_API_TOKEN']
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# model parameters
max_new_tokens = 512
temperature = 0.8
model_api_url_img2text = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"
model_api_url_textGen = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions"
model_api_url_text2image = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
model_api_url_text2audio = "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small"


### pipeline ###
# streamlit interface
st.set_page_config(page_title="GenAI Pipeline")
st.title("GenAI Pipeline")
st.write("Provide an image to run the img2text2story2img&audio pipeline.")

# init variables
selected_image = None

# interface to select an image and run the pipeline
with st.form(key="input_image", clear_on_submit=True, border=False):
    with st.container(border=True):
        # image selecter and button to run the pipeline
        selected_image = st.file_uploader("Select an Image", type=["jpg", "jpeg", "png"])
        process_button = st.form_submit_button("Gernerate AI Content")

    # run pipeline
    if process_button:
        if selected_image is not None:
            # show provided image
            with st.expander("Provided Image", expanded=True):
                st.image(selected_image, use_container_width=True)

            # run image to text model
            response = requests.post(
                model_api_url_img2text,
                headers={"Content-Type": "image/jpeg", **headers},
                data=selected_image
            )
            try:
                response = response.json()[0]["generated_text"]
            except:
                st.error(f"An error occurred while querying img2text model: '{response}, {response.reason}'")
                st.stop()
            with st.expander("Image Description Generated from the Provided Image", expanded=True):
                st.write(response)

            # run text generation model
            prompt = f"In one sentence, tell a story about {response}."
            response = requests.post(
                model_api_url_textGen,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature
                }
            )
            try:
                response = response.json()["choices"][0]["message"]["content"]
            except:
                st.error(f"An error occurred while querying text generation model: '{response}, {response.reason}'")
                st.stop()
            with st.expander("Story Generated from the Image Description", expanded=True):
                st.write(response)

            # run text to image model
            response_img = requests.post(
                model_api_url_text2image,
                headers=headers,
                json={"inputs": response, "temperature": temperature}
            )
            try:
                if "error" in str(response_img.content):
                    raise Exception()
                response_img = response_img.content
            except Exception as e:
                st.error(f"An error occurred while querying text2img model: '{response_img}, {response_img.reason}'")
                st.stop()
            with st.expander("Image Generated from the Story", expanded=True):
                try:
                    st.image(response_img, use_container_width=True)
                except:
                    st.error(f"Error receiving the image. Please try again.")
                    st.stop()

            # run text to audio model
            response_audio = requests.post(
                model_api_url_text2audio,
                headers=headers,
                json={"inputs": response, "temperature": temperature}
            )
            try:
                if "error" in str(response_audio.content):
                    raise Exception()
                response_audio = response_audio.content
            except:
                st.error(f"An error occurred while querying text2audio model: '{response_audio}, {response_audio.reason}'")
                st.stop()
            with st.expander("Audio Generated from the Story", expanded=True):
                st.audio(response_audio)
        else:
            # show message that image must be provided
            st.error("Please provide an image to proceed.")

            # show placeholders
            with st.expander("Image Description Generated from the Provided Image", expanded=False):
                st.markdown("")
            with st.expander("Story Generated from the Image Description", expanded=False):
                st.markdown("")
            with st.expander("Image Generated from the Story", expanded=False):
                st.markdown("")
            with st.expander("Audio Generated from the Story", expanded=False):
                st.markdown("")
    else:
        # show placeholders
        with st.expander("Image Description Generated from the Provided Image", expanded=False):
            st.markdown("")
        with st.expander("Story Generated from the Image Description", expanded=False):
            st.markdown("")
        with st.expander("Image Generated from the Story", expanded=False):
            st.markdown("")
        with st.expander("Audio Generated from the Story", expanded=False):
            st.markdown("")
