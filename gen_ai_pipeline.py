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
model_api_url_textGen = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"
model_api_url_text2audio = "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small"
model_api_url_text2image = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"


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
            image_description = requests.post(
                model_api_url_img2text,
                headers={"Content-Type": "image/jpeg", **headers},
                data=selected_image
            ).json()[0]["generated_text"]
            with st.expander("Image Description Generated from the Provided Image", expanded=True):
                st.write(image_description)

            # run text generation model
            prompt = f"In one sentence, tell a story about {image_description}."
            story = requests.post(
                model_api_url_textGen,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature
                }
            ).json()["choices"][0]["message"]["content"]
            with st.expander("Story Generated from the Image Description", expanded=True):
                st.write(story)

            # run text to image model
            illustration_visual = requests.post(
                model_api_url_text2image,
                headers=headers,
                json={"inputs": story, "temperature": temperature}
            ).content
            with st.expander("Image Generated from the Story", expanded=True):
                try:
                    st.image(illustration_visual, use_container_width=True)
                except:
                    st.error("Error receiving the image. Please try again.")

            # run text to audio model
            illustration_audio = requests.post(
                model_api_url_text2audio,
                headers=headers,
                json={"inputs": story, "temperature": temperature}
            ).content
            with st.expander("Audio Generated from the Story", expanded=True):
                st.audio(illustration_audio)
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

