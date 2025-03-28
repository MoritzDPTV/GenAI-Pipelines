# GenAI Pipelines
This repository provides mulitiple GenAI pipelines as presented in the following. All pipelines are built using [Hugging Face](https://huggingface.co/) and [Streamlit](https://streamlit.io/).


## Pipelines
The following pipelines are available in this repository:
- `chatbot.py`: Chatbot that can be run both locally and online via the Hugging Face API
- `q_and_a_system.py`: Q&A query system with RAG that allows to parse plain texts, links and PDF files
- `gen_ai_pipeline.py`: img2text2story2img&audio pipeline for experimenting with GenAI models across different modalities (text, image, audio)


## Setup
In order to run the scripts, the environment must first be set up using the terminal:
```sh
$ git clone https://github.com/MoritzDPTV/GenAI-Pipelines.git
$ cd GenAI-Pipelines
$ python3 -m venv venv
$ source venv/bin/activate  # for Linux/MacOS
$ venv\Scripts\activate  # for Windows
$ pip install -r requirements.txt
```

As the last step, a personal [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens/) must be placed in the `api_tokens.json` file:
```json
{
  "HUGGINGFACEHUB_API_TOKEN": "place_api_token_here"
}
```


## Running the Scripts
With everything set up, the scripts can be run with the following commands in the terminal, whereby the placeholder must be replaced with the name of the script of interest:
```bash
$ streamlit run "script_name"
```


## Configurations
All scripts provide adjustable parameters at the beginning, such as the models used for queries, whether the pipeline should run locally or online, and so on.


## Preview

### Chatbot
<p align="center">
  <img src="https://carloconnect.com/externals/git/chatbot.jpg" width="600"/>
</p>

### Q&A System
<p align="center">
  <img src="https://carloconnect.com/externals/git/q_and_a_system.jpg" width="600"/>
</p>

### GenAI Pipeline
<p align="center">
  <img src="https://carloconnect.com/externals/git/gen_ai_pipeline.jpg" width="600"/>
</p>
