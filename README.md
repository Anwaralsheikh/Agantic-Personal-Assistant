# ✦ Agentic RAG Personal Assistant

A production-ready agentic RAG (Retrieval-Augmented Generation) system that combines:
- **Multimodal document understanding** (PDF, DOCX, TXT, Images)
- **vector search** for semantic retrieval (or local fallback)
- **Web search** via DuckDuckGo for real-time information
- **LangChain agents** with automatic tool selection
- **LangSmith tracing** for observability


## Requirements

- Python 3.11.14

#### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n <env name> python=3.11.14
```
3) Activate the environment:
```bash
$ conda activate <env name>
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```
Set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.

