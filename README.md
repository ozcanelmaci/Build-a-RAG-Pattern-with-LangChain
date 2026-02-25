# Build-a-RAG-Pattern-with-LangChain
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ozcanelmaci/Build-a-RAG-Pattern-with-LangChain/blob/main/Build_a_RAG_Pattern_with_LangChain.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "IhtrGBl93dU4"
      },
      "source": [
        "# Retrieval Augmented Generation (RAG) with Langchain\n",
        "*Using IBM Granite Models*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "usNx-LTZ3dU6"
      },
      "source": [
        "## In this notebook\n",
        "This notebook contains instructions for performing Retrieval Augumented Generation (RAG). RAG is an architectural pattern that can be used to augment the performance of language models by recalling factual information from a knowledge base, and adding that information to the model query. The most common approach in RAG is to create dense vector representations of the knowledge base in order to retrieve text chunks that are semantically similar to a given user query.\n",
        "\n",
        "RAG use cases include:\n",
        "- Customer service: Answering questions about a product or service using facts from the product documentation.\n",
        "- Domain knowledge: Exploring a specialized domain (e.g., finance) using facts from papers or articles in the knowledge base.\n",
        "- News chat: Chatting about current events by calling up relevant recent news articles.\n",
        "\n",
        "In its simplest form, RAG requires 3 steps:\n",
        "\n",
        "- Initial setup:\n",
        "  - Index knowledge-base passages for efficient retrieval. In this recipe, we take embeddings of the passages, and store them in a vector database.\n",
        "- Upon each user query:\n",
        "  - Retrieve relevant passages from the database. In this recipe, we use an embedding of the query to retrieve semantically similar passages.\n",
        "  - Generate a response by feeding retrieved passage into a large language model, along with the user query."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "fRoZlbVD3dU8"
      },
      "source": [
        "## Setting up the environment"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tiGMw-rW3dU8"
      },
      "source": [
        "Install dependencies."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%%\n"
        },
        "id": "HfwGDR7g3dU9"
      },
      "outputs": [],
      "source": [
        "! echo \"::group::Install Dependencies\"\n",
        "%pip install uv\n",
        "! uv pip install git+https://github.com/ibm-granite-community/utils.git \\\n",
        "    transformers \\\n",
        "    langchain_classic \\\n",
        "    langchain_community \\\n",
        "    langchain_text_splitters \\\n",
        "    langchain_huggingface sentence_transformers \\\n",
        "    langchain_milvus 'pymilvus[milvus_lite]' \\\n",
        "    'langchain_replicate @ git+https://github.com/ibm-granite-community/langchain-replicate.git' \\\n",
        "    wget\n",
        "! echo \"::endgroup::\""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AyNzUnYh3dU-"
      },
      "source": [
        "## Selecting System Components"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zR9r1Hyn3dU_"
      },
      "source": [
        "### Choose your Embeddings Model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nVN8BQzK3dU_"
      },
      "source": [
        "Specify the model to use for generating embedding vectors from text.\n",
        "\n",
        "To use a model from a provider other than Huggingface, replace this code cell with one from [this Embeddings Model recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Embeddings_Models.ipynb)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ygyAdImR3dVA"
      },
      "outputs": [],
      "source": [
        "from langchain_huggingface import HuggingFaceEmbeddings\n",
        "from transformers import AutoTokenizer\n",
        "\n",
        "embeddings_model_path = \"ibm-granite/granite-embedding-30m-english\"\n",
        "embeddings_model = HuggingFaceEmbeddings(\n",
        "    model_name=embeddings_model_path,\n",
        ")\n",
        "embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0DZ8Hgk13dVB"
      },
      "source": [
        "### Choose your Vector Database\n",
        "\n",
        "Specify the database to use for storing and retrieving embedding vectors.\n",
        "\n",
        "To connect to a vector database other than Milvus substitute this code cell with one from [this Vector Store recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Vector_Stores.ipynb)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "q-r8VmNL3dVB"
      },
      "outputs": [],
      "source": [
        "from langchain_milvus import Milvus\n",
        "import tempfile\n",
        "\n",
        "db_file = tempfile.NamedTemporaryFile(prefix=\"milvus_\", suffix=\".db\", delete=False).name\n",
        "print(f\"The vector database will be saved to {db_file}\")\n",
        "\n",
        "vector_db = Milvus(\n",
        "    embedding_function=embeddings_model,\n",
        "    connection_args={\"uri\": db_file},\n",
        "    auto_id=True,\n",
        "    index_params={\"index_type\": \"AUTOINDEX\"},\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "i4EhTU6y3dVC"
      },
      "source": [
        "### Choose your LLM\n",
        "The LLM will be used for answering the question, given the retrieved text.\n",
        "\n",
        "Select a Granite Code model from the [`ibm-granite`](https://replicate.com/ibm-granite) org on Replicate. Here we use the Replicate Langchain client to connect to the model.\n",
        "\n",
        "To connect to a model on a provider other than Replicate, substitute this code cell with one from the [LLM component recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_LLMs.ipynb)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-9i681s73dVC"
      },
      "outputs": [],
      "source": [
        "from langchain_replicate import ChatReplicate\n",
        "from ibm_granite_community.notebook_utils import get_env_var\n",
        "\n",
        "model_path = \"ibm-granite/granite-4.0-h-small\"\n",
        "model = ChatReplicate(\n",
        "    model=model_path,\n",
        "    replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "aBZtUgEj3dVC"
      },
      "source": [
        "## Building the Vector Database\n",
        "\n",
        "In this example, we take the State of the Union speech text, split it into chunks, derive embedding vectors using the embedding model, and load it into the vector database for querying."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NdY1hbxY3dVD"
      },
      "source": [
        "### Download the document\n",
        "\n",
        "Here we use President Biden's State of the Union address from March 1, 2022."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%%\n"
        },
        "id": "kleGzmm23dVD"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import wget\n",
        "\n",
        "filename = 'state_of_the_union.txt'\n",
        "url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'\n",
        "\n",
        "if not os.path.isfile(filename):\n",
        "  wget.download(url, out=filename)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WRVWmahC3dVD"
      },
      "source": [
        "### Split the document into chunks\n",
        "\n",
        "Split the document into text segments that can fit into the model's context window."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%%\n"
        },
        "id": "8hz2fwJe3dVE"
      },
      "outputs": [],
      "source": [
        "from langchain_community.document_loaders import TextLoader\n",
        "from langchain_text_splitters import CharacterTextSplitter\n",
        "\n",
        "loader = TextLoader(filename)\n",
        "documents = loader.load()\n",
        "text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(\n",
        "    tokenizer=embeddings_tokenizer,\n",
        "    chunk_size=embeddings_tokenizer.max_len_single_sentence,\n",
        "    chunk_overlap=0,\n",
        ")\n",
        "texts = text_splitter.split_documents(documents)\n",
        "doc_id = 0\n",
        "for text in texts:\n",
        "    text.metadata[\"doc_id\"] = (doc_id:=doc_id+1)\n",
        "print(f\"{len(texts)} text document chunks created\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "VSpj0VhP3dVE"
      },
      "source": [
        "### Populate the vector database\n",
        "\n",
        "NOTE: Population of the vector database may take over a minute depending on your embedding model and service."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1VIMCxoq3dVE"
      },
      "outputs": [],
      "source": [
        "ids = vector_db.add_documents(texts)\n",
        "print(f\"{len(ids)} documents added to the vector database\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "F4B5uXJ03dVE"
      },
      "source": [
        "## Querying the Vector Database"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Aig7gmnn3dVF"
      },
      "source": [
        "### Conduct a similarity search\n",
        "\n",
        "Search the database for similar documents by proximity of the embedded vector in vector space."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H3zYSt8A3dVF"
      },
      "outputs": [],
      "source": [
        "query = \"What did the president say about Ketanji Brown Jackson?\"\n",
        "docs = vector_db.similarity_search(query)\n",
        "print(f\"{len(docs)} documents returned\")\n",
        "for doc in docs:\n",
        "    print(doc)\n",
        "    print(\"=\" * 80)  # Separator for clarity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "US2jTQcC3dVF"
      },
      "source": [
        "## Answering Questions"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2PtSAEuh3dVF"
      },
      "source": [
        "### Automate the RAG pipeline\n",
        "\n",
        "Build a RAG chain with the model and the document retriever.\n",
        "\n",
        "First we create the prompts for Granite to perform the RAG query. We use the Granite chat template and supply the placeholder values that the LangChain RAG pipeline will replace.\n",
        "\n",
        "Next, we construct the RAG pipeline by using the Granite prompt templates previously created."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OZzxPUy43dVG"
      },
      "outputs": [],
      "source": [
        "from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain\n",
        "from langchain_classic.chains.retrieval import create_retrieval_chain\n",
        "from langchain_core.prompts import ChatPromptTemplate\n",
        "\n",
        "# Create a Granite prompt for question-answering with the retrieved context\n",
        "prompt_template = ChatPromptTemplate.from_template(\"{input}\")\n",
        "\n",
        "# Assemble the retrieval-augmented generation chain\n",
        "combine_docs_chain = create_stuff_documents_chain(\n",
        "    llm = model,\n",
        "    prompt = prompt_template,\n",
        ")\n",
        "rag_chain = create_retrieval_chain(\n",
        "    retriever=vector_db.as_retriever(),\n",
        "    combine_docs_chain=combine_docs_chain,\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "WSqLSrHQ3dVH"
      },
      "source": [
        "### Generate a retrieval-augmented response to a question"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%% md\n"
        },
        "id": "xuC5Q2WL3dVH"
      },
      "source": [
        "Use the RAG chain to process a question. The document chunks relevant to that question are retrieved and used as context."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false
        },
        "pycharm": {
          "name": "#%%\n"
        },
        "id": "rVCnSap63dVH"
      },
      "outputs": [],
      "source": [
        "from ibm_granite_community.notebook_utils import wrap_text\n",
        "\n",
        "output = rag_chain.invoke({\"input\": query})\n",
        "\n",
        "print(wrap_text(output['answer']))"
      ]
    }
  ],
  "metadata": {
    "language_info": {
      "name": "python"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
