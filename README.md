# 🚀 Retrieval Augmented Generation (RAG) with LangChain
### *Powered by IBM Granite Models*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ozcanelmaci/Build-a-RAG-Pattern-with-LangChain/blob/main/Build_a_RAG_Pattern_with_LangChain.ipynb)

---

## 📌 Project Overview
This project demonstrates the implementation of a **Retrieval Augmented Generation (RAG)** architectural pattern. RAG enhances Large Language Models (LLMs) by retrieving factual information from a private knowledge base and providing it as context for the query.



### 💡 Key Features
- **Semantic Search:** Dense vector representations for high-accuracy retrieval.
- **Efficient Storage:** Integration with **Milvus Lite** for vector management.
- **Advanced LLMs:** Leveraging **IBM Granite** via Replicate.

---

## 🛠️ Environment Setup

Install the necessary dependencies to run the notebook:

```python
# Install core libraries and community utilities
%pip install uv
! uv pip install git+[https://github.com/ibm-granite-community/utils.git](https://github.com/ibm-granite-community/utils.git) \
    transformers \
    langchain_classic \
    langchain_community \
    langchain_text_splitters \
    langchain_huggingface sentence_transformers \
    langchain_milvus 'pymilvus[milvus_lite]' \
    'langchain_replicate @ git+[https://github.com/ibm-granite-community/langchain-replicate.git](https://github.com/ibm-granite-community/langchain-replicate.git)' \
    wget
```

---

## 🏗️ System Components

### 1. Embeddings Model
We use the **IBM Granite Embedding** model to convert text into high-dimensional vectors.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
```

### 2. Vector Database (Milvus)
Efficiently store and query embeddings using a local Milvus instance.

```python
from langchain_milvus import Milvus
import tempfile

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)
```

### 3. LLM Configuration
Connecting to **Granite 4.0** through the Replicate API.

```python
from langchain_replicate import ChatReplicate

model_path = "ibm-granite/granite-4.0-h-small"
model = ChatReplicate(
    model=model_path,
    replicate_api_token='YOUR_REPLICATE_API_TOKEN',
)
```

---

## 📖 Pipeline Implementation

### Data Ingestion
Loading the *State of the Union* address and splitting it into manageable chunks.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=embeddings_tokenizer.max_len_single_sentence,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)
vector_db.add_documents(texts)
```

### RAG Chain Execution
Automating the retrieval and generation process.

```python
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Define prompt and assemble chain
prompt_template = ChatPromptTemplate.from_template("{input}")
# ... (Chain assembly details)
```

---

## 🎯 Example Usage
**Query:** *"What did the president say about Ketanji Brown Jackson?"*

The system retrieves relevant segments from the speech and provides a grounded response using the Granite model.

---

### 👨‍💻 Author
**Özcan Elmacı** *Software Engineer & AI Developer*
