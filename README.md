# Build-a-RAG-Pattern-with-LangChain
Retrieval Augmented Generation (RAG) with Langchain
Using IBM Granite Models

In this notebook
This notebook contains instructions for performing Retrieval Augmented Generation (RAG). RAG is an architectural pattern that can be used to augment the performance of language models by recalling factual information from a knowledge base, and adding that information to the model query. The most common approach in RAG is to create dense vector representations of the knowledge base in order to retrieve text chunks that are semantically similar to a given user query.

RAG use cases include:

Customer service: Answering questions about a product or service using facts from the product documentation.

Domain knowledge: Exploring a specialized domain (e.g., finance) using facts from papers or articles in the knowledge base.

News chat: Chatting about current events by calling up relevant recent news articles.

In its simplest form, RAG requires 3 steps:

Initial setup: Index knowledge-base passages for efficient retrieval. In this recipe, we take embeddings of the passages, and store them in a vector database.

Upon each user query: Retrieve relevant passages from the database using semantically similar embeddings.

Response Generation: Generate a response by feeding retrieved passages into a large language model (LLM), along with the user query.

Setting up the environment
Install dependencies.

Python

! echo "::group::Install Dependencies"
%pip install uv
! uv pip install git+https://github.com/ibm-granite-community/utils.git \
    transformers \
    langchain_classic \
    langchain_community \
    langchain_text_splitters \
    langchain_huggingface sentence_transformers \
    langchain_milvus 'pymilvus[milvus_lite]' \
    'langchain_replicate @ git+https://github.com/ibm-granite-community/langchain-replicate.git' \
    wget
! echo "::endgroup::"
Selecting System Components
Choose your Embeddings Model
Specify the model to use for generating embedding vectors from text.

Python

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
Choose your Vector Database
Specify the database to use for storing and retrieving embedding vectors. We use Milvus Lite here.

Python

from langchain_milvus import Milvus
import tempfile

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)
Choose your LLM
The LLM will be used for answering the question, given the retrieved text. We use the Granite models from Replicate.

Python

from langchain_replicate import ChatReplicate
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-4.0-h-small"
model = ChatReplicate(
    model=model_path,
    replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),
)
Building the Vector Database
In this example, we take the State of the Union speech text, split it into chunks, and load it into the vector database.

Download the document
Python

import os
import wget

filename = 'state_of_the_union.txt'
url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

if not os.path.isfile(filename):
    wget.download(url, out=filename)
Split the document into chunks
Python

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=embeddings_tokenizer.max_len_single_sentence,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)
doc_id = 0
for text in texts:
    text.metadata["doc_id"] = (doc_id:=doc_id+1)
print(f"{len(texts)} text document chunks created")
Populate the vector database
Python

ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")
Querying the Vector Database
Conduct a similarity search
Python

query = "What did the president say about Ketanji Brown Jackson?"
docs = vector_db.similarity_search(query)
print(f"{len(docs)} documents returned")
for doc in docs:
    print(doc)
    print("=" * 80)
Answering Questions
Automate the RAG pipeline
Build a RAG chain with the model and the document retriever.

Python

from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Create a Granite prompt for question-answering
prompt_template = ChatPromptTemplate.from_template("{input}")

# Assemble the chain
combine_docs_chain = create_stuff_documents_chain(
    llm = model,
    prompt = prompt_template,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)
Generate a response
Python

from ibm_granite_community.notebook_utils import wrap_text

output = rag_chain.invoke({"input": query})
print(wrap_text(output['answer']))
