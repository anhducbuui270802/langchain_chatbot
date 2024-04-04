"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor
from dotenv import load_dotenv

import weaviate
from langchain_community.vectorstores import Chroma
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env.

def load_data():
    loader = DirectoryLoader('./data', glob="**/*.md", show_progress=True)
    return loader.load()


def get_embeddings_model() -> Embeddings:
    return OllamaEmbeddings(model='nomic-embed-text')

def ingest_docs():
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'default_database_host')
    DATABASE_PORT = os.getenv('DATABASE_PORT', 'default_database_port')
    DATABASE_USERNAME = os.getenv('DATABASE_USERNAME', 'default_database_user')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'default_database_password')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'default_database_name')
    RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", 'default_collection_name')


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    )

    record_manager = SQLRecordManager(
    f"weaviate/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    docs_from_data = load_data()
    logger.info(f"Loaded {len(docs_from_data)} docs from documentation")


    docs_transformed = text_splitter.split_documents(
        docs_from_data
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

if __name__ == "__main__":
    ingest_docs()