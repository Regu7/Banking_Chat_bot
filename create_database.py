# Import necessary libraries
import os
import shutil

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define constants
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\FAQ_RAG.csv")


# Main function
def main_func():
    create_data_store()


# Function to generate data store
def create_data_store():
    doc_list = load_doc_list()
    text_chunks = split_doc_text(doc_list)
    store_to_chroma(text_chunks)


# Function to load documents
def load_doc_list():
    # doc_loader = DirectoryLoader(DATA_DIR, glob="*.md")
    # doc_list = doc_loader.load()
    # return doc_list
    loader = CSVLoader(
        file_path=DATA_DIR, encoding="utf-8", csv_args={"delimiter": ","}
    )
    doc_list = loader.load()
    return doc_list


# Function to split text
def split_doc_text(doc_list: list[Document]):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True,
    )
    text_chunks = text_split.split_documents(doc_list)
    print(f"Spliting {len(doc_list)} documents into {len(text_chunks)} chunks.")

    sample_doc = text_chunks[10]
    print(sample_doc.page_content)
    print(sample_doc.metadata)

    return text_chunks


# Function to save to chroma
def store_to_chroma(text_chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        text_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DIR
    )
    db.persist()
    print(f"Saved {len(text_chunks)} chunks to {CHROMA_DIR}.")


# Main execution
if __name__ == "__main__":
    main_func()
