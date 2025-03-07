from global_var import PATH_PDFS, VS_DIR_VS, VS_CHUNK_SIZE, VS_CHUNK_OVERLAP

import os
import pandas as pd
from typing import Tuple, List

# LangChain Stuff
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter



##############################################################################
# HELPER FUNCTIONS (PDF-PARSING + VECTOR-STORE SETUPS)
##############################################################################
def get_pdf_text(doc):

    path_doc = f"{PATH_PDFS}/{doc}.pdf"
    pdf_reader = PyMuPDFLoader(path_doc)
    pdf_text = pdf_reader.load()

    return pdf_text


def build_vectorstore_retriever(docs: List[str], is_all: bool = False, embeddings=OpenAIEmbeddings()):
    if is_all:
        db_path = VS_DIR_VS + "/shared"
    else:
        db_path = VS_DIR_VS + "/" + docs[0]

    # Create Vector Store if not already existing
    if not os.path.exists(db_path):

        # Create folder for vector store
        os.mkdir(db_path)

        # Create vector store itself --> chrom.sqlite3 database
        if not os.path.exists(f"{db_path}/chroma.sqlite3"):
            vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
            vectordb.persist()

            # Add Documents to Vector store
            for doc in docs:
                pdf_text = get_pdf_text(doc)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=VS_CHUNK_SIZE,
                    chunk_overlap=VS_CHUNK_OVERLAP,
                )
                splitted_texts = text_splitter.split_documents(pdf_text)

                # Add to vector store
                vectordb.add_documents(documents=splitted_texts)
                vectordb.persist()

    else:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    return vectordb.as_retriever(), vectordb
