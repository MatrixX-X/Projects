import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from consts import index_name

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def ingest_docs():
    # Now, you can use this full path in your loader
    loader = ReadTheDocsLoader(
        path=r"docs/api.python.langchain.com/en/latest/chains", encoding="utf-8"
    )
    docs = loader.load()
    print(f"loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("docs", "https:/")
        new_url = new_url.replace("docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeStore.from_documents(documents, embeddings, index_name=index_name)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
