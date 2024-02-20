import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

if __name__ == "__main__":
    loader = PyPDFLoader(
        r"C:\Users\Abdul Mateen\Desktop\Udmey\2.17-1001\Chat_with_PDF\2210.03629.pdf"  # add your own pdf path
    )
    pages = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()  # openai_api_key=os.getenv("OPENAI_API_KEY")

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever()
    )
    query = "Give me gist of react in three sentences"
    res = qa.invoke({"query": query})
    print(res["result"])
