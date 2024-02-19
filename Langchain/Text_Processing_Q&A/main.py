import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = PodSpec(environment="gcp-starter")
index_name = "medium-blogs-embeddings-index"
index = pc.Index(index_name)


if __name__ == "__main__":
    print("Hello VectorStore")
    loader = TextLoader(
        r"C:\Users\Abdul Mateen\Desktop\Udmey\2.17-1001\Medium_Analyzer\mediumblogs\mediumblog1.txt",
        encoding="utf-8",
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    docsearch = PineconeStore.from_documents(texts, embeddings, index_name=index_name)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa.invoke({"query": query})
    print(result)
