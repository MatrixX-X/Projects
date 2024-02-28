# import sys
# print(sys.path)
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone
from consts import index_name
from typing import Any, Dict, List

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()

    docsearch = PineconeStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="Give me parameter for transform?"))
