from dotenv import load_dotenv
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from ai_models.chats import ChatModel
from ai_models.embeddings import EmbeddingModel
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from models.state import State

WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)

load_dotenv()

OPEN_API_KEY = os.environ.get("OPEN_AI_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2")

llm = ChatOpenAI(model=ChatModel.GPT_MINI, api_key=OPEN_API_KEY)
embeddings = OpenAIEmbeddings(model=EmbeddingModel.TEXT_EMBEDDING_SMALL,
                              api_key=OPEN_API_KEY)

vector_store = InMemoryVectorStore(embeddings)

# load page contents from web
loader = WebBaseLoader(web_paths=WEB_PATHS)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], 
                              "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
