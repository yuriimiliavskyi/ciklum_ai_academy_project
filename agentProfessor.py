from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import ConfigDict

from langchain_community.document_loaders import PyPDFLoader

from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain.chat_models import init_chat_model

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community.gmail.send_message import GmailSendMessage
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv


class State(TypedDict):
        input: str
        context: List[Document]
        answer_basic: str
        answer_middle: str
        answer_final: str
        post_status: str

class TextRetriever(BaseRetriever):
        vector_store: object
        
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def __init__(self, vector_store: object):
            super().__init__(
                vector_store=vector_store
            )

        def _get_relevant_documents(self, query: str) -> List[Document]:
            """
            Translates a string query into a graph invocation and returns documents.
            """
            # print(f"\n[StateGraphRetriever]: Processing query: {query}")
            
            # 1. CONSTRUCT THE INPUT STATE
            initial_state = {
                "input":  query,
                "context": "", 
            }

            # 2. INVOKE THE GRAPH
            # We run the graph to completion
            final_state = self.vector_store.similarity_search(initial_state["input"])

            return final_state

def create_chatbot(MY_PDF_FILE: str, RECIPIENT_EMAIL: str):
    load_dotenv()
    MY_API_KEY = os.getenv("MY_API_KEY")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    loader = PyPDFLoader(MY_PDF_FILE) 
    document = loader.load()
    client = QdrantClient(":memory:")
    ex_emb = embeddings.embed_query("sample text")
    vector_size = len(ex_emb)

    if not client.collection_exists("test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=embeddings,
    )
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai"
                        , google_api_key=MY_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(document)
    _ = vector_store.add_documents(documents=all_splits)
    template1 = """You are an assistant in AI-related questions and your knowledge is mostly based on the document provided. Use the following pieces of retrieved context to answer the question.  
    The answers can be quite long, but not too long, just enough to uncover the topic you are asked about and to include the material from the document on this topic

    Question: {input} 

    Context: {context} 

    Answer:"""
    prompt1 = ChatPromptTemplate.from_template(template1)
    template2 = """
    Please review the "Initial_answer" and check whether you can improve it. E.g., check if it is grammatically correct, 
    if it fully answers the question, if you accounted for all the context available, if it understandable for most people, if it is not offensive etc. You can shorten it a bit and make more readable.
    You may (but not must) incorporate relevant information from the provided "context" and initial "question".

    Question: {input} 
    Context: {context} 
    Initial_answer: {answer_basic}

    Answer: <please provide your improved answer here without additional comments>"""
    prompt2 = ChatPromptTemplate.from_template(template2)
    template3 = """You are an assistant for generating lecture notes (compendium) on AI-related topics. Your notes should be based on the "Initial_answer"  
    which has to be reformatted into a more engaging and professional style suitable for lecture notes. Ensure the notes are concise, highlights key points, and is easy to understand.
    You may (but not must) incorporate relevant information from the provided "context" and initial "question" to enhance the notes.

    Question: {input} 
    Context: {context} 
    Initial_answer: {answer_middle}

    Answer:"""
    prompt3 = ChatPromptTemplate.from_template(template3)
    
        
    text_retriever = TextRetriever(
        vector_store=vector_store
    )

    os.environ["LANGSMITH_TRACING"] = "False"  ##true does not work

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = text_retriever.invoke(state["input"])
        return {"context": retrieved_docs}

    def generate_basic(state: State):
        messages = prompt1.invoke({"input": state["input"], 
                                "context": state["context"]}) 
        response = llm.invoke(messages)
        #print("Basic answer:", response.content) ### temporary
        return {"answer_basic": response.content}

    def generate_middle(state: State):
        messages = prompt2.invoke({"input": state["input"], 
                                "context": state["context"],
                                "answer_basic": state["answer_basic"]}) 
        response = llm.invoke(messages)
        #print("Middle answer:", response.content) ### temporary
        return {"answer_middle": response.content}

    def generate_final(state: State):
        messages = prompt3.invoke({"input": state["input"], 
                                "context": state["context"],
                                "answer_middle": state["answer_middle"]}) 
        response = llm.invoke(messages)
        #print("Final answer:", response.content) ### temporary
        return {"answer_final": response.content}

    
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    EMAIL_SUBJECT = "Automated Lecture Notes"
    if not os.path.exists("credentials.json"):
            print("--- WARNING: 'credentials.json' not found. ---")
    else:
            tool = GmailSendMessage(scopes=SCOPES)


    def post_message(state: State):
        """
        This node takes the final output from the state
        and posts it using the selected tool.
        """
        
        # Get the message from the state
        message_to_post = state.get("answer_final", "No content generated.")
        
        try:
            # Call your tool
            result = tool.invoke({
                "message": message_to_post,
                "to": [RECIPIENT_EMAIL],
                "subject": EMAIL_SUBJECT
            })
            
            print(f"Posting result: {result}")
            return {"post_status": result}

        except Exception as e:
            print(f"Error during sending email: {e}")
            return {"post_status": f"Error: {e}"}
        
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate_basic, generate_middle, generate_final, post_message])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph

def my_agent(graph, question):
    res = graph.invoke({"input": question})
    return res["answer_final"]