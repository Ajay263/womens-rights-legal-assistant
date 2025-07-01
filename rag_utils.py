import os
import logging
import json
import time
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from db import generate_conversation_id, save_conversation, save_feedback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables and set up API keys"""
    logger.info("Loading environment variables")
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("No GROQ API key found in .env, using hardcoded key")
        groq_api_key = "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"
    
    return {
        "groq_api_key": groq_api_key,
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "collection_name": os.getenv("QDRANT_COLLECTION", "zimbabwe_legal_docs")
    }

def get_vectorstore_connection(qdrant_url, qdrant_api_key, collection_name):
    """Connect to the existing Qdrant vector store"""
    logger.info("Connecting to Qdrant vector store")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key if qdrant_api_key else None)
        
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            logger.warning(f"Collection {collection_name} does not exist")
            return None
        
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        logger.info("Connected to Qdrant vector store successfully")
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error connecting to vector store: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Create a conversation chain with the vector store"""
    logger.info("Creating conversation chain")
    
    env_vars = load_env_variables()
    api_key = env_vars["groq_api_key"]
    
    system_prompt = """
        You are a highly intelligent and articulate legal assistant with specialized expertise in women's rights under Zimbabwean civil law.

        You have full knowledge of:
        - The Constitution of Zimbabwe (especially  related to women's rights, equality, and non-discrimination)
        - Zimbabwean statutes and civil codes affecting marriage, inheritance, custody, domestic violence, and property rights
        - Landmark and recent case law related to women’s legal disputes (e.g., Magaya v Magaya, cases involving maintenance, custody, and gender-based violence)
        - Customary vs civil law conflicts and how courts have resolved them
        - Social, cultural, and economic factors affecting legal access and interpretation in Zimbabwe

        When assisting the user:
        1. Begin by identifying the legal domain(s) involved (e.g., inheritance law, marriage law, domestic violence, etc.).
        2. Retrieve and reference applicable constitutional provisions and statutes with section numbers.
        3. Cite at least one relevant case law precedent, explaining how it applies.
        4. Clearly explain legal jargon and procedures in accessible language for non-lawyers.
        5. Discuss multiple legal outcomes or strategic options if appropriate.
        6. Be context-sensitive (e.g., cultural barriers, gender dynamics).
        7. Clarify that the response is **informational only** and does **not constitute legal advice**.

        If the query falls outside your scope or lacks context, politely request more details or recommend seeking legal counsel.
        """

    
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=2048,
            top_p=0.95
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Context: {context}")
        ])
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            document_chain
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        conversation_chain = {
            "retrieval_chain": retrieval_chain,
            "memory": memory,
            "model_name": "llama-3.3-70b-versatile"
        }
        
        return conversation_chain
    
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        return None

def process_question(question, conversation_chain):
    """Process a question through the RAG system"""
    try:
        start_time = time.time()
        
        # Get response from the chain
        response = conversation_chain["retrieval_chain"].invoke({
            "input": question,
            "chat_history": conversation_chain["memory"].chat_memory.messages
        })
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Generate conversation ID and save to database
        conversation_id = generate_conversation_id()
        save_conversation(
            conversation_id=conversation_id,
            question=question,
            answer=response["answer"],
            model_used=conversation_chain["model_name"],
            response_time=response_time
        )
        
        return {
            "conversation_id": conversation_id,
            "answer": response["answer"],
            "response_time": response_time
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return None 