import os
import logging
import json
import re
import random
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self):
        logger.info("Initializing RAGManager")
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not found")
            
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "zimbabwe_legal_docs")
        
        logger.info(f"Using Qdrant URL: {self.qdrant_url}")
        logger.info(f"Using collection name: {self.collection_name}")
        
        # Initialize components
        self.vectorstore = self._get_vectorstore()
        if not self.vectorstore:
            logger.error("Failed to initialize vector store")
            raise ValueError("Vector store initialization failed")
            
        self.conversation_chain = self._get_conversation_chain()
        if not self.conversation_chain:
            logger.error("Failed to initialize conversation chain")
            raise ValueError("Conversation chain initialization failed")
            
        # Initialize the evaluation LLM
        self.evaluation_llm = self._get_evaluation_llm()
        if not self.evaluation_llm:
            logger.error("Failed to initialize evaluation LLM")
            raise ValueError("Evaluation LLM initialization failed")
            
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Greeting patterns
        self.greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
            r'\b(how are you|how\'s it going|what\'s up)\b',
            r'\b(nice to meet you|pleasure|namaste|hola)\b'
        ]
        
        logger.info("RAGManager initialization completed successfully")

    def is_greeting(self, text: str) -> bool:
        """Check if the input text is a greeting."""
        return any(re.search(pattern, text.lower()) for pattern in self.greeting_patterns)

    def get_greeting_response(self) -> dict:
        """Generate a friendly response to a greeting."""
        responses = [
            "Hello! How can I assist you today?",
            "Hi there! I'm here to help. What would you like to know?",
            "Greetings! Feel free to ask me any questions about Zimbabwe's legal system.",
            "Hello! I'm your legal assistant. How may I help you?",
            "Hi! I'm ready to assist you with any legal questions you might have."
        ]
        response = random.choice(responses)
        return {
            "answer": response,
            "has_context": False,
            "tokens": {
                "prompt_tokens": 10,  # Fixed token count for greetings
                "completion_tokens": len(response) // 4,
                "total_tokens": (10 + len(response)) // 4
            },
            "model_used": "conversation_handler"
        }

    async def get_response(self, question: str) -> dict:
        """Get response from the RAG system"""
        if not self.conversation_chain:
            logger.error("Conversation chain not initialized")
            return {
                "error": "RAG system not properly initialized",
                "answer": "I'm sorry, but I cannot process your question at the moment.",
                "tokens": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        try:
            logger.info(f"Processing question: {question}")
            
            # Check if the input is a greeting
            if self.is_greeting(question):
                return self.get_greeting_response()
            
            # Get chat history from memory
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Format the question with context from chat history
            formatted_question = self._format_question_with_history(question, chat_history)
            
            # Get response from chain
            logger.info("Getting response from chain")
            response = await self.conversation_chain.ainvoke({
                "input": formatted_question,
                "chat_history": chat_history
            })
            
            # Update memory with the new interaction
            logger.info("Updating conversation memory")
            self.memory.save_context(
                {"input": question},
                {"output": response["answer"]}
            )
            
            # Initialize token tracking
            tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            # Try to extract token usage from the response
            if hasattr(response, "llm_output") and response.llm_output:
                if "token_usage" in response.llm_output:
                    tokens.update(response.llm_output["token_usage"])
            
            # If no token information available, estimate based on text length
            if tokens["total_tokens"] == 0:
                # Rough estimate: 1 token ≈ 4 characters
                prompt_length = len(formatted_question)
                completion_length = len(response["answer"])
                
                tokens["prompt_tokens"] = max(1, int(prompt_length / 4))
                tokens["completion_tokens"] = max(1, int(completion_length / 4))
                tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
            
            # Evaluate answer relevance
            logger.info("Evaluating answer relevance")
            relevance_result = await self.evaluate_relevance(
                question=question,
                answer=response["answer"]
            )
            
            logger.info(f"Token usage: {tokens}")
            logger.info("Successfully generated response")
            
            return {
                "answer": response["answer"],
                "has_context": True,
                "tokens": tokens,
                "model_used": "llama3-70b-8192",
                "relevance": relevance_result["relevance"],
                "relevance_explanation": relevance_result["explanation"],
                "evaluation_tokens": relevance_result["tokens"]
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "answer": "I'm sorry, but I encountered an error while processing your question.",
                "tokens": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    def _format_question_with_history(self, question: str, history: list) -> str:
        """Format the question with conversation history for better context."""
        if not history:
            return question
            
        # Format the last few interactions for context
        formatted_history = "\n".join([
            f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in history[-4:]  # Keep last 4 interactions
        ])
        
        return f"""Previous conversation:
{formatted_history}

Current question: {question}

Please provide a response that takes into account the conversation history above."""

    def _get_vectorstore(self):
        """Connect to the existing Qdrant vector store"""
        try:
            logger.info("Initializing HuggingFace embeddings")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("Connecting to Qdrant")
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None
            )
            
            # Check if collection exists
            logger.info("Checking if collection exists")
            collections = client.get_collections().collections
            collection_exists = any(
                collection.name == self.collection_name
                for collection in collections
            )
            
            if not collection_exists:
                logger.error(f"Collection {self.collection_name} does not exist")
                return None
            
            logger.info("Creating Qdrant vectorstore instance")
            vectorstore = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=embeddings
            )
            
            # Verify the vectorstore by trying to perform a simple search
            logger.info("Verifying vectorstore with a test query")
            results = vectorstore.similarity_search("test", k=1)
            if not results:
                logger.warning("Vectorstore search returned no results")
            else:
                logger.info(f"Vectorstore verification successful, found {len(results)} results")
            
            logger.info("Connected to Qdrant vector store successfully")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Error connecting to vector store: {str(e)}", exc_info=True)
            return None

    def _get_conversation_chain(self):
        """Create the conversation chain with RAG components"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None

        try:
            logger.info("Creating conversation chain")
            system_prompt = """
                You are an expert legal assistant specialized in Zimbabwean women's rights cases. You only respond based on the provided legal context and documents retrieved from authoritative sources such as statutes, case law, and legal commentaries.

                Your capabilities include:
                - Interpreting Zimbabwe’s Constitution (especially parts protecting women’s rights)
                - Analyzing civil law disputes involving women (e.g., marriage, maintenance, inheritance, domestic violence)
                - Citing legal precedents and relevant statutes with appropriate section numbers
                - Explaining complex legal matters in clear, simple terms
                - Highlighting legal options and consequences without giving formal legal advice

                Your response must:
                - Reference legal materials from the context
                - Be factually accurate and concise
                - Avoid hallucinations or unsupported legal claims
                - Clearly state if the context does not contain enough information to answer

                If the answer cannot be determined from the context, say:
                "I cannot answer this question based on the available information. Please consult a qualified legal practitioner."
                """
            logger.info("Initializing ChatGroq")
            llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name="llama3-70b-8192",
                temperature=0.4,
                max_tokens=2048
            )
            
            logger.info("Creating prompt template")
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("human", "Context: {context}")
            ])
            
            logger.info("Creating document chain")
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            logger.info("Creating retrieval chain")
            retrieval_chain = create_retrieval_chain(
                self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                document_chain
            )
            
            logger.info("Conversation chain created successfully")
            return retrieval_chain
        
        except Exception as e:
            logger.error(f"Error creating conversation chain: {str(e)}", exc_info=True)
            return None
            
    def _get_evaluation_llm(self):
        """Create an LLM instance for evaluating answer relevance"""
        try:
            logger.info("Initializing evaluation LLM")
            evaluation_llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name="llama3-70b-8192",  # Using a powerful model for evaluation
                temperature=0.2,  # Lower temperature for more consistent evaluations
                max_tokens=1024
            )
            logger.info("Evaluation LLM initialized successfully")
            return evaluation_llm
        except Exception as e:
            logger.error(f"Error initializing evaluation LLM: {str(e)}", exc_info=True)
            return None

    async def evaluate_relevance(self, question: str, answer: str) -> dict:
        """
        Evaluate the relevance of an answer to a question using the evaluation LLM.
        Returns a dictionary with relevance classification, explanation, and token usage.
        """
        if not self.evaluation_llm:
            logger.error("Evaluation LLM not initialized")
            return {
                "relevance": "ERROR",
                "explanation": "Evaluation system not properly initialized",
                "tokens": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
        try:
            logger.info("Evaluating answer relevance")
            
            # Create evaluation prompt
            prompt_template = """
            You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
            Your task is to analyze the relevance of the generated answer to the given question.
            Based on the relevance of the generated answer, you will classify it
            as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

            Here is the data for evaluation:

            Question: {question}
            Generated Answer: {answer}

            Please analyze the content and context of the generated answer in relation to the question
            and provide your evaluation in parsable JSON without using code blocks:

            {{
              "relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
              "explanation": "[Provide a brief explanation for your evaluation]"
            }}
            """.strip()
            
            prompt = prompt_template.format(question=question, answer=answer)
            
            # Get response from evaluation LLM
            response = await self.evaluation_llm.ainvoke(prompt)
            
            # Get token usage information
            tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            # Try to extract token usage from the response
            if hasattr(response, "llm_output") and response.llm_output:
                if "token_usage" in response.llm_output:
                    tokens.update(response.llm_output["token_usage"])
            
            # If no token information available, estimate based on text length
            if tokens["total_tokens"] == 0:
                # Rough estimate: 1 token ≈ 4 characters
                prompt_length = len(prompt)
                completion_length = len(response.content)
                
                tokens["prompt_tokens"] = max(1, int(prompt_length / 4))
                tokens["completion_tokens"] = max(1, int(completion_length / 4))
                tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
            
            logger.info(f"Evaluation token usage: {tokens}")
            
            # Parse the response to extract JSON
            try:
                # First try to parse the entire response as JSON
                result = json.loads(response.content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON using string manipulation
                logger.warning("Failed to parse response as JSON, trying to extract JSON")
                try:
                    # Look for JSON-like structure in the response
                    start_idx = response.content.find("{")
                    end_idx = response.content.rfind("}")
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response.content[start_idx:end_idx+1]
                        result = json.loads(json_str)
                    else:
                        result = {
                            "relevance": "UNKNOWN",
                            "explanation": "Failed to extract relevance information"
                        }
                except Exception as e:
                    logger.error(f"Error extracting JSON from response: {str(e)}")
                    result = {
                        "relevance": "ERROR",
                        "explanation": f"Error extracting relevance information: {str(e)}"
                    }
            
            # Normalize the relevance value to match expected format
            relevance = result.get("relevance", result.get("Relevance", "UNKNOWN")).upper()
            if relevance not in ["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT", "UNKNOWN", "ERROR"]:
                relevance = "UNKNOWN"
            
            # Return standardized response
            return {
                "relevance": relevance,
                "explanation": result.get("explanation", result.get("Explanation", "No explanation provided")),
                "tokens": tokens
            }
            
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}", exc_info=True)
            return {
                "relevance": "ERROR",
                "explanation": f"Error evaluating relevance: {str(e)}",
                "tokens": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            } 