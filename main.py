import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
import PyPDF2
import faiss
import numpy as np
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import typer
import uvicorn
import json
import sys
# Add the sentence-transformers import
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    # API keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Vector database settings
    vector_dimension: int = 1536  # Default embedding dimension
    
    # Groq LLM settings
    llm_model: str = "deepseek-r1-distil-llama-70b"  # Alternatives: "qwen-qwq-32b", "deepseek-r1-distill-qwen-32b"
    temperature: float = 0.6
    top_p: float = 0.95
    max_completion_tokens: int = 1024
    stream: bool = True
    reasoning_format: str = "raw"
    
    # External resources
    zimlii_url: str = "https://zimlii.org/"
    
    # File paths
    constitution_pdf_path: str = "data/constitution.pdf"
    
    def validate(self):
        """Validate configuration settings."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        if not os.path.exists(self.constitution_pdf_path):
            raise FileNotFoundError(f"Constitution PDF not found at {self.constitution_pdf_path}")

class EmbeddingService:
    """Service for creating and managing embeddings using sentence-transformers."""
    
    def __init__(self, config: Config):
        self.config = config
        # Import sentence-transformers and load the model
        from sentence_transformers import SentenceTransformer
        # You can choose different models based on your needs
        # Common options include:
        # - 'all-MiniLM-L6-v2' (fast, 384 dimensions)
        # - 'all-mpnet-base-v2' (more accurate, 768 dimensions)
        # - 'all-distilroberta-v1' (balanced, 768 dimensions)
        self.model = SentenceTransformer('all-mpnet-base-v2')
        # Update vector dimensions in config to match the model's output
        self.config.vector_dimension = self.model.get_sentence_embedding_dimension()
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using sentence-transformers."""
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)  # Ensure float32 type for FAISS
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
class DocumentProcessor:
    """Process documents and prepare them for vector storage."""
    
    def __init__(self, config: Config, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF document and split into sections."""
        try:
            sections = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                current_section = {"title": "", "content": "", "page": 0}
                current_content = []
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    lines = text.split('\n')
                    
                    for line in lines:
                        # Heuristic for section headers in constitution documents
                        if line.strip().isupper() or (line.strip() and line.strip()[0].isdigit() and '.' in line[:5]):
                            # Save previous section if it exists
                            if current_section["content"]:
                                current_section["content"] = '\n'.join(current_content)
                                sections.append(current_section.copy())
                                current_content = []
                            
                            # Start new section
                            current_section = {
                                "title": line.strip(),
                                "content": "",
                                "page": page_num + 1
                            }
                        else:
                            current_content.append(line.strip())
                
                # Add the last section
                if current_content:
                    current_section["content"] = '\n'.join(current_content)
                    sections.append(current_section)
            
            return sections
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def create_chunks(self, sections: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create overlapping chunks from sections to improve retrieval."""
        chunks = []
        
        for section in sections:
            content = section["content"]
            
            if len(content) <= chunk_size:
                chunk = section.copy()
                chunk["embedding"] = self.embedding_service.get_embedding(chunk["title"] + " " + chunk["content"])
                chunks.append(chunk)
            else:
                # Split into overlapping chunks
                for i in range(0, len(content), chunk_size - overlap):
                    chunk_content = content[i:i + chunk_size]
                    chunk = {
                        "title": section["title"],
                        "content": chunk_content,
                        "page": section["page"],
                        "chunk_index": i // (chunk_size - overlap)
                    }
                    chunk["embedding"] = self.embedding_service.get_embedding(chunk["title"] + " " + chunk["content"])
                    chunks.append(chunk)
        
        return chunks

class VectorDatabase:
    """FAISS vector database for storing and retrieving document embeddings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from document embeddings."""
        try:
            self.documents = documents
            embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.config.vector_dimension)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding."""
        try:
            if self.index is None:
                raise ValueError("FAISS index not built")
            
            # Reshape query embedding for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search index
            distances, indices = self.index.search(query_embedding, k)
            
            # Return matched documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["distance"] = float(distances[0][i])
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            raise

class ZimliiLegalResearcher:
    """Specialized component for searching Zimlii legal website."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def search_legal_precedents(self, query: str, result_limit: int = 5) -> List[Dict[str, Any]]:
        """Search Zimlii for legal precedents based on query."""
        try:
            # Construct search URL
            search_url = f"{self.config.zimlii_url.rstrip('/')}/search.php"
            
            # Perform search request
            response = requests.get(
                search_url,
                params={"query": query},
                headers={"User-Agent": "LegalResearchBot/1.0"}
            )
            response.raise_for_status()
            
            # Parse response with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results (adjust selectors based on actual website structure)
            results = []
            result_elements = soup.select('.search-result') or soup.select('.case-listing') or soup.select('article')
            
            for i, element in enumerate(result_elements[:result_limit]):
                # Extract case title
                title_element = element.select_one('h3') or element.select_one('h2') or element.select_one('a')
                title = title_element.text.strip() if title_element else "Unknown Case"
                
                # Extract URL
                url_element = element.select_one('a')
                url = url_element['href'] if url_element and 'href' in url_element.attrs else None
                if url and not url.startswith('http'):
                    url = f"{self.config.zimlii_url.rstrip('/')}/{url.lstrip('/')}"
                
                # Extract snippet/summary
                summary_element = element.select_one('.summary') or element.select_one('p')
                summary = summary_element.text.strip() if summary_element else ""
                
                # Extract date if available
                date_element = element.select_one('.date') or element.select_one('time')
                date = date_element.text.strip() if date_element else "Unknown Date"
                
                results.append({
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "date": date
                })
            
            return results
        except requests.RequestException as e:
            logger.error(f"Error searching Zimlii: {str(e)}")
            return []

    def fetch_case_details(self, url: str) -> Dict[str, Any]:
        """Fetch detailed information about a specific case."""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "LegalResearchBot/1.0"}
            )
            response.raise_for_status()
            
            # Parse response with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract case details (adjust selectors based on actual website structure)
            title_element = soup.select_one('h1') or soup.select_one('h2')
            title = title_element.text.strip() if title_element else "Unknown Case"
            
            date_element = soup.select_one('.date') or soup.select_one('time')
            date = date_element.text.strip() if date_element else "Unknown Date"
            
            court_element = soup.select_one('.court') or soup.select_one('.metadata .court')
            court = court_element.text.strip() if court_element else "Unknown Court"
            
            judges_element = soup.select_one('.judges') or soup.select_one('.metadata .judges')
            judges = judges_element.text.strip() if judges_element else "Unknown"
            
            # Extract main content
            content_element = soup.select_one('.case-content') or soup.select_one('article') or soup.select_one('.content')
            content = content_element.text.strip() if content_element else ""
            
            return {
                "title": title,
                "date": date,
                "court": court,
                "judges": judges,
                "content": content,
                "url": url
            }
        except requests.RequestException as e:
            logger.error(f"Error fetching case details: {str(e)}")
            return {
                "title": "Error fetching case",
                "content": f"Could not retrieve case details: {str(e)}",
                "url": url
            }

class CrewAIManager:
    """Manage specialized agents using Crew AI."""
    
    def __init__(self, config: Config, zimlii_researcher: ZimliiLegalResearcher):
        self.config = config
        self.zimlii_researcher = zimlii_researcher
    
    def create_legal_research_crew(self):
        """Create and configure the legal research crew."""
        # Define specialized agents
        constitutional_expert = Agent(
            role="Constitutional Law Expert",
            goal="Analyze constitutional provisions related to women's rights",
            backstory="I am an expert in constitutional law with a focus on women's rights provisions.",
            verbose=True,
            allow_delegation=True
        )
        
        case_researcher = Agent(
            role="Legal Case Researcher",
            goal="Find relevant precedents and judgments related to women's rights cases",
            backstory="I specialize in researching legal precedents and case law to support legal arguments.",
            verbose=True,
            allow_delegation=True
        )
        
        legal_analyst = Agent(
            role="Legal Analyst",
            goal="Synthesize constitutional provisions and case law into comprehensive legal analysis",
            backstory="I combine constitutional principles with case precedents to provide thorough legal analysis.",
            verbose=True,
            allow_delegation=False
        )
        
        # Define tasks
        constitutional_analysis_task = Task(
            description="Analyze the constitutional provisions relevant to the given women's rights issue",
            agent=constitutional_expert,
            expected_output="Detailed analysis of relevant constitutional clauses"
        )
        
        case_research_task = Task(
            description="Research precedents, judgments, and similar cases related to the women's rights issue",
            agent=case_researcher,
            expected_output="List of relevant legal cases with summaries and citations"
        )
        
        legal_synthesis_task = Task(
            description="Synthesize constitutional analysis and case research into a comprehensive legal opinion",
            agent=legal_analyst,
            expected_output="Comprehensive legal opinion with constitutional basis and case support"
        )
        
        # Create crew
        legal_crew = Crew(
            agents=[constitutional_expert, case_researcher, legal_analyst],
            tasks=[constitutional_analysis_task, case_research_task, legal_synthesis_task],
            process=Process.sequential,
            verbose=True
        )
        
        return legal_crew
    
    def execute_zimlii_search(self, query: str) -> Dict[str, Any]:
        """Execute a search on Zimlii and compile results."""
        # Search for relevant cases
        search_results = self.zimlii_researcher.search_legal_precedents(query)
        
        # Fetch details for top results
        detailed_results = []
        for result in search_results[:3]:  # Limit to top 3 results for efficiency
            if result.get("url"):
                case_details = self.zimlii_researcher.fetch_case_details(result["url"])
                detailed_results.append(case_details)
        
        return {
            "search_query": query,
            "search_results": search_results,
            "detailed_results": detailed_results
        }

class GroqReasoner:
    """Integration with Groq API for LLM reasoning."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)
    
    def generate_response(self, 
                         question: str, 
                         constitutional_clauses: List[Dict[str, Any]], 
                         legal_precedents: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a reasoned response using Groq LLM."""
        try:
            # Format constitutional clauses
            const_content = ""
            for i, clause in enumerate(constitutional_clauses):
                const_content += f"[Constitution {i+1}] {clause['title']}\n{clause['content']}\n\n"
            
            # Format legal precedents
            prec_content = ""
            for i, case in enumerate(legal_precedents.get("detailed_results", [])):
                prec_content += f"[Case {i+1}] {case['title']} ({case['date']})\n"
                prec_content += f"Court: {case.get('court', 'Unknown')}\n"
                prec_content += f"Summary: {case.get('content', '')[:500]}...\n"
                prec_content += f"URL: {case.get('url', '')}\n\n"
            
            # Create prompt
            prompt = f"""
You are a legal expert specializing in women's rights law. You have been provided with relevant constitutional clauses and legal precedents to answer the following question:

QUESTION:
{question}

CONSTITUTIONAL PROVISIONS:
{const_content}

LEGAL PRECEDENTS:
{prec_content}

Your task is to provide a comprehensive legal analysis that:
1. States the constitutional basis for addressing the question
2. Supports this with relevant legal precedents, judgments, and similar cases
3. Includes explicit references (with citations) for every constitutional clause, case, or law mentioned
4. Uses step-by-step reasoning to reach your conclusion

Start with an analysis of the constitutional foundation, then discuss how legal precedents apply, and finally provide a conclusion.
"""
            
            # Generate response with Groq
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in women's rights law."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_completion_tokens,
                stream=False,  # Set to False for production code
            )
            
            return {
                "question": question,
                "reasoning": response.choices[0].message.content,
                "constitutional_references": [c["title"] for c in constitutional_clauses],
                "precedent_references": [c["title"] for c in legal_precedents.get("detailed_results", [])]
            }
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            raise

class WomensRightsLegalAssistant:
    """Main application class for the Women's Rights Legal Assistant."""
    
    def __init__(self, config_path: str = None):
        """Initialize the application."""
        # Load configuration
        self.config = Config()
        self.config.validate()
        
        # Initialize services
        self.embedding_service = EmbeddingService(self.config)
        self.document_processor = DocumentProcessor(self.config, self.embedding_service)
        self.vector_db = VectorDatabase(self.config)
        self.zimlii_researcher = ZimliiLegalResearcher(self.config)
        self.crew_manager = CrewAIManager(self.config, self.zimlii_researcher)
        self.groq_reasoner = GroqReasoner(self.config)
        
        # Flag to track if index is built
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the system by loading and indexing the constitution."""
        try:
            logger.info("Initializing Women's Rights Legal Assistant...")
            
            # Load constitution PDF
            logger.info(f"Loading constitution from {self.config.constitution_pdf_path}")
            sections = self.document_processor.extract_text_from_pdf(self.config.constitution_pdf_path)
            logger.info(f"Extracted {len(sections)} sections from constitution")
            
            # Create chunks and embeddings
            chunks = self.document_processor.create_chunks(sections)
            logger.info(f"Created {len(chunks)} chunks from sections")
            
            # Build FAISS index
            self.vector_db.build_index(chunks)
            logger.info("FAISS index built successfully")
            
            self.is_initialized = True
            logger.info("Initialization complete")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query about women's rights."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Get query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            
            # Search for relevant constitutional clauses
            constitutional_results = self.vector_db.search(query_embedding, k=3)
            logger.info(f"Found {len(constitutional_results)} relevant constitutional clauses")
            
            # Search for legal precedents on Zimlii
            legal_results = self.crew_manager.execute_zimlii_search(query)
            logger.info(f"Found {len(legal_results.get('search_results', []))} relevant legal precedents")
            
            # Generate response with Groq
            response = self.groq_reasoner.generate_response(
                question=query,
                constitutional_clauses=constitutional_results,
                legal_precedents=legal_results
            )
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": query,
                "error": str(e),
                "reasoning": "An error occurred while processing your query."
            }

# Create FastAPI app
app = FastAPI(
    title="Women's Rights Legal Assistant API",
    description="Agentic RAG system for women's rights legal research",
    version="1.0.0"
)

# Create Typer CLI app
cli = typer.Typer()

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    reasoning: str
    constitutional_references: List[str]
    precedent_references: List[str]

# Global instance of the assistant
legal_assistant = None

def get_legal_assistant():
    """Get or create the legal assistant instance."""
    global legal_assistant
    if legal_assistant is None:
        legal_assistant = WomensRightsLegalAssistant()
        legal_assistant.initialize()
    return legal_assistant

# API routes
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a legal query about women's rights."""
    try:
        assistant = get_legal_assistant()
        response = assistant.process_query(request.query)
        return response
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# CLI commands
@cli.command()
def query(question: str):
    """Process a legal query about women's rights."""
    assistant = get_legal_assistant()
    response = assistant.process_query(question)
    typer.echo(json.dumps(response, indent=2))

@cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the API server."""
    try:
        uvicorn.run("main:app", host=host, port=port, reload=False)
    except Exception as e:
        typer.echo(f"Error starting server: {str(e)}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host=host, port=port)
    else:
        cli()