import os
import logging
import sys
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import io
from pathlib import Path
import time
from functools import wraps
from tqdm import tqdm

# Import Qdrant components
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables")
load_dotenv()

# Check for Qdrant settings
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")  # Optional, if authentication is enabled
collection_name = os.getenv("QDRANT_COLLECTION", "zimbabwe_legal_docs")

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file"""
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return None

def load_documents_from_directory(directory_path="legal_docs"):
    """Loads all PDFs and DOCX files from a specified directory"""
    logger.info(f"Loading documents from directory: {directory_path}")
    Path(directory_path).mkdir(exist_ok=True)
    documents = []
    
    # Get all PDF and DOCX files in the directory
    all_files = []
    for ext in ["*.pdf", "*.docx"]:
        all_files.extend(list(Path(directory_path).glob(ext)))
    
    if not all_files:
        logger.warning(f"No PDF or DOCX files found in {directory_path}")
        return documents
    
    logger.info(f"Found {len(all_files)} document(s)")
    
    # Process each file
    for file_path in tqdm(all_files, desc="Loading documents"):
        try:
            file_name = file_path.name
            logger.info(f"Processing file: {file_name}")
            
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, "rb") as file:
                    documents.append({
                        "name": file_name,
                        "type": "pdf",
                        "content": io.BytesIO(file.read())
                    })
                    logger.info(f"Successfully loaded PDF: {file_name}")
            
            elif file_path.suffix.lower() == '.docx':
                text = extract_text_from_docx(file_path)
                if text:
                    documents.append({
                        "name": file_name,
                        "type": "docx",
                        "content": text
                    })
                    logger.info(f"Successfully loaded DOCX: {file_name}")
                else:
                    logger.error(f"Failed to extract text from DOCX: {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} document(s)")
    return documents

def get_document_text(documents):
    """Extract text from documents"""
    logger.info("Extracting text from documents")
    text_by_source = {}
    
    for doc in tqdm(documents, desc="Extracting text"):
        source_name = doc["name"]
        logger.info(f"Processing document: {source_name}")
        
        try:
            if doc["type"] == "pdf":
                pdf_reader = PdfReader(doc["content"])
                text = ""
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF {source_name} has {total_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text
                        if (page_num + 1) % 10 == 0:
                            logger.info(f"Processed {page_num + 1}/{total_pages} pages from {source_name}")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num} in {source_name}: {e}")
                
                if text:
                    text_by_source[source_name] = text
                    logger.info(f"Extracted {len(text)} characters from {source_name}")
                else:
                    logger.warning(f"No text extracted from {source_name}")
            
            elif doc["type"] == "docx":
                text = doc["content"]
                text_by_source[source_name] = text
                logger.info(f"Extracted {len(text)} characters from {source_name}")
            
        except Exception as e:
            logger.error(f"Error processing document {source_name}: {e}")
    
    return text_by_source

def get_text_chunks(text_by_source):
    """Split text into chunks with source tracking"""
    logger.info("Splitting text into chunks")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    metadata_list = []
    
    for source, text in text_by_source.items():
        logger.info(f"Creating chunks for source: {source}")
        try:
            chunks = text_splitter.split_text(text)
            # Create metadata for each chunk
            source_metadata = [{"source": source} for _ in chunks]
            
            all_chunks.extend(chunks)
            metadata_list.extend(source_metadata)
            
            logger.info(f"Created {len(chunks)} chunks from {source}")
        except Exception as e:
            logger.error(f"Error splitting text from {source}: {e}")
    
    logger.info(f"Created {len(all_chunks)} total chunks with metadata")
    return all_chunks, metadata_list

def check_qdrant_connection(url, api_key):
    """Test connection to Qdrant server"""
    logger.info(f"Testing connection to Qdrant at {url}")
    try:
        client = QdrantClient(url=url, api_key=api_key if api_key else None)
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x)
                    logger.info(f"Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

@retry_with_backoff(retries=3)
def initialize_embeddings():
    """Initialize HuggingFace embeddings with retry logic"""
    try:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")

        # First try loading from cache
        logger.info("Attempting to load embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=cache_dir,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test the embeddings with a simple input
        logger.info("Testing embeddings model...")
        test_text = "Testing embeddings model"
        test_embedding = embeddings.embed_query(test_text)
        logger.info(f"Successfully initialized embeddings model (vector size: {len(test_embedding)})")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        logger.info("Attempting alternative initialization...")
        
        try:
            # Try alternative initialization without cache
            logger.info("Attempting to load model without cache...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Test the embeddings
            test_text = "Testing embeddings model"
            test_embedding = embeddings.embed_query(test_text)
            logger.info(f"Successfully initialized embeddings model with alternative method")
            
            return embeddings
            
        except Exception as e2:
            logger.error(f"Failed alternative initialization: {str(e2)}")
            raise Exception(f"Failed to initialize embeddings after multiple attempts: {str(e)} | {str(e2)}")

def get_vectorstore(text_chunks, metadata_list):
    """Create Qdrant vector store from text chunks with metadata"""
    logger.info("Creating Qdrant vector store from text chunks")
    
    if not check_qdrant_connection(qdrant_url, qdrant_api_key):
        logger.error("Cannot proceed without Qdrant connection")
        return None
    
    try:
        # Initialize embeddings with better error handling
        logger.info("Initializing HuggingFace embeddings")
        embeddings = initialize_embeddings()
        if not embeddings:
            logger.error("Failed to initialize embeddings")
            return None
            
        logger.info(f"Connecting to Qdrant at {qdrant_url}")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=60  # Increase timeout for large operations
        )
        
        # Check if collection exists, if not create it
        collections = client.get_collections().collections
        collection_exists = any(
            collection.name == collection_name 
            for collection in collections
        )
        
        if collection_exists:
            logger.info(f"Collection {collection_name} exists. Recreating...")
            client.delete_collection(collection_name=collection_name)
            collection_exists = False
        
        if not collection_exists:
            logger.info(f"Creating new collection: {collection_name}")
            # Get embedding dimension
            sample_embedding = embeddings.embed_query("Sample text")
            dimension = len(sample_embedding)
            
            # Create collection with appropriate parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
        
        # Create vector store with metadata
        logger.info(f"Creating Qdrant vector store with {len(text_chunks)} chunks")
        batch_size = 100  # Process in smaller batches
        
        for i in range(0, len(text_chunks), batch_size):
            batch_end = min(i + batch_size, len(text_chunks))
            logger.info(f"Processing batch {i//batch_size + 1} ({i} to {batch_end})")
            
            batch_texts = text_chunks[i:batch_end]
            batch_metadata = metadata_list[i:batch_end]
            
            if i == 0:  # First batch
                vectorstore = Qdrant.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadata,
                    url=qdrant_url,
                    collection_name=collection_name,
                    api_key=qdrant_api_key if qdrant_api_key else None,
                )
            else:  # Subsequent batches
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadata
                )
            
            logger.info(f"Successfully processed batch {i//batch_size + 1}")
        
        logger.info("Qdrant vector store created successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating Qdrant vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def ingest_documents():
    """Main function to ingest documents into the vector database"""
    logger.info("Starting document ingestion")
    success_count = 0
    failure_count = 0
    
    try:
        # Load documents from directory
        documents = load_documents_from_directory()
        
        if documents:
            # Get document text with source tracking
            logger.info("Extracting text from documents")
            text_by_source = get_document_text(documents)
            
            if text_by_source:
                # Get text chunks and metadata
                logger.info("Creating text chunks with metadata")
                text_chunks, metadata_list = get_text_chunks(text_by_source)
                
                if text_chunks:
                    # Create vector store
                    logger.info("Creating Qdrant vector store")
                    vectorstore = get_vectorstore(text_chunks, metadata_list)
                    
                    if vectorstore:
                        logger.info(f"""
                        Document ingestion completed successfully:
                        - Total documents processed: {len(documents)}
                        - Total chunks created: {len(text_chunks)}
                        - Successfully ingested: {len(text_by_source)} documents
                        - Failed: {len(documents) - len(text_by_source)} documents
                        """)
                        return True
                    else:
                        logger.error("Failed to create vector store")
                        return False
                else:
                    logger.error("No text chunks created")
                    return False
            else:
                logger.error("No text extracted from documents")
                return False
        else:
            logger.warning("No legal documents found")
            logger.info("Creating empty vector store for testing")
            # Create empty vector store with placeholder text
            try:
                empty_chunks = ["This is a placeholder text for testing the Zimbabwe legal assistant without actual documents."]
                empty_metadata = [{"source": "placeholder"}]
                vectorstore = get_vectorstore(empty_chunks, empty_metadata)
                
                if vectorstore:
                    logger.info("Created empty vector store for testing")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to create empty vector store: {e}")
                return False
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Starting Legal Documents Ingestion ===")
    start_time = time.time()
    
    if ingest_documents():
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        logger.info(f"Document ingestion completed successfully in {duration} seconds!")
        print(f"Document ingestion completed successfully in {duration} seconds!")
    else:
        logger.error("Document ingestion failed. Check logs for details.")
        print("Document ingestion failed. Check logs for details.") 