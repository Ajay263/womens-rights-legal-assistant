import psycopg2
from psycopg2.extras import DictCursor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection with error handling and logging."""
    try:
        logger.info("Attempting to connect to the database")
        conn = psycopg2.connect(
            host="postgres",
            database="chat_feedback",
            user="chatapp",
            password="chatapp123",  # In production, use environment variables
            cursor_factory=DictCursor
        )
        logger.info("Successfully connected to the database")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def init_db():
    """Initialize the database schema with logging."""
    logger.info("Starting database initialization")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Drop existing tables if they exist
            logger.info("Dropping existing tables if they exist")
            cur.execute("""
                DROP TABLE IF EXISTS feedback;
                DROP TABLE IF EXISTS conversations;
            """)
            
            # Create conversations table
            logger.info("Creating conversations table")
            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_time FLOAT,
                    relevance VARCHAR(20),
                    relevance_explanation TEXT,
                    model_used VARCHAR(50),
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    eval_prompt_tokens INTEGER,
                    eval_completion_tokens INTEGER,
                    eval_total_tokens INTEGER,
                    api_cost FLOAT
                )
            """)
            
            # Create feedback table
            logger.info("Creating feedback table")
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback_type INTEGER NOT NULL,  -- 1 for thumbs up, -1 for thumbs down
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_conversation
                        FOREIGN KEY(conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)
            
        conn.commit()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()

def save_conversation(conversation_id: str, question: str, answer: str, metrics: dict = None) -> bool:
    """Save a conversation with metrics and logging."""
    logger.info(f"Saving conversation with ID: {conversation_id}")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Prepare metrics data if available
            if not metrics:
                metrics = {}
                
            response_time = metrics.get('response_time')
            relevance = metrics.get('relevance')
            relevance_explanation = metrics.get('relevance_explanation')
            model_used = metrics.get('model_used')
            prompt_tokens = metrics.get('prompt_tokens')
            completion_tokens = metrics.get('completion_tokens')
            total_tokens = metrics.get('total_tokens')
            eval_prompt_tokens = metrics.get('eval_prompt_tokens')
            eval_completion_tokens = metrics.get('eval_completion_tokens')
            eval_total_tokens = metrics.get('eval_total_tokens')
            api_cost = metrics.get('api_cost')
            
            cur.execute(
                """
                INSERT INTO conversations (
                    id, question, answer, response_time, relevance, 
                    relevance_explanation, model_used, prompt_tokens, 
                    completion_tokens, total_tokens, eval_prompt_tokens, 
                    eval_completion_tokens, eval_total_tokens, api_cost
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id, question, answer, response_time, relevance,
                    relevance_explanation, model_used, prompt_tokens, 
                    completion_tokens, total_tokens, eval_prompt_tokens,
                    eval_completion_tokens, eval_total_tokens, api_cost
                )
            )
        conn.commit()
        logger.info(f"Successfully saved conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_feedback(conversation_id: str, feedback_type: int) -> bool:
    """Save user feedback with logging."""
    logger.info(f"Saving feedback for conversation {conversation_id}: {feedback_type}")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (conversation_id, feedback_type)
                VALUES (%s, %s)
                """,
                (conversation_id, feedback_type)
            )
        conn.commit()
        logger.info(f"Successfully saved feedback for conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_conversation_feedback(conversation_id: str):
    """Retrieve feedback for a specific conversation."""
    logger.info(f"Retrieving feedback for conversation {conversation_id}")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT feedback_type, timestamp
                FROM feedback
                WHERE conversation_id = %s
                ORDER BY timestamp DESC
                """,
                (conversation_id,)
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        return None
    finally:
        conn.close() 