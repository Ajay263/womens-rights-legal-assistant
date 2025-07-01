import pytest
from unittest.mock import patch, MagicMock
from rag_utils import (
    load_env_variables,
    get_vectorstore_connection,
    get_conversation_chain,
    process_question,
)

# --- Fixtures and Mocks ---
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test_qdrant_key")
    monkeypatch.setenv("QDRANT_COLLECTION", "zimbabwe_legal_docs")

@pytest.fixture
def mock_vectorstore():
    mock_retriever = MagicMock()
    mock_retriever.as_retriever.return_value = mock_retriever
    mock_retriever.as_retriever().search_kwargs = {"k": 5}
    return mock_retriever

# --- Test load_env_variables ---
def test_load_env_variables(mock_env):
    env_vars = load_env_variables()
    assert env_vars["groq_api_key"] == "test_groq_key"
    assert env_vars["qdrant_url"] == "http://localhost:6333"
    assert env_vars["qdrant_api_key"] == "test_qdrant_key"
    assert env_vars["collection_name"] == "zimbabwe_legal_docs"

# --- Test get_vectorstore_connection ---
@patch("rag_utils.HuggingFaceEmbeddings")
@patch("rag_utils.QdrantClient")
@patch("rag_utils.Qdrant")
def test_get_vectorstore_connection(mock_qdrant_class, mock_client_class, mock_embeddings_class, mock_env):
    mock_client_instance = MagicMock()
    mock_client_instance.get_collections.return_value.collections = [
        MagicMock(name="zimbabwe_legal_docs")
    ]
    mock_client_class.return_value = mock_client_instance
    mock_qdrant_class.return_value = MagicMock()

    result = get_vectorstore_connection("http://localhost:6333", "test_api_key", "zimbabwe_legal_docs")
    assert result is not None
    mock_qdrant_class.assert_called_once()

# --- Test get_conversation_chain ---
@patch("rag_utils.ChatGroq")
@patch("rag_utils.create_stuff_documents_chain")
@patch("rag_utils.create_retrieval_chain")
def test_get_conversation_chain(mock_create_retrieval, mock_create_doc_chain, mock_chatgroq, mock_env):
    mock_vectorstore = MagicMock()
    mock_create_retrieval.return_value = MagicMock()
    mock_create_doc_chain.return_value = MagicMock()
    mock_chatgroq.return_value = MagicMock()

    chain = get_conversation_chain(mock_vectorstore)
    assert "retrieval_chain" in chain
    assert "memory" in chain
    assert "model_name" in chain

# --- Test process_question ---
@patch("rag_utils.save_conversation")
@patch("rag_utils.generate_conversation_id", return_value="test-convo-id")
def test_process_question(mock_convo_id, mock_save_convo):
    mock_chain = {
        "retrieval_chain": MagicMock(),
        "memory": MagicMock(),
        "model_name": "llama-3.3-70b-versatile"
    }
    mock_chain["retrieval_chain"].invoke.return_value = {
        "answer": "This is a test answer"
    }
    mock_chain["memory"].chat_memory.messages = []

    result = process_question("What is the law on inheritance for women?", mock_chain)

    assert result["conversation_id"] == "test-convo-id"
    assert result["answer"] == "This is a test answer"
    assert isinstance(result["response_time"], float)
