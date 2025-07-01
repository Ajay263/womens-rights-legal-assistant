import pytest
import json
import os
from datetime import datetime, date
from unittest.mock import patch, MagicMock, AsyncMock
from werkzeug.security import generate_password_hash

# Set test environment variables
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['STRIPE_SECRET_KEY'] = 'sk_test_fake_key_for_testing'

# Mock the RAG manager before importing app
import sys
from unittest.mock import MagicMock

# Create a mock RAG manager
mock_rag_manager = MagicMock()
mock_rag_manager.get_response = AsyncMock()
mock_rag_manager.evaluate_relevance = AsyncMock()

# Replace the module before importing
sys.modules['rag_manager'] = MagicMock()
sys.modules['rag_manager'].RAGManager = MagicMock(return_value=mock_rag_manager)

# Now import the app
from app import app, db, User, Conversation, SUBSCRIPTION_LIMITS


@pytest.fixture(scope='function')
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SECRET_KEY'] = 'test-secret-key'
    
    with app.test_client() as test_client:
        with app.app_context():
            db.create_all()
            yield test_client
            db.session.remove()
            db.drop_all()


@pytest.fixture
def test_user():
    """Create a test user in the database."""
    user = User(
        email='test@example.com',
        password=generate_password_hash('testpassword'),
        first_name='John',
        last_name='Doe',
        date_of_birth=date(1990, 1, 1),
        phone_number='1234567890',
        address='123 Test St',
        subscription='free',
        usage_count=0
    )
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def logged_in_session(client, test_user):
    """Create a logged-in session for testing protected routes."""
    with client.session_transaction() as sess:
        sess['user_id'] = test_user.id
    return test_user


def test_register_get(client):
    """Test GET request to register page."""
    response = client.get('/register')
    assert response.status_code == 200


def test_register_post_success(client):
    """Test successful user registration."""
    data = {
        'email': 'newuser@example.com',
        'password': 'newpassword',
        'first_name': 'Jane',
        'last_name': 'Smith',
        'date_of_birth': '1995-05-15',
        'phone_number': '9876543210',
        'address': '456 New St'
    }
    response = client.post('/register', data=data)
    assert response.status_code == 302  # Redirect to login
    
    # Verify user was created
    user = User.query.filter_by(email='newuser@example.com').first()
    assert user is not None
    assert user.first_name == 'Jane'


def test_login_get(client):
    """Test GET request to login page."""
    response = client.get('/login')
    assert response.status_code == 200


def test_login_post_success(client, test_user):
    """Test successful login."""
    data = {
        'email': 'test@example.com',
        'password': 'testpassword'
    }
    response = client.post('/login', data=data)
    assert response.status_code == 302  # Redirect to index


def test_login_post_invalid_credentials(client, test_user):
    """Test login with invalid credentials."""
    data = {
        'email': 'test@example.com',
        'password': 'wrongpassword'
    }
    response = client.post('/login', data=data)
    assert b'Invalid credentials!' in response.data


def test_logout(client, logged_in_session):
    """Test user logout."""
    response = client.get('/logout')
    assert response.status_code == 302  # Redirect to login


def test_index_requires_login(client):
    """Test that index page requires login."""
    response = client.get('/')
    assert response.status_code == 302  # Redirect to login


def test_index_with_login(client, logged_in_session):
    """Test index page when logged in."""
    response = client.get('/')
    assert response.status_code == 200


def test_get_conversations_success(client, logged_in_session):
    """Test successful retrieval of conversations."""
    # Create test conversation
    conv = Conversation(
        user_id=logged_in_session.id,
        question="Test question",
        answer="Test answer",
        response_time=1.5,
        relevance="HIGH",
        relevance_explanation="Very relevant",
        model_used="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        api_cost=0.001
    )
    db.session.add(conv)
    db.session.commit()
    
    response = client.get('/get_conversations')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'conversations' in data
    assert len(data['conversations']) == 1


def test_get_conversations_requires_login(client):
    """Test that get_conversations requires login."""
    response = client.get('/get_conversations')
    assert response.status_code == 302  # Redirect to login


@patch('app.run_async')
def test_ask_llm_success(mock_run_async, client, logged_in_session):
    """Test successful LLM query."""
    # Mock the async responses
    mock_rag_response = {
        'answer': 'This is a test answer',
        'model_used': 'llama3-70b-8192',
        'tokens': {
            'prompt_tokens': 50,
            'completion_tokens': 100,
            'total_tokens': 150
        }
    }
    
    mock_eval_response = {
        'relevance': 'HIGH',
        'explanation': 'Very relevant answer',
        'tokens': {
            'prompt_tokens': 20,
            'completion_tokens': 30,
            'total_tokens': 50
        }
    }
    
    mock_run_async.side_effect = [mock_rag_response, mock_eval_response]
    
    data = {'question': 'What is the meaning of life?'}
    response = client.post('/ask-llm', 
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'answer' in response_data
    assert response_data['answer'] == 'This is a test answer'


def test_ask_llm_usage_limit_exceeded(client, logged_in_session):
    """Test LLM query when usage limit is exceeded."""
    # Set user usage to exceed free limit
    logged_in_session.usage_count = SUBSCRIPTION_LIMITS['free']
    db.session.commit()
    
    data = {'question': 'Test question'}
    response = client.post('/ask-llm',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 403
    response_data = json.loads(response.data)
    assert 'error' in response_data


def test_ask_llm_no_question(client, logged_in_session):
    """Test LLM query without providing a question."""
    data = {}
    response = client.post('/ask-llm',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert response_data['error'] == 'No question provided'


def test_save_feedback_success(client, logged_in_session):
    """Test successful feedback saving."""
    # Create a conversation
    conversation = Conversation(
        user_id=logged_in_session.id,
        question="Test question",
        answer="Test answer",
        response_time=1.0,
        relevance="HIGH",
        model_used="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        api_cost=0.001
    )
    db.session.add(conversation)
    db.session.commit()
    
    data = {
        'conversation_id': conversation.id,
        'feedback': 'positive'
    }
    response = client.post('/save_feedback',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['message'] == 'Feedback saved successfully'


def test_delete_conversation_success(client, logged_in_session):
    """Test successful conversation deletion."""
    conversation = Conversation(
        user_id=logged_in_session.id,
        question="Test question",
        answer="Test answer",
        response_time=1.0,
        relevance="HIGH",
        model_used="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        api_cost=0.001
    )
    db.session.add(conversation)
    db.session.commit()
    
    data = {'conversation_id': conversation.id}
    response = client.post('/delete_conversation',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['success'] is True


@patch('stripe.checkout.Session.create')
def test_create_checkout_session_success(mock_stripe_create, client, logged_in_session):
    """Test successful checkout session creation."""
    mock_session = MagicMock()
    mock_session.id = 'cs_test_123'
    mock_stripe_create.return_value = mock_session
    
    data = {'price_id': 'price_test_123'}
    response = client.post('/create-checkout-session',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['id'] == 'cs_test_123'


def test_create_checkout_session_no_price_id(client, logged_in_session):
    """Test checkout session creation without price_id."""
    data = {}
    response = client.post('/create-checkout-session',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert response_data['error'] == 'No price_id provided'


def test_cancel_page(client, logged_in_session):
    """Test cancel page."""
    response = client.get('/cancel')
    assert response.status_code == 200


def test_calculate_groq_cost_llama3_70b():
    """Test cost calculation for LLaMA3-70B model."""
    from app import calculate_groq_cost
    
    tokens = {'prompt_tokens': 1000, 'completion_tokens': 500, 'total_tokens': 1500}
    cost = calculate_groq_cost(tokens, 'llama3-70b-8192')
    expected_cost = 1500 * 0.0000007  # $0.7 per million tokens
    assert cost == round(expected_cost, 6)


def test_user_model_creation(client):
    """Test User model creation and attributes."""
    user = User(
        email='model_test@example.com',
        password=generate_password_hash('password123'),
        first_name='Test',
        last_name='User',
        date_of_birth=date(1992, 6, 15),
        phone_number='555-0123',
        address='789 Model St',
        subscription='premium',
        usage_count=10
    )
    db.session.add(user)
    db.session.commit()
    
    assert user.id is not None
    assert user.email == 'model_test@example.com'
    assert user.subscription == 'premium'
    assert user.usage_count == 10