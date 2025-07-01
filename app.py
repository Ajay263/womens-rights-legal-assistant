from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
import os
from functools import wraps
from rag_manager import RAGManager
import asyncio
from datetime import datetime
import logging
from sqlalchemy import inspect, Sequence
import time
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-123')
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'postgresql://chatapp:chatapp123@localhost:5432/chat_feedback'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
stripe.api_key = os.environ.get(
    'STRIPE_SECRET_KEY',
    'sk_test_51Ljl0CCtuWZn1W9YWK0hDRpTaHtTSqZEusWzLkbisUdzP6qToInFssfrObrsvj'
    'OXMXCfryywQdxTFwqrMHGeIDWc00AbSy5Cvc'
)
YOUR_DOMAIN = 'http://localhost:4242'

# Initialize RAG Manager
rag_manager = RAGManager()

# Price to plan mapping
PRICE_ID_TO_PLAN = {
    'price_1RRYWoCtuWZn1W9YWK0hDRpTaHtTSqZEusWzLkbisUdzP6qToInFssfrObrsvj': 'basic',
    'price_1RRYX8CtuWZn1W9Y7wrzsEu6': 'premium'
}

SUBSCRIPTION_LIMITS = {
    'free': 5,
    'basic': 15,
    'premium': 100
}

class User(db.Model):
    id = db.Column(
        db.Integer,
        Sequence('user_id_seq', start=1, increment=1),
        primary_key=True
    )
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    phone_number = db.Column(db.String(20))
    address = db.Column(db.String(200))
    subscription = db.Column(db.String(50), default='free')
    usage_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    conversations = db.relationship('Conversation', backref='user', lazy=True)

class Conversation(db.Model):
    id = db.Column(
        db.Integer, 
        Sequence('conversation_id_seq', start=1, increment=1),
        primary_key=True
    )
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    feedback = db.Column(db.String(20), nullable=True)  # 'positive' or 'negative'
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    response_time = db.Column(db.Float, nullable=False)
    relevance = db.Column(db.String(20), nullable=False)
    relevance_explanation = db.Column(db.Text)
    model_used = db.Column(db.String(100), nullable=False)
    prompt_tokens = db.Column(db.Integer, nullable=False, default=0)
    completion_tokens = db.Column(db.Integer, nullable=False, default=0)
    total_tokens = db.Column(db.Integer, nullable=False, default=0)
    eval_prompt_tokens = db.Column(db.Integer, nullable=False, default=0)
    eval_completion_tokens = db.Column(db.Integer, nullable=False, default=0)
    eval_total_tokens = db.Column(db.Integer, nullable=False, default=0)
    api_cost = db.Column(db.Float, nullable=False, default=0.0)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d').date()
        phone_number = request.form.get('phone_number')
        address = request.form.get('address')

        if User.query.filter_by(email=email).first():
            return 'Email already exists!'
            
        new_user = User(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            phone_number=phone_number,
            address=address
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        return 'Invalid credentials!'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    user = db.session.get(User, session['user_id'])
    if not user:
        # If user not found, clear session and redirect to login
        session.pop('user_id', None)
        return redirect(url_for('login'))
    
    # Get initial conversations for the user
    conversations = Conversation.query.filter_by(user_id=user.id)\
        .order_by(Conversation.created_at.desc())\
        .all()
    
    conversation_list = [{
        'id': conv.id,
        'question': conv.question,
        'answer': conv.answer,
        'feedback': conv.feedback,
        'created_at': conv.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for conv in conversations]
    
    # Ensure subscription and usage data is available
    subscription = user.subscription or 'free'
    usage_count = user.usage_count or 0
    
    return render_template(
        'index.html',
        user=user,
        SUBSCRIPTION_LIMITS=SUBSCRIPTION_LIMITS,
        conversations=conversation_list,
        subscription=subscription,
        usage_count=usage_count
    )

@app.route('/get_conversations', methods=['GET'])
@login_required
def get_conversations():
    try:
        user = db.session.get(User, session['user_id'])
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get all conversations for the user, ordered by creation date
        conversations = Conversation.query.filter_by(user_id=user.id)\
            .order_by(Conversation.created_at.desc())\
            .all()

        # Format conversations for the response
        conversation_list = [{
            'id': conv.id,
            'question': conv.question,
            'answer': conv.answer,
            'feedback': conv.feedback,
            'created_at': conv.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for conv in conversations]

        return jsonify({"conversations": conversation_list})
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    try:
        data = request.get_json()
        if not data or 'price_id' not in data:
            print("Error: No price_id in request data")
            return jsonify(error="No price_id provided"), 400

        print(f"Creating checkout session for price_id: {data['price_id']}")
        print(f"User ID: {session['user_id']}")

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': data['price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.host_url.rstrip('/') + 
                '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.host_url.rstrip('/') + '/cancel',
            metadata={'user_id': session['user_id']}
        )
        print(f"Checkout session created successfully: {checkout_session.id}")
        return jsonify({'id': checkout_session.id})
    except stripe.error.StripeError as e:
        print(f"Stripe error: {str(e)}")
        return jsonify(error=str(e)), 403
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/success')
@login_required
def success():
    session_id = request.args.get('session_id')
    if session_id:
        checkout_session = stripe.checkout.Session.retrieve(
            session_id,
            expand=['line_items']
        )
        price_id = checkout_session.line_items.data[0].price.id
        user = db.session.get(User, session['user_id'])
        user.subscription = PRICE_ID_TO_PLAN.get(price_id, 'free')
        user.usage_count = 0  # Reset usage on new subscription
        db.session.commit()
    return render_template('success.html')

@app.route('/cancel')
@login_required
def cancel():
    return render_template('cancel.html')

@app.route('/subscription')
@login_required
def subscription():
    user = db.session.get(User, session['user_id'])
    return render_template('subscription.html', user=user)

def run_async(coro):
    """Helper function to run async code in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/ask-llm', methods=['POST'])
@login_required
def ask_llm():
    logger.info(f"Session user_id: {session.get('user_id')}")
    user = db.session.get(User, session['user_id'])
    if not user:
        logger.error(f"User not found in database for id: {session['user_id']}")
        session.pop('user_id', None)
        return jsonify({"error": "User not found. Please login again."}), 401
        
    if user.usage_count >= SUBSCRIPTION_LIMITS.get(user.subscription, 0):
        return jsonify({
            "error": (
                f"You have used up your {user.subscription} tier limit. "
                "Please upgrade your plan."
            ),
            "current_plan": user.subscription,
            "usage": user.usage_count,
            "limit": SUBSCRIPTION_LIMITS.get(user.subscription, 0)
        }), 403
    
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Get response from RAG system using our async helper with timing
        start_time = time.time()
        response = run_async(rag_manager.get_response(question))
        end_time = time.time()
        response_time = round(end_time - start_time, 3)
        
        if "error" in response:
            return jsonify(response), 500

        # Evaluate answer relevance
        relevance_result = run_async(
            evaluate_answer_relevance(
                question=question,
                answer=response.get('answer', '')
            )
        )
        
        # Get model name from response or use default
        model_name = response.get('model_used', 'llama3-70b-8192')
        
        # Extract token information with proper defaults
        tokens = response.get('tokens', {})
        prompt_tokens = max(0, tokens.get('prompt_tokens', 0))
        completion_tokens = max(0, tokens.get('completion_tokens', 0))
        total_tokens = max(0, tokens.get('total_tokens', 0) or (prompt_tokens + completion_tokens))
        
        # Extract evaluation tokens with proper defaults
        eval_tokens = relevance_result.get('tokens', {})
        eval_prompt_tokens = max(0, eval_tokens.get('prompt_tokens', 0))
        eval_completion_tokens = max(0, eval_tokens.get('completion_tokens', 0))
        eval_total_tokens = max(0, eval_tokens.get('total_tokens', 0) or 
                              (eval_prompt_tokens + eval_completion_tokens))
        
        # Calculate API costs for both main response and evaluation
        main_cost = calculate_groq_cost(
            {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            },
            model_name
        )
        
        eval_cost = calculate_groq_cost(
            {
                'prompt_tokens': eval_prompt_tokens,
                'completion_tokens': eval_completion_tokens,
                'total_tokens': eval_total_tokens
            },
            'llama3-70b-8192'  # Evaluation model
        )
        
        total_cost = round(main_cost + eval_cost, 6)
        
        logger.info(
            f"Main tokens: {total_tokens} (${main_cost:.6f}), "
            f"Eval tokens: {eval_total_tokens} (${eval_cost:.6f}), "
            f"Total cost: ${total_cost:.6f}"
        )
        
        # Create a new conversation record with complete metrics
        conversation = Conversation(
            user_id=user.id,
            question=question,
            answer=response.get('answer', ''),
            response_time=response_time,
            relevance=relevance_result.get('relevance', 'UNKNOWN'),
            relevance_explanation=relevance_result.get('explanation', ''),
            model_used=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            eval_prompt_tokens=eval_prompt_tokens,
            eval_completion_tokens=eval_completion_tokens,
            eval_total_tokens=eval_total_tokens,
            api_cost=total_cost,
            created_at=datetime.utcnow()
        )
        
        db.session.add(conversation)
        user.usage_count += 1
        db.session.commit()

        remaining = SUBSCRIPTION_LIMITS.get(user.subscription, 0) - user.usage_count
        
        return jsonify({
            "answer": response.get('answer', ''),
            "remaining": remaining,
            "conversation_id": conversation.id,
            "metrics": {
                "response_time": response_time,
                "relevance": relevance_result.get('relevance'),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                    "eval_total": eval_total_tokens
                },
                "api_cost": total_cost
            }
        })

    except Exception as e:
        logger.error(f"Error in ask_llm: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

async def evaluate_answer_relevance(question, answer):
    """
    Evaluate the relevance of an answer to a question using an LLM.
    Returns relevance classification and explanation.
    """
    try:
        # Get evaluation from the RAG manager
        eval_result = await rag_manager.evaluate_relevance(question=question, answer=answer)
        
        # Parse the result
        relevance = eval_result.get('relevance', 'UNKNOWN')
        explanation = eval_result.get('explanation', 'No explanation provided')
        tokens = eval_result.get('tokens', {})
        
        return {
            'relevance': relevance,
            'explanation': explanation,
            'tokens': tokens
        }
    except Exception as e:
        logger.error(f"Error evaluating answer relevance: {str(e)}")
        return {
            'relevance': 'ERROR',
            'explanation': f"Error evaluating relevance: {str(e)}",
            'tokens': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        }

def calculate_groq_cost(tokens: Dict[str, int], model: str = 'llama3-70b-8192') -> float:
    """Calculates the cost of Groq API usage based on token count.

    Args:
        tokens (Dict[str, int]): Dictionary containing token counts.
        model (str): The name of the model used.

    Returns:
        float: The estimated cost in USD.
    """
    # Extract total tokens, defaulting to sum of prompt and completion if not provided
    prompt_tokens = tokens.get('prompt_tokens', 0)
    completion_tokens = tokens.get('completion_tokens', 0)
    total_tokens = tokens.get('total_tokens', 0) or (prompt_tokens + completion_tokens)

    # Normalize model name for comparison
    model_lower = model.lower()

    # Groq pricing per token (as of March 2024)
    if 'llama3-70b' in model_lower or 'llama-3.1-70b' in model_lower:
        # $0.7 per million tokens ($0.0000007 per token)
        return round(total_tokens * 0.0000007, 6)
    elif 'llama3-8b' in model_lower or 'llama-3.1-8b' in model_lower:
        # $0.2 per million tokens ($0.0000002 per token)
        return round(total_tokens * 0.0000002, 6)
    elif 'deepseek' in model_lower:
        # $0.5 per million tokens ($0.0000005 per token)
        return round(total_tokens * 0.0000005, 6)
    else:
        # Default rate - $0.7 per million tokens
        return round(total_tokens * 0.0000007, 6)

@app.route('/save_feedback', methods=['POST'])
@login_required
def save_feedback():
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        feedback = data.get('feedback')
        
        if not conversation_id or not feedback:
            return jsonify({"error": "Missing required parameters"}), 400
            
        conversation = db.session.get(Conversation, conversation_id)
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        if conversation.user_id != session['user_id']:
            return jsonify({"error": "Unauthorized"}), 403
            
        conversation.feedback = feedback
        db.session.commit()
        
        return jsonify({"message": "Feedback saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_conversation', methods=['POST'])
@login_required
def delete_conversation():
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "No conversation ID provided"}), 400
            
        # Get the conversation and verify ownership
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        if conversation.user_id != session['user_id']:
            return jsonify({"error": "Unauthorized"}), 403
            
        # Delete the conversation
        db.session.delete(conversation)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Conversation deleted"})
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        logger.info("Dropping all tables...")
        db.drop_all()
        logger.info("Creating database tables...")
        db.create_all()
        logger.info("Database tables created successfully!")
        
        # Verify if tables exist
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        logger.info(f"Available tables: {tables}")
        
        # Log table schemas
        for table_name in tables:
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            logger.info(f"Table {table_name} columns: {columns}")
        
    app.run(host='0.0.0.0', port=4242)