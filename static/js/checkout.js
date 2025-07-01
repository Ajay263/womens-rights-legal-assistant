
const stripe = Stripe('pk_test_51Ljl0CCtuWZn1W9YDIiAIM5R5aSFFFqcolf1WXFu2JKTxupjBIWlS9T4KAYZIC7qv5NOnpQEBP1WHU7JWKnV8vgo00Aq4SIK1y');

document.querySelectorAll('.checkout-button').forEach(button => {
    button.addEventListener('click', function() {
        const priceId = this.dataset.priceId;
        
        fetch('/create-checkout-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ price_id: priceId })
        })
        .then(response => response.json())
        .then(session => stripe.redirectToCheckout({ sessionId: session.id }))
        .catch(error => console.error('Error:', error));
    });
});