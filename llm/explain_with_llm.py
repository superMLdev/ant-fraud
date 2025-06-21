from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_transaction(transaction_features: dict, prediction: int, probability: float = None):
    """
    Use an LLM to generate a human-readable explanation of the prediction.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        return "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
    if not transaction_features or not isinstance(transaction_features, dict):
        return "Invalid transaction features provided. Please provide a dictionary of features."
    if prediction not in [0, 1]:
        return "Invalid prediction value. Must be 0 (Legit) or 1 (Fraud)."
    if probability is not None and (not isinstance(probability, (float, int)) or not (0 <= probability <= 1)):
        return "Invalid probability value. Must be a float between 0 and 1."

    # Convert transaction features to a string representation
    prompt = f"""A fraud detection system made a prediction. 
Here are the transaction details:
{transaction_features}

Prediction: {"Fraud" if prediction == 1 else "Legit"}
Probability: {round(probability * 100, 2)}%

User History (only if available): {transaction_features.get("user_history", "N/A")}

Explain the decision in simple terms. Why might this prediction make sense based on the data?
"""
    print(prompt)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert fraud analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        explanation = response.choices[0].message.content
        return explanation.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"
