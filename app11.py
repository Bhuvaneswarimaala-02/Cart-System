from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import openai  # Replace with any LLM provider like Mistral, Claude

app = Flask(__name__)
CORS(app)  # Enable cross-origin access for frontend requests

# ✅ Fixed website for scraping (REPLACE WITH YOUR WEBSITE)
FIXED_WEBSITE_URL = "website.com"

# ✅ OpenAI API Key (REPLACE WITH YOUR KEY)
openai.api_key = "API_KEY"

# ✅ Function to Scrape Website Content
def scrape_website():
    try:
        response = requests.get(FIXED_WEBSITE_URL)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from paragraphs (modify as needed)
        content = " ".join([p.text for p in soup.find_all("p")])  
        return content[:5000]  # Limit to avoid token overflow
    except Exception as e:
        return f"Error scraping website: {e}"

# ✅ Function to Generate Fun and Interactive Replies
def generate_response(user_input, website_content):
    prompt = f"""
    You are an AI chatbot that only answers questions based on the provided website content. 
    Your tone is fun, engaging, and professional. 

    Website Content:
    {website_content}

    User: {user_input}
    Assistant:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with Mistral, Claude, etc.
            messages=[{"role": "system", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating response: {e}"

# ✅ Serve `index2.html`
@app.route('/')
def home():
    return render_template("index2.html")

# ✅ NEW API Endpoint for `index2.html` Chatbot
@app.route('/process_chat_web', methods=['POST'])  # 🔹 Renamed to avoid conflict
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    # Fetch latest website content
    website_content = scrape_website()

    # Generate chatbot response
    response = generate_response(user_message, website_content)

    return jsonify({"response": response})

# ✅ Run Flask Server on a Different Port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)  # 🔹 Changed port to 5002