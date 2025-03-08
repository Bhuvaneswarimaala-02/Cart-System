# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os
# import re
# import pymongo
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import groq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_together import TogetherEmbeddings
# from langchain_groq import ChatGroq
# from datetime import datetime

# # ✅ Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend communication

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")

# # Initialize Groq client
# client = groq.Client(api_key=api_key)
# os.environ["TOGETHER_API_KEY"] = 'c97530aceaffcb28eecbaefd032551e2075fd3d323aea71f598cad92122c7d69'
# store = {}

# # ✅ MongoDB Setup
# mongo_client = MongoClient("mongodb://localhost:27017/")
# db = mongo_client["farm2bag"]
# products_collection = db["products"]
# cart_collection = db["cart"]

# # ✅ Vector Embeddings Model
# embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# # ✅ LLM Model
# llm1 = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=api_key)

# # ✅ Function to Add Items to Cart
# def add_to_cart(user_id, item_name, quantity):
#     item = products_collection.find_one({"name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
#     if not item:
#         return f"❌ Item '{item_name}' not found in the database."
    
#     price = item["price"]
#     cart_collection.insert_one({"user_id": user_id, "item_name": item_name, "quantity": quantity, "price": price})
    
#     return f"🛒 Added {item_name} (Quantity: {quantity}) to your cart at ₹{price} per unit."

# # ✅ Function to Remove Item from Cart
# def remove_from_cart(user_id, item_name):
#     result = cart_collection.delete_one({"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
#     if result.deleted_count > 0:
#         return f"✅ Removed {item_name} from your cart."
#     else:
#         return f"❌ {item_name} is not in your cart."

# # ✅ Function to Update Cart Item Quantity
# def update_cart_item(user_id, item_name, new_quantity):
#     item = cart_collection.find_one({"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
#     if item:
#         price_per_unit = item["price"]
#         new_price = new_quantity * price_per_unit
#         cart_collection.update_one(
#             {"user_id": user_id, "item_name": item["item_name"]},
#             {"$set": {"quantity": new_quantity, "price": new_price}}
#         )
#         return f"✅ Updated {item_name} to {new_quantity} units in your cart."
#     return f"❌ {item_name} is not in your cart."

# # ✅ Function to Show Cart
# def show_cart(user_id):
#     cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1, "price": 1}))
#     if not cart_items:
#         return "🛒 Your cart is currently empty."

#     response = "**🛍️ Your Cart:**\n\n"
#     subtotal = sum(item["price"] * item["quantity"] for item in cart_items)

#     for item in cart_items:
#         response += f"🔹 {item['quantity']} x {item['item_name']} - ₹{item['price']} each\n"

#     response += f"\n**Total: ₹{subtotal}**\nWould you like to continue shopping or checkout?"
#     return response

# # ✅ Function to Recommend Products
# def recommend_products(user_id):
#     """Recommends products based on items in the user's cart."""
#     user_cart_items = list(cart_collection.find({"user_id": user_id}))
#     cart_product_names = [item["item_name"] for item in user_cart_items]
#     all_products = list(products_collection.find({}))

#     recommended_items = []
#     for product in all_products:
#         if product["name"] not in cart_product_names:
#             recommended_items.append(product["name"])
#         if len(recommended_items) >= 5:
#             break
    
#     return recommended_items if recommended_items else ["Organic Rice", "Cold-Pressed Oil"]

# # ✅ Function to get product category
# def get_product_category(product_name):
#     product = products_collection.find_one({"name": product_name})
#     return product.get("category", "") if product else ""

# def get_price(item_name):
#     """Fetches the price of an item from the products collection."""
#     product = products_collection.find_one({"name": item_name})
#     return product["price"] if product else 0

    
# def get_cart_items(user_id):
#     cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))
#     return cart_items

# # ✅ Serve Chatbot UI
# @app.route('/chat', methods=['GET'])
# def chat_page():
#     return render_template("index.html")  # Loads templates/index.html

# @app.route('/')
# def home():
#     return render_template("index2.html")  # Load index2.html by default

# # ✅ API Endpoint for Processing User Queries
# @app.route('/process_chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get("message", "").strip().lower()
#     user_id = "user123"  # Replace with actual user ID

#     # Recognizing different commands
#     if re.search(r"\bshow\s+(my\s+)?cart\b", user_message):
#         response = show_cart(user_id)
#     elif re.search(r"\brecommend\s+(me\s+)?(some\s+)?items?\b", user_message):
#         recommended_items = recommend_products(user_id)
#         response = "🌟 You might also like: " + ", ".join(recommended_items)
#     elif re.search(r"\badd\s+(\d+)?\s*([\w\s]+)\s+to\s+my\s+cart\b", user_message):
#         match = re.search(r"\badd\s+(\d+)?\s*([\w\s]+)\s+to\s+my\s+cart\b", user_message)
#         quantity = int(match.group(1)) if match.group(1) else 1
#         item_name = match.group(2).strip()
#         response = add_to_cart(user_id, item_name, quantity)
#     elif re.search(r"\bremove\s+([\w\s]+)\s+from\s+my\s+cart\b", user_message):
#         match = re.search(r"\bremove\s+([\w\s]+)\s+from\s+my\s+cart\b", user_message)
#         item_name = match.group(1).strip()
#         response = remove_from_cart(user_id, item_name)
#     elif re.search(r"\bupdate\s+([\w\s]+)\s+to\s+(\d+)\s*(kg|pcs)?\b", user_message):
#         match = re.search(r"\bupdate\s+([\w\s]+)\s+to\s+(\d+)\s*(kg|pcs)?\b", user_message)
#         item_name = match.group(1).strip()
#         new_quantity = int(match.group(2))
#         response = update_cart_item(user_id, item_name, new_quantity)

#     # ✅ Extra Functionality: Fetch Cart Items
#     elif re.search(r"\b(list|what's|what are)\s+(in\s+)?(my\s+)?cart\b", user_message):
#         cart_items = get_cart_items(user_id)
#         if cart_items:
#             response = "**🛍️ Items in Your Cart:**\n"
#             response += "\n".join([f"🔹 {item['quantity']} x {item['item_name']}" for item in cart_items])
#         else:
#             response = "🛒 Your cart is currently empty."

#     # ✅ Extra Functionality: Check Item Prices
#     elif re.search(r"\b(price of|cost of|how much is)\s+([\w\s]+)\b", user_message):
#         match = re.search(r"\b(price of|cost of|how much is)\s+([\w\s]+)\b", user_message)
#         item_name = match.group(2).strip()
#         price = get_price(item_name)
#         if price:
#             response = f"💰 The price of {item_name} is ₹{price} per unit."
#         else:
#             response = f"❌ Sorry, {item_name} is not available in our inventory."

#     # ✅ Extra Functionality: Check Product Category
#     elif re.search(r"\b(category of|what type is|which category does)\s+([\w\s]+)\b", user_message):
#         match = re.search(r"\b(category of|what type is|which category does)\s+([\w\s]+)\b", user_message)
#         item_name = match.group(2).strip()
#         category = get_product_category(item_name)
#         if category:
#             response = f"📦 {item_name} belongs to the '{category}' category."
#         else:
#             response = f"❌ Sorry, I couldn't find the category for {item_name}."

#     # ✅ Extra Functionality: Recommend Based on Cart
#     elif re.search(r"\b(suggest|recommend|show me)\s+(some\s+)?(similar|complementary)?\s*products?\b", user_message):
#         cart_items = get_cart_items(user_id)
#         if cart_items:
#             recommendations = recommend_products(user_id)
#             response = "🌟 Based on your cart, you might like:\n" + "\n".join(recommendations)
#         else:
#             response = "🛒 Your cart is empty. Add items to get recommendations!"

#     else:
#         response = f"How can I help?"

#     return jsonify({"response": response})

# # ✅ Run Flask Server
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)










from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import pymongo
import requests
from bs4 import BeautifulSoup
import openai
from pymongo import MongoClient
from dotenv import load_dotenv
import groq
from langchain_together import TogetherEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime
import groq
import os
import json
import re
import PyPDF2
import pymongo
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_together import TogetherEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = groq.Client(api_key=api_key)
os.environ["TOGETHER_API_KEY"] = 'c97530aceaffcb28eecbaefd032551e2075fd3d323aea71f598cad92122c7d69'
store = {}

# ✅ Initialize Groq client
client = groq.Client(api_key=api_key)
os.environ["TOGETHER_API_KEY"] = "c97530aceaffcb28eecbaefd032551e2075fd3d323aea71f598cad92122c7d69"

# ✅ MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["farm2bag"]
products_collection = db["products"]
cart_collection = db["cart"]

# ✅ Vector Embeddings Model
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# ✅ LLM Model
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=api_key)
# LLM Model
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ✅ Function to Scrape Website Content (Farm2Bag Assistant)
FIXED_WEBSITE_URL = "https://www.farm2bag.com/en"
# openai.api_key = "YOUR_OPENAI_API_KEY"
# openai.api_key = os.getenv("OPENAI_API_KEY")


def scrape_website():
    """Scrapes the content from the fixed Farm2Bag website."""
    try:
        response = requests.get(FIXED_WEBSITE_URL)
        soup = BeautifulSoup(response.text, "html.parser")
        content = " ".join([p.text for p in soup.find_all("p")])
        return content[:5000]
    except Exception as e:
        return f"Error scraping website: {e}"

# ✅ Function to Generate Replies Based on Website
# deepseek model
def generate_response(user_input, website_content):
    """Generates chatbot response using website content via DeepSeek."""
    prompt = f"""
    You are an AI chatbot that only answers questions based on the provided website content.
    Your tone is fun, engaging, and professional.

    Website Content:
    {website_content}

    User: {user_input}
    Assistant:
    """
    try:
        response = llm.invoke(prompt)  # ✅ Using DeepSeek via ChatGroq
        return response
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ✅ Function to Add Items to Cart
def add_to_cart(user_id, item_name, quantity):
    item = products_collection.find_one({"name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
    if not item:
        return f"❌ Item '{item_name}' not found\n\n"
    
    price = item["price"]
    cart_collection.insert_one({"user_id": user_id, "item_name": item_name, "quantity": quantity, "price": price})
    return f"🛒 Added {item_name} (Quantity: {quantity}) at ₹{price} per unit\n\n"

# ✅ Function to Remove Item from Cart
def remove_from_cart(user_id, item_name):
    result = cart_collection.delete_one({"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
    if result.deleted_count > 0:
        return f"✅ Removed {item_name} from your cart\n\n"
    else:
        return f"❌ {item_name} is not in your cart\n\n"

# ✅ Function to Show Cart
def show_cart(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1, "price": 1}))
    if not cart_items:
        return "🛒 Your cart is empty.\n\n"

    response = "**🛍️ Your Cart:**\n\n"
    for item in cart_items:
        response += f"🔹 {item['quantity']} x {item['item_name']} - ₹{item['price']} each\n\n"

    return response

def update_cart_item(user_id, item_name, new_quantity):
    """Updates the quantity of an item in the cart properly."""
    
    # Normalize item name (remove extra spaces)
    item_name = item_name.strip()

    # Find existing item in a case-insensitive way
    existing_item = cart_collection.find_one(
        {"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}}
    )

    if existing_item:
        # Get price per unit from the cart, or fetch from products if not stored
        price_per_unit = existing_item.get("price", 0)
        if not price_per_unit:
            product = products_collection.find_one({"name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
            price_per_unit = product["price"] if product else 0

        # Calculate new total price
        new_price = new_quantity * price_per_unit

        # Update the item quantity and price
        cart_collection.update_one(
            {"user_id": user_id, "item_name": existing_item["item_name"]},  # Exact name from DB
            {"$set": {"quantity": new_quantity, "price": new_price}}
        )
        return f"✅ Updated {existing_item['item_name']} to {new_quantity} units in your cart."

    return f"❌ {item_name} is not in your cart."

# ✅ Serve Chatbot UIs
@app.route('/')
def home():
    return render_template("index2.html")  # Default: Farm2Bag Assistant

@app.route('/chat')
def chat_page():
    return render_template("index.html")  # Loads Cart Assistant chatbot

# ✅ Function to recommend products based on cart items
def recommend_products(user_id):
    """Recommends products based on items in the user's cart."""
    user_cart_items = list(cart_collection.find({"user_id": user_id}))
    cart_product_names = [item["item_name"] for item in user_cart_items]
    all_products = list(products_collection.find({}))
    
    recommended_items = []
    for product in all_products:
        if product["name"] in cart_product_names:
            continue
        
        for cart_item in cart_product_names:
            if is_similar_or_complementary(product["name"], cart_item):
                recommended_items.append(product["name"])
                break
    
    return recommended_items[:5] if recommended_items else ["Organic Rice", "Cold-Pressed Oil"]

# ✅ Function to determine similar or complementary products
def is_similar_or_complementary(product_name, cart_item):
    product_category = get_product_category(product_name)
    cart_item_category = get_product_category(cart_item)
    
    if product_category == cart_item_category:
        return True
    
    if cart_item.lower() in ["oranges", "apples"] and "juice" in product_name.lower():
        return True
    
    return False

# ✅ Function to get product category
def get_product_category(product_name):
    product = products_collection.find_one({"name": product_name})
    return product.get("category", "") if product else ""

def get_cart_items(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))
    return cart_items

def get_price(item_name):
    """Fetches the price of an item from the products collection."""
    product = products_collection.find_one({"name": item_name})
    return product["price"] if product else 0




# ✅ Flask API to process user chat
# @app.route('/process_chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get("message", "").strip()
#     user_id = "user123"

#     # 🧠 Use LLM to classify intent and extract entities
#     extracted_data = classify_intent_and_extract_entities(user_message)
#     intent = extracted_data.get("intent")
#     item_name = extracted_data.get("item", "").strip()
#     quantity = extracted_data.get("quantity")

#     # 🎯 Call the appropriate function based on the intent
#     if intent == "show_cart":
#         response = show_cart(user_id)
#     elif intent == "recommend_items":
#         recommended_items = recommend_products(user_id)
#         response = "🌟 You might also like: " + ", ".join(recommended_items)
#     elif intent == "add_to_cart" and item_name:
#         response = add_to_cart(user_id, item_name, int(quantity) if quantity else 1)
#     elif intent == "remove_from_cart" and item_name:
#         response = remove_from_cart(user_id, item_name)
#     elif intent == "update_cart" and item_name and quantity:
#         response = update_cart_item(user_id, item_name, int(quantity))
#     elif intent == "get_price" and item_name:
#         price = get_price(item_name)
#         response = f"💰 The price of {item_name} is ₹{price} per unit." if price else f"❌ Sorry, {item_name} is not available in our inventory."
#     elif intent == "get_category" and item_name:
#         category = get_product_category(item_name)
#         response = f"📦 {item_name} belongs to the '{category}' category." if category else f"❌ Sorry, I couldn't find the category for {item_name}."
#     else:
#         response = "❓ Sorry, I didn't understand. Can you rephrase?"

#     return jsonify({"response": response})

# @app.route('/process_chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get("message", "").strip()
#     user_id = "user123"

#     # 🧠 Use LLM to extract action, item, and quantity
#     extracted_data = extract_action_details(user_message)
    
#     action = extracted_data.get("action")
#     item_name = extracted_data.get("item", "").strip()
#     quantity = extracted_data.get("quantity")

#     print(f"🔎 Extracted Data: {extracted_data}")  # ✅ Debugging step

#     # 🎯 Call the appropriate function dynamically
#     if action == "show_cart":
#         response = show_cart(user_id)
#     elif action == "recommend_products":
#         recommended_items = recommend_products(user_id)
#         response = "🌟 You might also like: " + ", ".join(recommended_items)
#     elif action == "add_to_cart" and item_name:
#         response = add_to_cart(user_id, item_name, int(quantity) if quantity else 1)
#     elif action == "remove_from_cart" and item_name:
#         response = remove_from_cart(user_id, item_name)
#     elif action == "update_cart" and item_name and quantity:
#         response = update_cart_item(user_id, item_name, int(quantity))
#     elif action == "get_price" and item_name:
#         price = get_price(item_name)
#         response = f"💰 The price of {item_name} is ₹{price} per unit." if price else f"❌ Sorry, {item_name} is not available in our inventory."
#     else:
#         response = "❓ Sorry, I didn't understand. Can you rephrase?"

#     return jsonify({"response": response})

import json

def extract_intent_and_entities(user_message):
    """
    Uses the LLM to extract intent and key entities (item, quantity) from user input.
    """
    prompt = f'''
    You are an intelligent chatbot that extracts the user’s intent and relevant details from their message.
    Extract the details in JSON format.

    User Message: "{user_message}"

    Respond in JSON format:
    {{
        "intent": "<intent>",
        "item": "<item_name>",
        "quantity": "<quantity>"
    }}

    Possible intents:
    - "show_cart" (when the user wants to view their cart)
    - "add_to_cart" (when adding an item)
    - "remove_from_cart" (when removing an item)
    - "update_cart" (when updating item quantity)
    - "get_price" (when asking for an item's price)
    - "recommend_items" (when asking for product recommendations)
    - "unknown" (if the intent is unclear)
    '''

    try:
        response = llm.invoke(prompt)  # Send prompt to LLM
        return json.loads(response)  # Convert LLM response to JSON
    except:
        return {"intent": "unknown"}  # Default response if extraction fails


# @app.route('/process_chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get("message", "").strip()
#     user_id = "user123"  # Replace with actual user identification logic

#     # ✅ Add to Cart
#     add_to_cart_pattern = re.compile(r"add\s+(\d+)?\s*([a-zA-Z\s]+)\s+to\s+my\s+cart", re.IGNORECASE)
#     match = add_to_cart_pattern.search(user_message)

#     if match:
#         quantity = int(match.group(1)) if match.group(1) else 1
#         item_name = match.group(2).strip()
#         response = add_to_cart(user_id, item_name, quantity)
#         return jsonify({"response": response})

#     # ✅ Remove from Cart
#     remove_from_cart_pattern = re.compile(r"remove\s+([a-zA-Z\s]+)\s+from\s+my\s+cart", re.IGNORECASE)
#     remove_match = remove_from_cart_pattern.search(user_message)

#     if remove_match:
#         item_name = remove_match.group(1).strip()
#         response = remove_from_cart(user_id, item_name)
#         return jsonify({"response": response})

#     # ✅ Update Cart Item Quantity
#     update_cart_pattern = re.compile(r"update\s+([a-zA-Z\s]+)\s+to\s+(\d+)\s*(kg|pcs)?", re.IGNORECASE)
#     update_match = update_cart_pattern.search(user_message)

#     if update_match:
#         item_name = update_match.group(1).strip()
#         new_quantity = int(update_match.group(2))
#         response = update_cart_item(user_id, item_name, new_quantity)
#         return jsonify({"response": response})

#     # ✅ Show Cart
#     show_cart_pattern = re.compile(r"show\s+(my\s+)?cart", re.IGNORECASE)
#     if show_cart_pattern.search(user_message):
#         response = show_cart(user_id)
#         return jsonify({"response": response})

#     # ✅ Recommend Items
#     recommend_pattern = re.compile(r"recommend\s+(me\s+)?(some\s+)?items?", re.IGNORECASE)
#     if recommend_pattern.search(user_message):
#         recommended_items = recommend_products(user_id)
#         response = "🌟 You might also like:\n" + "\n".join(recommended_items) if recommended_items else "🌟 No recommendations available at the moment."
#         return jsonify({"response": response})

#     # ✅ Load FAISS Vectorstore for RAG
#     try:
#         vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         retriever = vectorstore.as_retriever()

#         # ✅ Create Chat Prompt Template
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a Farm2Bag customer service assistant. {context}"),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}")
#         ])

#         # ✅ Create Retrieval-Augmented Generation (RAG) Chain
#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
#         )

#         # ✅ Generate Response from RAG
#         response = conversational_rag_chain.invoke({"input": user_message}, config={"configurable": {"session_id": "abc123"}})["answer"]
#         bot_reply = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

#         return jsonify({"response": bot_reply})
#     except Exception as e:
#         return jsonify({"response": "❌ Error processing your request. Please try again."})
    
@app.route('/process_chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = "user123"  # Replace with actual user identification logic

    # 🧠 Extract intent and entities using the LLM
    extracted_data = extract_intent_and_entities(user_message)
    intent = extracted_data.get("intent")
    item_name = extracted_data.get("item", "").strip()
    quantity = extracted_data.get("quantity")

    print(f"🔎 Extracted Data: {extracted_data}")  # Debugging step

    # 🎯 Define actions based on extracted intent
    intent_mapping = {
        "show_cart": lambda: show_cart(user_id),
        "recommend_items": lambda: "🌟 " + ", ".join(recommend_products(user_id)),
        "add_to_cart": lambda: add_to_cart(user_id, item_name, int(quantity) if quantity else 1),
        "remove_from_cart": lambda: remove_from_cart(user_id, item_name),
        "update_cart": lambda: update_cart_item(user_id, item_name, int(quantity)),
        "get_price": lambda: f"💰 {item_name} costs ₹{get_price(item_name)}" if get_price(item_name) else f"❌ {item_name} not available."
    }

    # 🔥 Handle unknown intent by falling back to RAG (Vector Search)
    if intent == "unknown":
        try:
            # ✅ Process regex-based actions as fallback
            add_to_cart_pattern = re.compile(r"add\s+(\d+)?\s*([a-zA-Z\s]+)\s+to\s+my\s+cart", re.IGNORECASE)
            match = add_to_cart_pattern.search(user_message)

            if match:
                quantity = int(match.group(1)) if match.group(1) else 1
                item_name = match.group(2).strip()
                response = add_to_cart(user_id, item_name, quantity)
                return jsonify({"response": response})

            remove_from_cart_pattern = re.compile(r"remove\s+([a-zA-Z\s]+)\s+from\s+my\s+cart", re.IGNORECASE)
            remove_match = remove_from_cart_pattern.search(user_message)

            if remove_match:
                item_name = remove_match.group(1).strip()
                response = remove_from_cart(user_id, item_name)
                return jsonify({"response": response})

            update_cart_pattern = re.compile(r"update\s+([a-zA-Z\s]+)\s+to\s+(\d+)\s*(kg|pcs)?", re.IGNORECASE)
            update_match = update_cart_pattern.search(user_message)

            if update_match:
                item_name = update_match.group(1).strip()
                new_quantity = int(update_match.group(2))
                response = update_cart_item(user_id, item_name, new_quantity)
                return jsonify({"response": response})

            show_cart_pattern = re.compile(r"show\s+(my\s+)?cart", re.IGNORECASE)
            if show_cart_pattern.search(user_message):
                response = show_cart(user_id)
                return jsonify({"response": response})

            recommend_pattern = re.compile(r"recommend\s+(me\s+)?(some\s+)?items?", re.IGNORECASE)
            if recommend_pattern.search(user_message):
                recommended_items = recommend_products(user_id)
                response = "🌟 You might also like:\n" + "\n".join(recommended_items) if recommended_items else "🌟 No recommendations available at the moment."
                return jsonify({"response": response})

            # ✅ Load FAISS Vectorstore for RAG
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()

            # ✅ Create Chat Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Farm2Bag customer service assistant. {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            # ✅ Create Retrieval-Augmented Generation (RAG) Chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
            )

            # ✅ Generate Response from RAG
            response = conversational_rag_chain.invoke({"input": user_message}, config={"configurable": {"session_id": "abc123"}})["answer"]
            bot_reply = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            return jsonify({"response": bot_reply})

        except Exception as e:
            response = f"❌ Error processing your request: {str(e)}"

    else:
        # ✅ Execute the mapped function for recognized intent
        response = intent_mapping.get(intent, lambda: "I didn't understand your request.")()

    return jsonify({"response": response})



# ✅ Function to Get Chat Session History
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



# ✅ API Endpoint for Farm2Bag Assistant Chatbot
@app.route('/process_chat_web', methods=['POST'])
def process_chat_web():
    data = request.json
    user_message = data.get("message", "").strip()

    website_content = scrape_website()
    response = generate_response(user_message, website_content)

    return jsonify({"response": response})

# ✅ Run Flask Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
