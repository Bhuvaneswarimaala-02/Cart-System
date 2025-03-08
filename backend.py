# ✅ API Endpoint for Cart Assistant Chatbot
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