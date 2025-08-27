# /app/config/rules.py

import re

# This file contains the "rules engine" for understanding text messages.
# Rules are processed in order, defining their priority.

# Precompiled regex for performance
WORD_RE = re.compile(r'\w+')

# A controlled vocabulary of known keywords for fuzzy matching.
# IMPORTANT: Update this list if you add new product types or materials.
VALID_KEYWORDS = [
    "ruby", "necklace", "earring", "bangle", "bracelet", "ring", "pendant",
    "choker", "chain", "set", "jhumka", "kundan", "victorian", "oxidised",
    "matte", "cz", "nakshi", "layered", "short", "long", "bridal", "stone",
    "diamond", "emerald", "sapphire", "pearl", "gold", "silver", "jewelry"
]

# Mapping plurals to singular for consistent searching
PLURAL_MAPPINGS = {
    "necklaces": "necklace", "earrings": "earring", "bangles": "bangle",
    "bracelets": "bracelet", "rings": "ring", "pendants": "pendant", "charms": "charm",
    "chains": "chain", "anklets": "anklet", "chokers": "choker", "sets": "set",
    "collections": "collection", "pieces": "piece", "jhumkas": "jhumka", "diamonds": "diamond",
    "rubies": "ruby", "emeralds": "emerald", "sapphires": "sapphire", "pearls": "pearl",
    "stones": "stone", "accessories": "accessory"
}


# Intent rules organized by priority.
# Each rule is a tuple: ({single_word_tokens}, [multi_word_phrases], "intent_name")
INTENT_RULES = [
    # Group 1: High-Priority Commands (Directly trigger product lists)
    ({"latest", "new", "newest", "recent", "arrivals", "fresh", "just", "added"}, [], "latest_arrivals_inquiry"),
    ({"bestseller", "popular", "trending", "top", "selling", "favorite"}, ["best selling", "best sellers"], "bestseller_inquiry"),

    # Group 2: Transactional & Support Inquiries
    ({"human", "agent", "person", "representative", "someone"}, ["talk to human", "speak to a person", "talk to someone", "customer service"], "human_escalation"),
    ({"help", "support", "problem", "issue", "complaint", "refund", "return", "exchange", "cancel", "payment", "billing", "damaged", "broken", "defective", "wrong", "incorrect", "bad", "poor", "dull", "unfortunate", "nonsense"}, ["speak to someone", "talk to agent", "not the same", "wrong item", "bad delivery", "poor quality"], "support"),
    ({"shipping", "shipped", "ship", "deliver", "courier", "cost", "charge", "charges", "fee", "fees", "policy", "policies", "before", "urgent", "asap", "tomorrow", "today", "rush", "express", "fast", "quick", "deadline"}, ["shipping cost", "shipping policy", "delivery time", "how long", "when it will be delivered", "needed by", "out for delivery", "in transit", "on the way"], "shipping_inquiry"),
    ({"tracking", "dispatched", "delayed", "cancelled", "processing", "confirmed", "pending", "status"}, ["where is my order", "order status", "track my order", "shipping status", "my order"], "order_inquiry"),

    # Group 3: Informational Inquiries
    ({"contact", "phone", "email", "address", "location", "store", "visit"}, ["how to contact", "get in touch", "customer care"], "contact_inquiry"),
    ({"review", "reviews", "rating", "ratings", "feedback", "testimonial", "testimonials"}, ["what do people say", "customer reviews", "google reviews"], "review_inquiry"),
    ({"discount", "offer", "sale", "coupon", "deal", "promo", "code"}, ["any offers", "current deals", "discount code"], "discount_inquiry"),
    ({"reseller", "reselling", "broadcast", "group"}, ["reseller group", "whatsapp group"], "reseller_inquiry"),
    ({"wholesale", "bulk"}, ["bulk order", "buy in bulk", "bulk pricing"], "bulk_order_inquiry"),
    ({"expensive", "cheap", "costly", "affordable", "budget", "pricey", "reasonable", "high", "low", "fair"}, ["too much", "worth it", "value for money", "overpriced", "good deal"], "price_feedback"),
    ({"price", "cost", "much", "rate", "rupees", "rs", "₹"}, ["how much", "what is the price", "whats the price"], "price_inquiry"),
    ({"size", "fit", "adjustable", "length", "diameter", "measurement", "loose", "tight"}, ["too big", "too small", "what size", "ring size", "chain length"], "product_inquiry"),
    ({"stock", "available", "inventory", "restock"}, ["in stock", "out of stock", "sold out", "back in stock", "when available"], "stock_inquiry"),

    # Group 4: Search & Conversational Flow
    ({"more", "other", "different", "alternatives", "similar", "else"}, ["show more", "any other", "something else", "more options"], "more_results"),
    ({"earring", "earrings", "necklace", "necklaces", "ring", "rings", "bracelet", "bracelets", "bangle", "bangles", "pendant", "pendants", "chain", "chains", "jhumka", "jhumkas", "set", "sets", "gold", "silver", "diamond", "ruby", "emerald", "sapphire", "pearl", "navaratna", "jewelry", "jewellery"}, [], "product_search"),
]

# Conversational patterns (handled separately due to context requirements)
GREETING_KEYWORDS_SET = {"hi", "hello", "hey", "morning", "afternoon", "evening", "namaste"}
THANK_KEYWORDS_SET = {"thanks", "thank", "grateful", "appreciate", "thankyou"}
AFFIRMATIVE_RESPONSES = {"yes", "sure", "ok", "okay", "yep", "y"}
NEGATIVE_RESPONSES = {"no", "nope", "nah", "not really"}

# High-priority context words to differentiate a greeting from a support request
HIGH_PRIORITY_CONTEXT_WORDS = {
    "order", "track", "return", "refund", "shipping", "delivery",
    "payment", "cancel", "exchange", "problem", "issue", "help",
    "urgent", "deadline", "before", "by", "needed"
}

# Prefixes for interactive message replies
INTERACTIVE_PREFIXES = {
    "buy_": "interactive_button_reply",
    "more_": "interactive_button_reply",
    "similar_": "interactive_button_reply",
    "option_": "interactive_button_reply",
    "product_": "product_detail"
}