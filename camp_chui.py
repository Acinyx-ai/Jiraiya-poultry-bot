"""
Chui — Jiraiya Poultry Farm WhatsApp AI & Dashboard (Flask)

- Uses OpenAI client with fallback models
- Twilio WhatsApp outbound
- Stores conversations & orders
- Simple dashboard
Run:
python camp_chui.py
"""

import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
from functools import wraps
from twilio.rest import Client
from openai import OpenAI

# -------------------------------------------------
# Business config (Jiraiya Poultry Farm)
# -------------------------------------------------

BUSINESS_NAME = "Jiraiya Poultry Farm"
LOCATION = "Green Valley, Narok – Olopito"
SUPPORT_PHONE = "0746522703"

EGG_PRICE = 20

DELIVERY_AREAS = {
    "narok town": 30,
    "rotian": 30
}

MPESA_NAME = "Pochi la Biashara"
MPESA_NUMBER = "0746522703"

OPENING_HOURS = "Daily 6:00 AM – 9:00 PM"


# -------------------------------------------------
# Load env
# -------------------------------------------------

def load_env():
    load_dotenv(override=False)
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
        "TWILIO_WHATSAPP_NUMBER": os.getenv("TWILIO_WHATSAPP_NUMBER"),
        "MY_WHATSAPP_NUMBER": os.getenv("MY_WHATSAPP_NUMBER"),
        "ADMIN_USER": os.getenv("ADMIN_USER", "admin"),
        "ADMIN_PASS": os.getenv("ADMIN_PASS", "chui_admin_pass"),
        "FLASK_SECRET": os.getenv("FLASK_SECRET", "chui_dev_secret"),
        "CHAT_MODEL": os.getenv("CHAT_MODEL", ""),
        "EMB_MODEL": os.getenv("EMB_MODEL", "text-embedding-3-small"),
        "PORT": int(os.getenv("PORT", 5000)),
    }


cfg = load_env()


# -------------------------------------------------
# OpenAI client
# -------------------------------------------------

def create_openai_client(api_key):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


client = create_openai_client(cfg["OPENAI_API_KEY"])


PREFERRED_CHAT_MODELS = [
    "gpt-5-turbo",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-instruct"
]

if cfg["CHAT_MODEL"]:
    PREFERRED_CHAT_MODELS.insert(0, cfg["CHAT_MODEL"])

EMB_MODEL = cfg["EMB_MODEL"]
APP_PORT = cfg["PORT"]


# -------------------------------------------------
# Twilio
# -------------------------------------------------

try:
    twilio_client = Client(cfg["TWILIO_ACCOUNT_SID"], cfg["TWILIO_AUTH_TOKEN"])
except Exception as e:
    print("Twilio init error:", e)
    twilio_client = None

TWILIO_WHATSAPP_NUMBER = cfg["TWILIO_WHATSAPP_NUMBER"]
MY_WHATSAPP_NUMBER = cfg["MY_WHATSAPP_NUMBER"]

ADMIN_USER = cfg["ADMIN_USER"]
ADMIN_PASS = cfg["ADMIN_PASS"]

SECRET_KEY = cfg["FLASK_SECRET"]

DB_PATH = Path("camp.db")


# -------------------------------------------------
# Flask
# -------------------------------------------------

app = Flask(__name__)
app.secret_key = SECRET_KEY


# -------------------------------------------------
# Database
# -------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        whatsapp_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        whatsapp_id TEXT,
        order_json TEXT,
        status TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS kb (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        embedding BLOB,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_conversation(whatsapp_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (whatsapp_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (whatsapp_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def save_order_structured(whatsapp_id, order_dict, status="new"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO orders (whatsapp_id, order_json, status, created_at) VALUES (?, ?, ?, ?)",
        (whatsapp_id, json.dumps(order_dict), status, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# -------------------------------------------------
# OpenAI helpers
# -------------------------------------------------

def refresh_openai_client():
    load_dotenv(override=True)
    new_key = os.getenv("OPENAI_API_KEY")
    global client
    client = create_openai_client(new_key)
    return client


def call_chat_with_fallback(prompt_text, system_prompt):

    if not client:
        refresh_openai_client()

    if not client:
        return "AI system not configured."

    last_err = None

    for model_name in PREFERRED_CHAT_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            if "401" in msg or "invalid_api_key" in msg:
                refresh_openai_client()
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e2:
                    last_err = e2
                    continue

            continue

    return f"[AI error: {last_err}]"


def embedding_for_text(text):
    if not client:
        refresh_openai_client()

    if not client:
        return np.zeros(1536, dtype=np.float32)

    try:
        r = client.embeddings.create(model=EMB_MODEL, input=[text])
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception:
        return np.zeros(1536, dtype=np.float32)


# -------------------------------------------------
# SYSTEM PROMPT (CHICKEN BUSINESS)
# -------------------------------------------------

SYSTEM_PROMPT = f"""
You are the official WhatsApp assistant for {BUSINESS_NAME}, a poultry business in Kenya.

Business details:
- Location: {LOCATION}
- We currently sell eggs only.
- Price: Ksh {EGG_PRICE} per egg.
- Delivery areas and fees:
  - Narok town: Ksh 30
  - Rotian: Ksh 30
- Payment method: {MPESA_NAME}, number {MPESA_NUMBER}
- Opening hours: {OPENING_HOURS}
- Support phone: {SUPPORT_PHONE}

Your job:
1. Answer customer questions clearly and briefly.
2. Help customers place egg orders.
3. Always confirm:
   - number of eggs
   - delivery area (Narok town or Rotian)
   - whether it is pickup or delivery
4. Do not invent other products.
5. Be simple and professional.
"""


# -------------------------------------------------
# WhatsApp send
# -------------------------------------------------

def send_whatsapp_text(to, text):
    if not twilio_client:
        return
    try:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=text
        )
    except Exception as e:
        print("Send error:", e)


# -------------------------------------------------
# Webhook
# -------------------------------------------------

@app.route("/webhook", methods=["POST"])
def webhook():

    data = request.form
    from_number = data.get("From", "")
    msg_body = data.get("Body", "").strip()

    if not msg_body:
        return ("", 200)

    # Optional restriction (keep if you want)
    if MY_WHATSAPP_NUMBER and from_number != MY_WHATSAPP_NUMBER:
        send_whatsapp_text(
            from_number,
            "⛔ This business number is currently private. Please contact the owner."
        )
        return ("", 200)

    add_conversation(from_number, "user", msg_body)

    text = msg_body.lower()

    # -------------------------------------------------
    # Order capture shortcut
    # -------------------------------------------------

    if any(k in text for k in ["order", "buy", "eggs", "yai", "mayai"]):

        order = {
            "raw_message": msg_body,
            "product": "eggs",
            "price_per_egg": EGG_PRICE,
            "business": BUSINESS_NAME
        }

        save_order_structured(from_number, order)

        reply = (
            "✅ Your egg request has been received.\n\n"
            "Please reply with:\n"
            "• Number of eggs\n"
            "• Delivery area (Narok town or Rotian)\n"
            "• Pickup or delivery\n\n"
            "Our staff will confirm your order shortly."
        )

        add_conversation(from_number, "assistant", reply)
        send_whatsapp_text(from_number, reply)
        return ("", 200)

    # -------------------------------------------------
    # AI response
    # -------------------------------------------------

    _ = embedding_for_text(msg_body)

    reply = call_chat_with_fallback(msg_body, SYSTEM_PROMPT)

    add_conversation(from_number, "assistant", reply)
    send_whatsapp_text(from_number, reply)

    return ("", 200)


# -------------------------------------------------
# Dashboard
# -------------------------------------------------

if not Path("dashboard.html").exists():
    DASH_HTML = "<h3>Dashboard file missing. Upload dashboard.html to same folder.</h3>"
else:
    DASH_HTML = open("dashboard.html", "r", encoding="utf-8").read()


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
            request.form.get("username") == ADMIN_USER
            and request.form.get("password") == ADMIN_PASS
        ):
            session["logged_in"] = True
            return redirect("/dashboard")

    return """
    <form method="post">
      <input name="username" placeholder="username">
      <input name="password" type="password" placeholder="password">
      <button>Login</button>
    </form>
    """


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template_string(DASH_HTML)


@app.route("/api/conversations")
@login_required
def api_conversations():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT whatsapp_id, role, content, created_at FROM conversations ORDER BY id DESC LIMIT 200"
    )
    rows = cur.fetchall()
    conn.close()

    return jsonify([
        {
            "whatsapp_id": r[0],
            "role": r[1],
            "content": r[2],
            "created_at": r[3]
        }
        for r in rows
    ])


@app.route("/api/orders")
@login_required
def api_orders():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT whatsapp_id, order_json, status, created_at FROM orders ORDER BY id DESC LIMIT 200"
    )
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        try:
            o = json.loads(r[1])
        except:
            o = {"raw": r[1]}

        out.append({
            "whatsapp_id": r[0],
            "order": o,
            "status": r[2],
            "created_at": r[3]
        })

    return jsonify(out)


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":
    init_db()
    print(f"🐔 {BUSINESS_NAME} bot running on http://127.0.0.1:{APP_PORT}/dashboard")
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
