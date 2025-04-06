import os
import json
import uuid
import redis
import markdown
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, session
from model import get_response_from_messages_hf
from pyngrok import ngrok

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

redis_client = redis.StrictRedis(host=os.getenv("HOST"), port=os.getenv("PORT"),
                                 db=os.getenv("DB"), decode_responses=True)


def get_user_chats(user_id):
    chats_data = redis_client.hget("users", user_id)
    return json.loads(chats_data) if chats_data else {}


def save_user_chats(user_id, chats):
    redis_client.hset("users", user_id, json.dumps(chats))


@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_id = session['user_id']
    chats = get_user_chats(user_id)

    return render_template('index.html', chats=chats)


@app.route('/new_chat', methods=['POST'])
def new_chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_id = session['user_id']
    chats = get_user_chats(user_id)

    new_chat_id = str(uuid.uuid4())
    chats[new_chat_id] = {"chat_name": f"Чат {len(chats) + 1}",
                          "messages": [{"role": "system",
                                        "content": 'Ты чат-бот'}]}

    save_user_chats(user_id, chats)

    return jsonify({"chat_id": new_chat_id, "chats": list(chats.keys())})


@app.route('/chat/<chat_id>', methods=['GET', 'POST'])
def chat(chat_id):
    user_id = session['user_id']
    chats = get_user_chats(user_id)

    if chat_id not in chats:
        chats[chat_id] = {"chat_name": f"Чат {len(chats) + 1}", "messages": []}

    if request.method == 'POST':
        user_message = request.json.get('message', '')
        chats[chat_id]['messages'].append({"role": "user", "content": user_message})

        # Не используем summarization для ответов модели (недостаточно мощностей), будем просто ограничивать
        assistant_message = get_response_from_messages_hf([chats[chat_id]['messages'][0]] +
                                                          chats[chat_id]['messages'][1:][-5:])
        assistant_message = markdown.markdown(assistant_message)
        chats[chat_id]['messages'].append({"role": "assistant", "content": assistant_message})

        save_user_chats(user_id, chats)
        return jsonify({"response": assistant_message})

    return jsonify({"messages": chats[chat_id]["messages"][1:]})


if __name__ == '__main__':
    print(ngrok.connect(5000))
    app.run(debug=False)
