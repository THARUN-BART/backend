import os
import json
import traceback
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask App
app = Flask(__name__)

# Environment variables
FCM_SERVER_KEY = os.getenv('FCM_SERVER_KEY')
ONESIGNAL_APP_ID = os.getenv('ONESIGNAL_APP_ID')
ONESIGNAL_API_KEY = os.getenv('ONESIGNAL_API_KEY')
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')

# Firebase Initialization
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    cred = credentials.Certificate(json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']))
else:
    cred_path = 'matche-39f37-firebase-adminsdk-fbsvc-50793ed379.json'
    cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Big 5 keys
BIG5_KEYS = ['big5_O', 'big5_C', 'big5_E', 'big5_A', 'big5_N']

def flatten_availability(user):
    availability = user.get('availability', {})
    flat_slots = []
    if isinstance(availability, dict):
        for day, slots in availability.items():
            if isinstance(slots, list):
                flat_slots.extend([f"{day}_{slot}" for slot in slots])
            elif isinstance(slots, str):
                flat_slots.append(f"{day}_{slots}")
    elif isinstance(availability, list):
        flat_slots = availability
    return flat_slots

def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def skill_difference(set1, set2):
    return 1 - jaccard(set1, set2)

def big5_vector(user):
    return np.array([float(user.get(k, 0.5)) for k in BIG5_KEYS]).reshape(1, -1)

def personality_compatibility(vec1, vec2):
    sim = cosine_similarity(vec1, vec2)[0, 0]
    return (sim + 1) / 2  # scale from [-1, 1] to [0, 1]

# Weights for matching
MATCH_WEIGHTS = {
    'interest': 2.0,
    'availability': 1.5,
    'skills': 2.0,
    'personality': 1.0
}


@app.route('/cluster', methods=['GET'])
def cluster():
    user_id = request.args.get('userId')
    top_n = int(request.args.get('top', 5))

    users_snap = db.collection('users').stream()
    users = []
    user_ids = []

    for doc in users_snap:
        data = doc.to_dict()
        data['uid'] = doc.id
        users.append(data)
        user_ids.append(doc.id)

    if user_id not in user_ids:
        return jsonify({'error': 'User not found'}), 404

    idx = user_ids.index(user_id)
    user = users[idx]
    user_interests = user.get('interests', [])
    user_skills = user.get('skills', [])
    user_avail = flatten_availability(user)
    user_personality = big5_vector(user)

    scores = []
    for i, other in enumerate(users):
        if i == idx:
            continue
        interest_sim = jaccard(user_interests, other.get('interests', []))
        avail_sim = jaccard(user_avail, flatten_availability(other))
        skill_diff = skill_difference(user_skills, other.get('skills', []))
        other_personality = big5_vector(other)
        personality_sim = personality_compatibility(user_personality, other_personality)

        score = (
            MATCH_WEIGHTS['interest'] * interest_sim +
            MATCH_WEIGHTS['availability'] * avail_sim +
            MATCH_WEIGHTS['skills'] * skill_diff +
            MATCH_WEIGHTS['personality'] * personality_sim
        )
        scores.append((score, i))

    scores.sort(reverse=True)
    top = scores[:top_n]

    result = []
    for score, i in top:
        result.append({
            "uid": users[i]['uid'],
            "similarity": f"{round(score * 10, 1)}"
        })

    return jsonify(result)


@app.route('/user/<user_id>', methods=['GET'])
def get_user(user_id):
    try:
        doc = db.collection('users').document(user_id).get()
        if not doc.exists:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save-fcm-token', methods=['POST'])
def save_fcm_token():
    data = request.json
    user_id = data.get('user_id')
    fcm_token = data.get('fcm_token')
    if not user_id or not fcm_token:
        return jsonify({'error': 'Missing user_id or fcm_token'}), 400
    db.collection('users').document(user_id).set({'fcm_token': fcm_token}, merge=True)
    return jsonify({'message': 'Token saved'}), 200

@app.route('/notify-user', methods=['POST'])
def notify_user():
    data = request.json
    user_id = data.get('user_id')
    title = data.get('title', 'Notification')
    message = data.get('message', '')
    if not user_id or not message:
        return jsonify({'error': 'Missing user_id or message'}), 400

    doc = db.collection('users').document(user_id).get()
    if not doc.exists:
        return jsonify({'error': 'User not found'}), 404
    fcm_token = doc.to_dict().get('fcm_token')
    if not fcm_token:
        return jsonify({'error': 'User has no FCM token'}), 404

    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "include_external_user_ids": [fcm_token],
        "headings": {"en": title},
        "contents": {"en": message}
    }
    response = requests.post("https://onesignal.com/api/v1/notifications", json=payload, headers={
        "Authorization": f"Basic {ONESIGNAL_API_KEY}",
        "Content-Type": "application/json"
    })
    return jsonify(response.json()), response.status_code


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
