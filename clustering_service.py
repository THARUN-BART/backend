import os
import json
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, messaging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv

from notification import send_onesignal_notification, send_fcm_notification

# Load environment variables
load_dotenv()

# Flask app initialization (if not already present)
app = Flask(__name__)

# FCM and OneSignal credentials from environment variables
FCM_SERVER_KEY = os.getenv('FCM_SERVER_KEY')
ONESIGNAL_APP_ID = os.getenv('ONESIGNAL_APP_ID')
ONESIGNAL_API_KEY = os.getenv('ONESIGNAL_API_KEY')

# --- Firebase Admin SDK Initialization ---
# Use environment variable if set, otherwise default to backend/serviceAccountKey.json
SERVICE_ACCOUNT_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'matche-39f37-firebase-adminsdk-fbsvc-50793ed379.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

# Firestore client (if needed elsewhere)
db = firestore.client()

BIG5_KEYS = ['big5_O', 'big5_C', 'big5_E', 'big5_A', 'big5_N']

def flatten_availability(user):
    availability = user.get('availability', {})
    flat_slots = []
    if isinstance(availability, dict):
        for day, slots in availability.items():
            if isinstance(slots, list):
                for slot in slots:
                    flat_slots.append(f"{day}_{slot}")
            elif isinstance(slots, str):
                flat_slots.append(f"{day}_{slots}")
    elif isinstance(availability, list):
        flat_slots = availability
    return flat_slots

@app.route('/cluster', methods=['GET'])
def cluster():
    user_id = request.args.get('userId')
    top_n = int(request.args.get('top', 5))
    users_ref = db.collection('users')
    users_snap = users_ref.stream()
    users = []
    user_ids = []
    for doc in users_snap:
        data = doc.to_dict()
        data['uid'] = doc.id
        users.append(data)
        user_ids.append(doc.id)
    if len(users) < 2:
        return jsonify([])
    all_skills = set()
    all_interests = set()
    all_avail = set()
    for u in users:
        all_skills.update(u.get('skills', []))
        all_interests.update(u.get('interests', []))
        all_avail.update(flatten_availability(u))
    all_skills = sorted(list(all_skills))
    all_interests = sorted(list(all_interests))
    all_avail = sorted(list(all_avail))
    feature_matrix = []
    for u in users:
        skills_vec = [1 if s in u.get('skills', []) else 0 for s in all_skills]
        interests_vec = [1 if i in u.get('interests', []) else 0 for i in all_interests]
        user_avail = flatten_availability(u)
        avail_vec = [1 if slot in user_avail else 0 for slot in all_avail]
        big5_vec = [float(u.get(k, 0.5)) for k in BIG5_KEYS]
        feature_matrix.append(skills_vec + interests_vec + avail_vec + big5_vec)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    k = min(3, len(users))
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    if user_id not in user_ids:
        return jsonify({'error': 'User not found'}), 404
    idx = user_ids.index(user_id)
    user_cluster = clusters[idx]
    user_vec = X_scaled[idx].reshape(1, -1)
    cluster_indices = [i for i, c in enumerate(clusters) if c == user_cluster and i != idx]
    if not cluster_indices:
        return jsonify([])
    cluster_vectors = X_scaled[cluster_indices]
    similarities = cosine_similarity(user_vec, cluster_vectors)[0]
    similarity_pairs = [(similarities[i], cluster_indices[i]) for i in range(len(similarities))]
    similarity_pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = similarity_pairs[:top_n]
    result = [
        {
            "uid": users[user_idx]['uid'],
            "similarity": round(((float(sim) + 1) / 2) * 100, 2)
        }
        for sim, user_idx in top_pairs
    ]
    return jsonify(result)

@app.route('/user/<uid>', methods=['GET'])
def get_user(uid):
    doc = db.collection('users').document(uid).get()
    if not doc.exists:
        return jsonify({'error': 'User not found'}), 404
    data = doc.to_dict()
    if not data:
        return jsonify({'error': 'User data is empty'}), 404
    data['uid'] = doc.id
    return jsonify(data)

# --- FCM Notification Utility ---
def send_fcm_notification(token, title, body):
    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"key={FCM_SERVER_KEY}"
    }
    payload = {
        "to": token,
        "notification": {
            "title": title,
            "body": body
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"FCM Error: {response.text}")
        return {"error": str(e), "details": response.text}

# --- OneSignal Notification Utility ---
def send_onesignal_notification(player_id, title, body):
    url = "https://onesignal.com/api/v1/notifications"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {ONESIGNAL_API_KEY}"
    }
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "include_player_ids": [player_id],
        "headings": {"en": title},
        "contents": {"en": body}
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"OneSignal Error: {response.text}")
        return {"error": str(e), "details": response.text}

# --- Flask Endpoint for FCM ---
@app.route('/send-fcm', methods=['POST'])
def send_fcm_route():
    data = request.json
    token = data.get('token')
    title = data.get('title')
    body = data.get('body')
    if not all([token, title, body]):
        return jsonify({"error": "Missing required fields"}), 400
    result = send_fcm_notification(token, title, body)
    return jsonify(result)

# --- Flask Endpoint for OneSignal ---
@app.route('/send-onesignal', methods=['POST'])
def send_onesignal_route():
    data = request.json
    player_id = data.get('player_id')
    title = data.get('title')
    body = data.get('body')
    if not all([player_id, title, body]):
        return jsonify({"error": "Missing required fields"}), 400
    result = send_onesignal_notification(player_id, title, body)
    return jsonify(result)

@app.route('/save-fcm-token', methods=['POST'])
def save_fcm_token():
    data = request.json
    user_id = data.get('user_id')
    fcm_token = data.get('fcm_token')
    if not user_id or not fcm_token:
        return jsonify({'error': 'Missing user_id or fcm_token'}), 400
    # Save the token in Firestore under the user's document
    db.collection('users').document(user_id).update({'fcm_token': fcm_token})
    return jsonify({'message': 'Token saved'}), 200

@app.route('/notify-user', methods=['POST'])
def notify_user():
    data = request.json
    user_id = data.get('user_id')
    title = data.get('title', 'Notification')
    message = data.get('message', '')
    if not user_id or not message:
        return jsonify({'error': 'Missing user_id or message'}), 400

    # Fetch FCM token from Firestore
    doc = db.collection('users').document(user_id).get()
    if not doc.exists:
        return jsonify({'error': 'User not found'}), 404
    user_data = doc.to_dict()
    fcm_token = user_data.get('fcm_token')
    if not fcm_token:
        return jsonify({'error': 'User has no FCM token'}), 404

    # Use FCM token as OneSignal external user ID
    url = "https://onesignal.com/api/v1/notifications"
    headers = {
        "Authorization": f"Basic {ONESIGNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "include_external_user_ids": [fcm_token],
        "headings": {"en": title},
        "contents": {"en": message}
    }
    response = requests.post(url, json=payload, headers=headers)
    return jsonify(response.json()), response.status_code

# --- Utility: Notify New Connection ---
def notify_new_connection(user_id, other_user_name):
    payload = {
        "user_id": user_id,
        "title": "New Connection!",
        "message": f"You are now connected with {other_user_name}."
    }
    try:
        response = requests.post("http://localhost:5000/notify-user", json=payload)
        print(f"Notify new connection response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error notifying new connection: {e}")

# --- Utility: Notify Group Invitation ---
def notify_group_invitation(user_id, group_name):
    payload = {
        "user_id": user_id,
        "title": "Group Invitation",
        "message": f"You have been invited to join the group: {group_name}."
    }
    try:
        response = requests.post("http://localhost:5000/notify-user", json=payload)
        print(f"Notify group invitation response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error notifying group invitation: {e}")

# --- Utility: Notify New Message ---
def notify_new_message(user_id, sender_name, message_preview, chat_id=None):
    payload = {
        "user_id": user_id,
        "title": f"New message from {sender_name}",
        "message": message_preview
    }
    if chat_id:
        payload["data"] = {"type": "chat", "chat_id": chat_id}
    try:
        response = requests.post("http://localhost:5000/notify-user", json=payload)
        print(f"Notify new message response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error notifying new message: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)