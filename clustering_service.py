import os
import json
import traceback
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
import requests
from dotenv import load_dotenv

# Load env variables
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

    if len(users) < 2:
        return jsonify([])

    all_skills, all_interests, all_avail = set(), set(), set()

    for u in users:
        all_skills.update(u.get('skills', []))
        all_interests.update(u.get('interests', []))
        all_avail.update(flatten_availability(u))

    all_skills, all_interests, all_avail = sorted(all_skills), sorted(all_interests), sorted(all_avail)

    feature_matrix = []
    for u in users:
        skills_vec = [1 if s in u.get('skills', []) else 0 for s in all_skills]
        interests_vec = [1 if i in u.get('interests', []) else 0 for i in all_interests]
        avail_vec = [1 if slot in flatten_availability(u) else 0 for slot in all_avail]
        big5_vec = [float(u.get(k, 0.5)) for k in BIG5_KEYS]
        feature_matrix.append(skills_vec + interests_vec + avail_vec + big5_vec)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    inertias = []
    max_k = min(len(users), 10)
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    kneedle = KneeLocator(range(1, max_k + 1), inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow or 3

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
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

    similarity_pairs = sorted(
        [(similarities[i], cluster_indices[i]) for i in range(len(similarities))],
        key=lambda x: x[0], reverse=True
    )
    top_pairs = similarity_pairs[:top_n]

    result = [
        {
            "uid": users[user_idx]['uid'],
            "similarity": round(((float(sim) + 1) / 2) * 100, 2)
        }
        for sim, user_idx in top_pairs
    ]
    return jsonify(result)

@app.route('/user/<user_id>', methods=['GET'])
def get_user(user_id):
    try:
        doc = db.collection('users').document(user_id).get()
        if not doc.exists:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
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
