import os
import json
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    with open('firebase_creds.json', 'w') as f:
        f.write(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred_path = 'firebase_creds.json'
else:
    cred_path = 'matche-39f37-firebase-adminsdk-fbsvc-50793ed379.json'
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Helper: Convert Firestore user doc to feature vector
BIG5_KEYS = ['big5_O', 'big5_C', 'big5_E', 'big5_A', 'big5_N']

def flatten_availability(user):
    """
    Convert user availability data to a flat list of time slots.
    Assumes availability is stored as a dict with days as keys and time slots as values.
    
    Example input: {'monday': ['9:00-10:00', '14:00-15:00'], 'tuesday': ['10:00-11:00']}
    Example output: ['monday_9:00-10:00', 'monday_14:00-15:00', 'tuesday_10:00-11:00']
    """
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
        # If availability is already a flat list
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
    
    # Build all unique skills, interests, and availability slots
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
    
    # Build feature vectors
    feature_matrix = []
    for u in users:
        # Skills one-hot
        skills_vec = [1 if s in u.get('skills', []) else 0 for s in all_skills]
        # Interests one-hot
        interests_vec = [1 if i in u.get('interests', []) else 0 for i in all_interests]
        # Availability one-hot (flattened)
        user_avail = flatten_availability(u)
        avail_vec = [1 if slot in user_avail else 0 for slot in all_avail]
        # Big5
        big5_vec = [float(u.get(k, 0.5)) for k in BIG5_KEYS]
        # Combine all
        feature_matrix.append(skills_vec + interests_vec + avail_vec + big5_vec)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    
    # KMeans clustering
    k = min(3, len(users))
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Find target user
    if user_id not in user_ids:
        return jsonify({'error': 'User not found'}), 404
    
    idx = user_ids.index(user_id)
    user_cluster = clusters[idx]
    user_vec = X_scaled[idx].reshape(1, -1)
    
    # Find users in same cluster (excluding self)
    cluster_indices = [i for i, c in enumerate(clusters) if c == user_cluster and i != idx]
    
    if not cluster_indices:
        return jsonify([])
    
    cluster_vectors = X_scaled[cluster_indices]
    similarities = cosine_similarity(user_vec, cluster_vectors)[0]
    
    # Create pairs of (similarity, cluster_index) and sort by similarity descending
    similarity_pairs = [(similarities[i], cluster_indices[i]) for i in range(len(similarities))]
    similarity_pairs.sort(key=lambda x: x[0], reverse=True)  # Sort by similarity descending
    
    # Take top N results
    top_pairs = similarity_pairs[:top_n]
    
    # Return uid and similarity percentage in descending order, rescaled to [0, 100]
    result = [
        {
            "uid": users[user_idx]['uid'],
            "similarity": round(((float(sim) + 1) / 2) * 100, 2)
        }
        for sim, user_idx in top_pairs
    ]
    
    return jsonify(result)

# New endpoint to get user details by UID
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)