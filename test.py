from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from kneed import KneeLocator
from flask import request, jsonify, Flask
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = Flask(__name__)

# 🔐 Firebase Admin SDK Initialization
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    with open('firebase_creds.json', 'w') as f:
        f.write(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred_path = 'firebase_creds.json'
else:
    cred_path = 'matche-39f37-firebase-adminsdk-fbsvc-50793ed379.json'

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

BIG5_KEYS = ['big5_O', 'big5_C', 'big5_E', 'big5_A', 'big5_N']

# 🧠 Helper: Flatten availability into "Day_TimeSlot" format
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

# 🚀 Clustering endpoint
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

    # 🔍 Collect all unique features
    all_skills = set()
    all_interests = set()
    all_avail = set()

    for u in users:
        all_skills.update(u.get('skills', []))
        all_interests.update(u.get('interests', []))
        all_avail.update(flatten_availability(u))

    all_skills = sorted(all_skills)
    all_interests = sorted(all_interests)
    all_avail = sorted(all_avail)

    # 🎯 Construct feature matrix
    feature_matrix = []
    for u in users:
        skills_vec = [1 if s in u.get('skills', []) else 0 for s in all_skills]
        interests_vec = [1 if i in u.get('interests', []) else 0 for i in all_interests]
        avail_vec = [1 if slot in flatten_availability(u) else 0 for slot in all_avail]
        big5_vec = [float(u.get(k, 0.5)) for k in BIG5_KEYS]
        feature_matrix.append(skills_vec + interests_vec + avail_vec + big5_vec)

    # 📊 Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # 🔎 Find optimal K using Elbow Method
    inertias = []
    max_k = min(len(users), 10)
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    kneedle = KneeLocator(range(1, max_k + 1), inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow or 3

    # 🧠 Apply KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    if user_id not in user_ids:
        return jsonify({'error': 'User not found'}), 404

    idx = user_ids.index(user_id)
    user_cluster = clusters[idx]
    user_vec = X_scaled[idx].reshape(1, -1)

    # 🤝 Find similar users in same cluster
    cluster_indices = [i for i, c in enumerate(clusters) if c == user_cluster and i != idx]

    if not cluster_indices:
        return jsonify([])

    cluster_vectors = X_scaled[cluster_indices]
    similarities = cosine_similarity(user_vec, cluster_vectors)[0]

    # 🧮 Sort by similarity
    similarity_pairs = sorted(
        [(similarities[i], cluster_indices[i]) for i in range(len(similarities))],
        key=lambda x: x[0],
        reverse=True
    )

    top_pairs = similarity_pairs[:top_n]

    # 🎁 Build response
    result = [
        {
            "uid": users[user_idx]['uid'],
            "similarity": round(((float(sim) + 1) / 2) * 100, 2)  # Normalize to 0–100%
        }
        for sim, user_idx in top_pairs
    ]

    return jsonify(result)

# 🔧 Run locally
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)