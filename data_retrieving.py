import firebase_admin
from firebase_admin import credentials, firestore
import csv
from datetime import datetime

# Step 1: Initialize Firebase Admin SDK
cred = credentials.Certificate('matche-39f37-firebase-adminsdk-fbsvc-50793ed379.json')  # 🔁 Replace this with your actual path
firebase_admin.initialize_app(cred)

# Step 2: Connect to Firestore
db = firestore.client()
users_ref = db.collection('users')
docs = users_ref.stream()

# Step 3: Set up desired CSV headers
fieldnames = [
    'name', 'email', 'skills',
    'availability_Monday', 'availability_Tuesday', 'availability_Wednesday',
    'availability_Thursday', 'availability_Friday', 'availability_Saturday', 'availability_Sunday',
    'interests', 'phone', 'gender', 'dob',
    'big5_O', 'big5_N', 'big5_A', 'big5_C', 'big5_E',
    'age', 'uid', 'userId'
]

# Step 4: Define ISO datetime formatter
def format_date(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ')
    return str(value)

# Step 5: Create the CSV file and write user data
with open('exported_users.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for doc in docs:
        data = doc.to_dict()
        row = {}

        # 📌 Basic fields
        for key in ['name', 'email', 'phone', 'gender', 'age', 'uid', 'userId']:
            row[key] = data.get(key, '')

        # 📌 Format 'dob' as full datetime string
        row['dob'] = format_date(data.get('dob', ''))

        # 📌 Convert list-type fields to readable strings
        for key in ['skills', 'interests']:
            val = data.get(key, [])
            row[key] = ', '.join(map(str, val)) if isinstance(val, list) else val

        # 📌 Handle nested availability — store as Python-style arrays
        availability = data.get('availability', {})
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            times = availability.get(day, [])
            row[f'availability_{day}'] = str(times if isinstance(times, list) else [times])

        # 📌 Handle nested Big Five traits
        big5 = data.get('big5', {})
        for trait in ['O', 'N', 'A', 'C', 'E']:
            row[f'big5_{trait}'] = big5.get(trait, '')

        writer.writerow(row)

print("✅ All user data exported successfully to 'exported_users.csv'")