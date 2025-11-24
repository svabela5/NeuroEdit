import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

# Configuration
SERVICE_ACCOUNT_FILE = "imgai-52f51-firebase-adminsdk-fbsvc-ecf0a2ce89.json"
MEMORY_FILE = "neuroedit_memory.json"
COLLECTION_NAME = "neuroedit_memory"

def migrate_data():
    # 1. Initialize Firebase
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"Error: Service account file '{SERVICE_ACCOUNT_FILE}' not found.")
        return

    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    try:
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully.")
    except ValueError:
        print("Firebase app already initialized.")

    db = firestore.client()

    # 2. Read Local Memory
    if not os.path.exists(MEMORY_FILE):
        print(f"Error: Memory file '{MEMORY_FILE}' not found.")
        return

    try:
        with open(MEMORY_FILE, 'r') as f:
            memory_data = json.load(f)
            print(f"Loaded {len(memory_data)} items from local memory.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON memory file.")
        return

    # 3. Push to Firestore
    count = 0
    batch = db.batch()
    
    print("Starting migration...")
    for item in memory_data:
        # Create a new document reference
        doc_ref = db.collection(COLLECTION_NAME).document()
        
        # Add timestamp if not present (using server timestamp is best for new data, 
        # but for migration we might want to just let it be or add current time)
        # The app uses server timestamp, so let's add it here too for consistency if possible,
        # but we can't easily batch server timestamps in the same way without careful handling.
        # For simplicity in migration, we'll just push the data as is + a local timestamp or 
        # let Firestore handle it if we were adding one by one.
        # Actually, the app adds `timestamp: firestore.SERVER_TIMESTAMP`.
        # We can do that here too.
        
        item_data = item.copy()
        item_data['timestamp'] = firestore.SERVER_TIMESTAMP
        
        batch.set(doc_ref, item_data)
        count += 1
        
        # Commit batches of 500 (Firestore limit)
        if count % 500 == 0:
            batch.commit()
            batch = db.batch()
            print(f"Committed {count} items...")

    # Commit remaining
    if count % 500 != 0:
        batch.commit()
        
    print(f"Migration complete! Successfully pushed {count} items to Firestore collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    migrate_data()
