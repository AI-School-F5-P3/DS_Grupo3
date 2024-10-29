import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import timedelta, datetime

class OptimizedFirebase:
    def __init__(self):
        self.db = None
        self._init_firebase()
        self._cache = {}
        
    def _init_firebase(self):
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'projectId': os.getenv('FIREBASE_PROJECT_ID'),
            })
        self.db = firestore.client()
    
    def save_prediction(self, data):
        # Implementar rate limiting
        collection = self.db.collection('predictions')
        daily_count = len(list(collection.where(
            'timestamp', '>', 
            datetime.now() - timedelta(days=1)
        ).get()))
        
        if daily_count < 19000:  # Buffer para el límite de 20K
            collection.add(data)
        else:
            raise Exception("Daily write limit approached")