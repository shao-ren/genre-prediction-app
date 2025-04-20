import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

# Define the genres your model will predict
genres = [
    'acoustic', 'afrobeat', 'alternative', 'blues', 'children', 
    'chill', 'club', 'country', 'dance', 'disco', 'disney', 
    'edm', 'electro', 'emo', 'funk', 'groove', 'happy', 
    'house', 'jazz', 'pop'
]

# Define features the model will use based on the notebook
selected_features = [
    'energy', 'mode', 'key', 'valence', 'tempo',
    'emotion_joy', 'emotion_anger', 'emotion_fear', 'emotion_sadness', 'emotion_love', 'emotion_surprise',
    'joy_to_sadness_ratio', 'anger_to_love_ratio', 'energy_to_valence_ratio', 'surprise_to_fear_ratio',
    'energy_x_tempo', 'energy_x_valence', 'joy_x_tempo', 'sadness_x_valence',
    'energy_squared', 'tempo_log', 'valence_squared',
    'total_emotion_intensity', 'positive_emotions', 'negative_emotions', 'emotion_ratio', 'emotional_diversity',
    'dominant_joy', 'dominant_sadness', 'dominant_love'
]

# Create a mock model class
class MockModel:
    def __init__(self):
        self.classes_ = np.array(genres)
    
    def predict(self, X):
        # Always predict 'pop' for simplicity
        return np.array(['pop'])
    
    def predict_proba(self, X):
        # Generate probabilities with 'pop' having highest probability
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.ones((n_samples, n_classes)) * 0.02
        
        # Find the index for 'pop'
        pop_idx = np.where(self.classes_ == 'pop')[0][0]
        
        # Make pop more likely
        probas[:, pop_idx] = 0.6
        
        # Normalize
        row_sums = probas.sum(axis=1)
        probas = probas / row_sums[:, np.newaxis]
        
        return probas

# Create a mock scaler
class MockScaler:
    def transform(self, X):
        # Just return the input unchanged for simplicity
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

# Create model parameters dictionary
model_params = {
    'model': MockModel(),
    'encoder': None  # No encoder needed as we're using string labels directly
}

# Create a scaler
scaler = MockScaler()

# Save the files
print("Saving mock joblib files...")
joblib.dump(model_params, 'best_genre_prediction_model_params.joblib')
joblib.dump(scaler, 'genre_prediction_scaler.joblib')
joblib.dump(selected_features, 'selected_features.joblib')

print("Successfully created mock files!")
print("Files saved:")
print("- best_genre_prediction_model_params.joblib")
print("- genre_prediction_scaler.joblib")
print("- selected_features.joblib")

# Test to ensure files were created correctly
print("\nTesting the files...")
try:
    loaded_model_params = joblib.load('best_genre_prediction_model_params.joblib')
    loaded_scaler = joblib.load('genre_prediction_scaler.joblib')
    loaded_features = joblib.load('selected_features.joblib')
    
    print("All files loaded successfully!")
    print(f"Number of features: {len(loaded_features)}")
    print(f"Available genres: {loaded_model_params['model'].classes_}")
    
    # Test prediction
    dummy_data = np.random.random((1, len(selected_features)))
    prediction = loaded_model_params['model'].predict(dummy_data)
    probabilities = loaded_model_params['model'].predict_proba(dummy_data)
    
    print(f"\nTest prediction: {prediction[0]}")
    print(f"Confidence: {probabilities[0][np.where(loaded_model_params['model'].classes_ == prediction[0])[0][0]]:.2f}")
    
except Exception as e:
    print(f"Error testing files: {e}")