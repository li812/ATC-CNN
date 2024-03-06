import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load

df = pd.read_csv('dataset/ug_pg.csv')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['UG'])

# Train a k-NN model
model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(X)

# Save the model and vectorizer
dump(model, 'KNN_Model/model.joblib')
dump(vectorizer, 'KNN_Model/vectorizer.joblib')