import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Create sample training data
data = {
    "message": [
        "Congratulations you won a free prize",
        "Claim your free gift now",
        "Win cash now",
        "You won a lottery",
        "Click here to claim your reward",
        "Get a free iPhone today",
        "Urgent claim your prize",
        "You have won free money",
        "Please send me the report",
        "Can we meet tomorrow",
        "Your meeting starts at 10 AM",
        "Thank you for your email",
        "Please call me when you are free",
        "Your appointment is confirmed",
        "The project meeting is tomorrow",
        "I will send the document today"
    ],
    "label": [
        "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

# Step 2: Separate messages and labels
X = df["message"]
y = df["label"]

# Step 3: Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 4: Train the model
model.fit(X, y)

# Step 5: Save the trained model
joblib.dump(model, "spam_detector_model.joblib")

print("Model trained successfully!")
print("Model saved successfully!")
