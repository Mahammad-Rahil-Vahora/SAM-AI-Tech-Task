import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv("sentiment.csv")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# Test with new sentence
sentence = input("Enter a sentence: ")
sentence = clean_text(sentence)
result = model.predict([sentence])
print("Predicted Sentiment:", result[0])