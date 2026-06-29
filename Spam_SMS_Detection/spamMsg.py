import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv")
df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test New SMS
msg = ["Congratulations! You have won Rs.10000. Call now."]
msg = tfidf.transform(msg)

result = model.predict(msg)

if result[0] == 1:
    print("Spam")
else:
    print("Ham")