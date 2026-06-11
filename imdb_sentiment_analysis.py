import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("IMDB Dataset.csv")

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

duplicates_before = df.shape[0]

df.drop_duplicates(inplace=True)

duplicates_after = df.shape[0]

print(f"Rows Before Removing Duplicates: {duplicates_before}")
print(f"Rows After Removing Duplicates : {duplicates_after}")

def clean_text(text):

    
    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()

    return text

print("\nCleaning Review Text...")

df['review'] = df['review'].apply(clean_text)

print("Text Cleaning Completed.")


print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df['review_length'] = df['review'].apply(len)

df['word_count'] = df['review'].apply(
    lambda x: len(x.split())
)

df['avg_word_length'] = df.apply(
    lambda row:
    sum(len(word) for word in row['review'].split())
    /
    max(len(row['review'].split()), 1),
    axis=1
)

df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

print("\n" + "=" * 60)
print("OUTLIER DETECTION & TREATMENT")
print("=" * 60)

numeric_features = [
    'review_length',
    'word_count',
    'avg_word_length'
]

for col in numeric_features:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"\n{col}")
    print(f"Outliers Found: {len(outliers)}")

    df[col] = np.where(
        df[col] < lower_bound,
        lower_bound,
        df[col]
    )

    df[col] = np.where(
        df[col] > upper_bound,
        upper_bound,
        df[col]
    )

print("\nOutlier Treatment Completed Successfully.")
print("\nFeature Engineering Completed.")

print("\nNew Features Created:")
print("- review_length")
print("- word_count")
print("- avg_word_length")

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['review_length'], bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Review Length")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    x='sentiment',
    y='review_length',
    data=df
)
plt.title("Review Length vs Sentiment")
plt.show()

plt.figure(figsize=(8, 6))

corr = df[
    [
        'review_length',
        'word_count',
        'avg_word_length',
        'sentiment'
    ]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")
plt.show()

print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X = tfidf.fit_transform(df['review'])

y = df['sentiment']

print("TF-IDF Shape:", X.shape)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model Training Completed.")

print("\nMaking Predictions...")

y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


print("\n" + "=" * 60)
print("TOP POSITIVE & NEGATIVE WORDS")
print("=" * 60)

feature_names = tfidf.get_feature_names_out()

coefficients = model.coef_[0]

top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

print("\nTop Positive Words:")

for idx in reversed(top_positive):
    print(feature_names[idx])

print("\nTop Negative Words:")

for idx in top_negative:
    print(feature_names[idx])


print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print("""
1. The dataset contains both positive and negative reviews.

2. TF-IDF successfully converts textual reviews into numerical features.

3. Logistic Regression performs well for sentiment classification.

4. Review text contains strong sentiment indicators.

5. The model can be used by streaming platforms,
   movie websites, and entertainment companies
   to automatically analyze customer feedback.

6. This solution helps businesses understand
   audience satisfaction and improve decision making.
""")

print("\nPROJECT COMPLETED SUCCESSFULLY!")