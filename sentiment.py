import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

df = pd.read_csv('D:\\Maanushree\\SIT\\UML Mini Project\\backend\\train.csv')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocess
df['cleaned_reviews'] = df['review'].apply(preprocess_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(review):
    return sia.polarity_scores(review)

df['sentiment_scores'] = df['cleaned_reviews'].apply(get_sentiment_scores)
df['compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

# Group by columns to calculate the average compound score for each drug-condition pair
drug_sentiment = df.groupby(['drugName', 'condition'])['compound'].mean().reset_index()

print(drug_sentiment.head()) 

# Standardize scores
scaler = StandardScaler()
drug_sentiment['compound_scaled'] = scaler.fit_transform(drug_sentiment[['compound']])

#K-Means 
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters if needed
drug_sentiment['cluster'] = kmeans.fit_predict(drug_sentiment[['compound_scaled']])

# Map clusters to sentiment labels
cluster_sentiments = {0: 'Good', 1: 'Average', 2: 'Bad'}
drug_sentiment['sentiment'] = drug_sentiment['cluster'].map(cluster_sentiments)

# Evaluate 
X_scaled = drug_sentiment[['compound_scaled']].values
silhouette = silhouette_score(X_scaled, drug_sentiment['cluster'])
calinski_harabasz = calinski_harabasz_score(X_scaled, drug_sentiment['cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, drug_sentiment['cluster'])

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Select only the specific columns 
output_df = drug_sentiment[['drugName', 'condition', 'sentiment']]

# Save the new dataframe 
output_path = 'D:\\Maanushree\\SIT\\UML Mini Project\\backend\\output_with_kmeans_sentiments.csv'
output_df.to_csv(output_path, index=False)

print(f"Sentiment analysis with K-Means clustering complete and saved to {output_path}.")
