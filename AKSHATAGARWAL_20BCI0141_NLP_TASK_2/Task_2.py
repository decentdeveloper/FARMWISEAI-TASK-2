import json
import pandas as pd
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load training data
with open('Reddit_data_train.json', 'r') as f:
    td = json.load(f)

# Convert training data to a list of dictionaries
train_data_list = [{'user': user, 'posts': [post['text'] for post in posts]} for user, posts in td.items()]

# Create DataFrame for training data
traindf = pd.DataFrame(train_data_list)

# Tokenize training data
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
traindf['tokenized_posts'] = traindf['posts'].apply(lambda x: tokenizer.encode(x, return_tensors="pt")[0])

# Train LDA model for topic modeling
vectorizer = CountVectorizer(stop_words='english')
# Flatten lists of posts into a single string for each user
traindf['flattened_posts'] = traindf['posts'].apply(lambda x: ' '.join(x))
X = vectorizer.fit_transform(traindf['flattened_posts'])

lda = LatentDirichletAllocation(n_components=10, random_state=42)
traindf['topic'] = lda.fit_transform(X).argmax(axis=1)

# User Profile
prof = traindf.groupby('user')['topic'].apply(lambda x: x.value_counts().idxmax()).reset_index()
prof.columns = ['user', 'preferred_topic']

# Load testing data
with open('Reddit_data_test.json', 'r') as f:
    test_data = json.load(f)

# Convert testing data to a list of dictionaries
tdl = [{'post': user_entry['text']} for user_entry in test_data]

# Create DataFrame for testing data
test_df = pd.DataFrame(tdl)

# Tokenize testing data
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
test_df['tokenized_post'] = test_df['post'].apply(lambda x: tokenizer.encode(x, return_tensors="pt")[0])


# Recommendation Engine
def recommend_post(user):
    user_entry = prof[prof['user'] == user]

    if user_entry.empty:
        print(f"No profile found for user {user}. Skipping recommendation.")
        return []

    user_topic = user_entry['preferred_topic'].values[0]
    relevant_posts = traindf[traindf['topic'] == user_topic]['posts']

    if relevant_posts.empty:
        print(f"No relevant posts found for user {user}. Skipping recommendation.")
        return []

    sample_size = min(5, len(relevant_posts))
    return relevant_posts.sample(sample_size).tolist()


# Example usage
for entry in test_data:
    user_to_recommend = entry['GT'][0]['user']
    recom = recommend_post(user_to_recommend)

    print(f"Recommended posts for {user_to_recommend}:\n", recom)
