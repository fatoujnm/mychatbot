import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text file and preprocess the data
file_path = 'C:/Users/dell/Desktop/chatbot/15080-8.txt'
with open(file_path, 'r', encoding='latin-1') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in nltk.corpus.stopwords.words('english') and word not in string.punctuation]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    query = preprocess(query)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus + [query])
    cosine_similarities = cosine_similarity(X[-1], X[:-1])
    most_relevant_index = cosine_similarities.argmax()
    most_relevant_sentence = sentences[most_relevant_index]
    return most_relevant_sentence

# Test the chatbot function
query = "What is the main theme of the text?"
print(get_most_relevant_sentence(query))

