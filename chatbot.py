import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st

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
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in nltk.corpus.stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on the corpus and the query
    X = vectorizer.fit_transform(corpus + [query])
    
    # Compute the cosine similarity between the query and each sentence in the text
    cosine_similarities = cosine_similarity(X[-1], X[:-1])
    
    # Find the most relevant sentence
    most_relevant_index = cosine_similarities.argmax()
    most_relevant_sentence = sentences[most_relevant_index]
    
    return most_relevant_sentence

# Define the chatbot function
def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence

# Create a Streamlit app
def main():
    st.title("English Literature Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()


