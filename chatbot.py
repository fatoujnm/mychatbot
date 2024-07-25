import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st

# Assurez-vous que les ressources nécessaires sont téléchargées
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger le fichier texte et prétraiter les données
file_path = 'C:/Users/dell/Desktop/chatbot/15080-8.txt'

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        data = f.read().replace('\n', ' ')
    sentences = nltk.sent_tokenize(data)
    return sentences

sentences = load_and_preprocess_data(file_path)

# Définir une fonction pour prétraiter chaque phrase
def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in nltk.corpus.stopwords.words('english') and word not in string.punctuation]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Prétraiter chaque phrase dans le texte
corpus = [preprocess(sentence) for sentence in sentences]

# Définir une fonction pour trouver la phrase la plus pertinente donnée une requête
def get_most_relevant_sentence(query):
    query = preprocess(query)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus + [query])
    cosine_similarities = cosine_similarity(X[-1], X[:-1])
    most_relevant_index = cosine_similarities.argmax()
    most_relevant_sentence = sentences[most_relevant_index]
    return most_relevant_sentence

# Définir la fonction de chatbot
def chatbot(question):
    most_relevant_sentence = get_most_relevant_sentence(question)
    return most_relevant_sentence

# Créer une application Streamlit
def main():
    st.title("English Literature Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    
    # Zone de texte pour entrer une question
    question = st.text_input("You:")
    
    # Créer un bouton pour soumettre la question
    if st.button("Submit"):
        if question:
            response = chatbot(question)
            st.write("Chatbot: " + response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()


