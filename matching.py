from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess
from faqs import faqs

preprocessed_questions = [preprocess(faq['question']) for faq in faqs]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

def get_best_match(user_question):
    user_question_prep = preprocess(user_question)
    user_vec = vectorizer.transform([user_question_prep])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_idx = similarities.argmax()
    return faqs[best_idx]['answer']
