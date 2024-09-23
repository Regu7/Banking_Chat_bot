import argparse
import os
import pickle
import random
import warnings
from dataclasses import dataclass

import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import pipeline

from utils.preprocess import finalpreprocess

warnings.filterwarnings("ignore")

abs_path = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(abs_path, "chroma")

PROMPT_TEMPLATE = """
Hello! As an AI developed by OpenAI, I'm serving as a banking assistant for Nova Bank. I'm here to provide accurate responses to your banking inquiries.

Relevant Information:
{context}

Now, let's address your question: {question}
"""


def sentiment_analysis(input):
    sentiment_classifier = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    sentiment_results = sentiment_classifier(input)
    return sentiment_results[0]["label"]


def intent_recognition(text):
    with open(
        os.path.join(abs_path, r"model_artifacts\class_label_dict.pkl"), "rb"
    ) as f:
        class_label_dict = pickle.load(f)

    question_pp = finalpreprocess(text)
    tfidf_vectorizer = pickle.load(
        open(os.path.join(abs_path, r"model_artifacts\tfidf1.pkl"), "rb")
    )
    question_vec = tfidf_vectorizer.transform([question_pp])
    lr_tfidf = pickle.load(
        open(os.path.join(abs_path, r"model_artifacts\logistic_reg.pkl"), "rb")
    )
    pred = lr_tfidf.predict(question_vec)
    return class_label_dict[pred[0]]


def out_of_context(text):
    greet_dict = {
        "patterns": [
            "hello",
            "hi!",
            "good morning!",
            "good evening",
            "good afternoon",
            "hey",
            "whats up",
            "is anyone there?",
            "hi there!",
            "greetings!",
            "good to see you!",
            "hi",
            "hii",
        ],
        "responses": [
            "Hello. Welcome to Nova Bank bank",
            "Hi! We are here to provide our service",
            "Welcome",
            "Hi there",
        ],
    }
    if text.lower() in greet_dict["patterns"]:
        return random.choice(greet_dict["responses"])
    else:
        return """Sorry, I'm not trained to answer this specific question.
                Please ask me a different question"""


def chat_response(user_query):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    chat = ChatOpenAI()
    results = db.similarity_search_with_relevance_scores(user_query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return out_of_context(user_query)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_query)
    print(prompt)

    conversation = ConversationChain(
        llm=chat,
        memory=ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048),
        verbose=False,
    )

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print("Sources :", sources)

    response_text = conversation.invoke(input=prompt)
    return response_text["response"]


def main():
    st.title("Nova Bank Chatbot")
    st.write("A banking assistant chatbot for Nova Bank.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("How can I help you?")
    if user_input:
        response = chat_response(user_input)
        # intent = intent_recognition(user_input)
        intent = "_"
        SA = sentiment_analysis(user_input)

        st.session_state.history.append((user_input, response, intent, SA))

    for user_input, response, intent, SA in st.session_state.history:
        st.write(f"You: {user_input}")
        st.write(f"Chatbot: {response}")
        # st.write(f"Intent: {intent}")
        # st.write(f"Sentiment Analysis: {SA}")

        # Display intent and sentiment in a small tag or dialog box
        st.markdown(
            f"""
            <div style="display: inline-block; padding: 5px; border: 1px solid #ccc; border-radius: 5px; margin-top: 5px;">
                <strong>Sentiment:</strong> {SA}
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
