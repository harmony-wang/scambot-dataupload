import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file.")
    st.stop()

# Set API key for LangChain/OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Google Sheets logging
def get_gsheet_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client

def log_interaction(session_id, user_input, bot_response):
    try:
        gc = get_gsheet_client()
        worksheet = gc.open("ISSS_Responses").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([timestamp, session_id, user_input, bot_response])
    except Exception as e:
        st.warning(f"⚠️ Failed to log interaction: {e}")

# Session setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [AIMessage(content="Hello! I'm here to help you determine if a call, email, or other message you've received might be a scam. Just share a brief summary of the situation, and I'll analyze it for common scam indicators like suspicious requests, too-good-to-be-true offers, or unusual urgency. Remember, it's always better to be cautious! Let’s get started—describe what happened, and I'll guide you through it")]

# Page config
st.set_page_config(page_title="Scam Safety Chatbot", page_icon="🛡️", layout="centered")

# Styling
st.markdown("""
    <style>
    body, html, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f4f5;
    }
    html, body {
        margin: 0;
        padding: 0;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .title h1 {
        margin-top: 0.25rem;
        font-size: 1.5em;
        font-weight: 700;
        text-align: center;
    }
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding-bottom: 5rem;
    }
    .message-bubble {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 0.85rem;
    }
    .user-bubble {
        background-color: #dcfce7;
        color: #065f46;
        font-size: 0.85rem;
    }
    .bot-bubble {
        background-color: #e0f2fe;
        color: #075985;
        font-size: 0.85rem;
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 1rem;
        width: 80%;
        left: 10%;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown("""
<div class='title'>
    <h1>Scam Safety Chatbot</h1>
    <p style="text-align: center; font-size: 0.95rem; color: #4B5563; margin-top: 0.5rem;">
        The ISSS Scam Bot is designed to assist you in determining if a situation you experienced—such as a call, email, or offer—is likely a scam. You can input a summary of the interaction, and the bot will evaluate it based on key scam indicators and give an informed opinion on whether or not it appears to be a scam.
    </p>
</div>
""", unsafe_allow_html=True)

# Load RAG system
try:
    prompt_path = os.path.join(os.path.dirname(__file__), "Chatbot_Instructions.md")
    vectorstore_path = os.path.join(os.path.dirname(__file__), "basevector")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    with open(prompt_path, "r") as f:
        system_prompt = f.read()

    llm = ChatOpenAI(model="gpt-4o")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    st.session_state.retriever = retriever

    history_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    rag_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    st.session_state.rag_chain = rag_chain

except Exception as e:
    st.error(f"❌ Failed to set up chatbot: {e}")
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    alignment = "flex-end" if role == "user" else "flex-start"

    st.markdown(f"""
    <div style='display: flex; justify-content: {alignment};'>
        <div class='message-bubble {bubble_class}'>
            {msg.content}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Handle input
user_input = st.chat_input("Please enter your question")
if "user_input" in st.session_state and not user_input:
    user_input = st.session_state.pop("user_input")

if user_input:
    st.markdown(f"""
    <div style='display: flex; justify-content: flex-end;'>
        <div class='message-bubble user-bubble'>
            {user_input}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Loading..."):
        try:
            retrieved_docs_with_scores = vector_store.similarity_search_with_score(user_input, k=5)
            response = st.session_state.rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]

            log_interaction(st.session_state.session_id, user_input, answer)
            st.session_state.chat_history.append(AIMessage(content=answer))

            high_score_sources = [
                doc.metadata.get("source")
                for doc, score in retrieved_docs_with_scores
                if score < 0.3 and doc.metadata.get("source")
            ]

            if high_score_sources:
                source_links = "\n\n**Sources:**\n" + "\n".join(f"- [Link]({src})" for src in set(high_score_sources))
                answer += source_links

            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start;'>
                <div class='message-bubble bot-bubble'>
                    {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating answer: {e}")
