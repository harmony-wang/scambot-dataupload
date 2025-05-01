import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file.")
    st.stop()

# Set API key for LangChain/OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Constants
CONTEXT_FOLDER = "BaseContext"
VECTORSTORE_DIR = "basevector"

# UI
st.title("Internal: Upload or Paste Materials for the ISSS Scambot")

st.markdown("""
Welcome! Please upload **.txt or .html files** OR fill in the form below to add materials directly to the chatbot's knowledge base.

- Files will be processed and saved into the system's knowledge base.
- If you're pasting content, include a **title**, an optional **URL**, and the **material content**.
- For .txt files, only UTF-8 plain text is supported.

""")

st.markdown("<h3 style='font-size:1.1rem; margin-top:2rem;'>A. Please upload or submit only relevant, verified information in .txt or .html file</h3>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a .txt or .html file", type=["txt", "html"])

if uploaded_file and st.button("Ingest File"):
    file_path = os.path.join(CONTEXT_FOLDER, uploaded_file.name)

    if os.path.exists(file_path):
        st.error(f"❌ A file named '{uploaded_file.name}' already exists. Please rename your file.")
    else:
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and embed
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata["source"] = uploaded_file.name

            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(raw_docs)

            # Embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            # Load existing vectorstore or create new
            if os.path.exists(VECTORSTORE_DIR):
                vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(docs)
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)

            # Save vectorstore
            vectorstore.save_local(VECTORSTORE_DIR)
            st.success(f"✅ '{uploaded_file.name}' successfully added to vector store.")
        except Exception as e:
            st.error(f"❌ Ingestion failed: {e}")

st.markdown("<h3 style='font-size:1.1rem; margin-top:2rem;'>B. Paste Materials Instead</h3>", unsafe_allow_html=True)

with st.form("paste_material_form"):
    title = st.text_input("Title (required)")
    url = st.text_input("URL (optional)")
    pasted_material = st.text_area("Material Content (required)", height=200)
    submit_paste = st.form_submit_button("Submit Text")

if submit_paste:
    if not title.strip() or not pasted_material.strip():
        st.error("❌ Title and material content are required.")
    else:
        # Save the content as a new text file in BaseContext
        safe_filename = f"{title.strip().replace(' ', '_')}.txt"
        file_path = os.path.join(CONTEXT_FOLDER, safe_filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n{url}\n\n{pasted_material}")

        # Load and embed
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata["source"] = url if url else title

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(raw_docs)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            if os.path.exists(VECTORSTORE_DIR):
                vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(docs)
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)

            vectorstore.save_local(VECTORSTORE_DIR)
            st.success(f"✅ Text successfully added to vector store as '{safe_filename}'.")
        except Exception as e:
            st.error(f"❌ Failed to ingest pasted content: {e}")

st.markdown("---")
st.subheader("Existing Uploaded Files")

if not os.path.exists(CONTEXT_FOLDER):
    st.warning("The BaseContext folder does not exist.")

else:
    files = [f for f in os.listdir(CONTEXT_FOLDER) if f.endswith(".txt")]
    if not files:
        st.info("No files have been uploaded yet.")
    else:
        for filename in sorted(files):
            file_path = os.path.join(CONTEXT_FOLDER, filename)
            with st.expander(f"📄 {filename}"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Display content with Arial font
                st.markdown("**File Content:**")
                st.markdown(
                    f"<div style='white-space: pre-wrap; font-family: Arial, sans-serif; font-size: 14px;'>{content}</div>",
                    unsafe_allow_html=True
                )

                # Two buttons aligned left with minimal spacing
                col1, col2, _ = st.columns([2.5, 2.5, 8])
                with col1:
                    st.download_button("Download", data=content, file_name=filename)
                with col2:
                    if st.button("Delete", key=f"delete_{filename}"):
                        os.remove(file_path)
                        st.success(f"'{filename}' has been deleted. Please refresh to update the list.")
                        st.experimental_rerun()