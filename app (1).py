from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile
import os

# 🌿 Load environment variables
load_dotenv()

# 🌿 Streamlit Page Config
st.set_page_config(page_title="🧠 Mental Health Chatbot", layout="wide")

# 🧩 Tabs
tabs = st.tabs(["📘 About", "💬 Mental Health Chatbot"])

# 📘 About Tab
with tabs[0]:
    st.header("💡 About")
    st.write(
        """
        ⿡ This is a *Mental Health Chatbot* application built using *Streamlit*.  
        ⿢ It summarizes the user's input and provides *supportive responses*.  
        ⿣ Simplifies *complex mental health concerns* into easy-to-understand terms.  
        ⿤ Offers *empathetic and supportive feedback* to users.  
        ⿥ Also provides assistance for *other medical conditions*.
        """
    )

# 💬 Chatbot Tab
with tabs[1]:
    st.header("🩺 Medical Chatbot: Your Mental Health Companion")

    # File uploader
    uploaded_file = st.file_uploader("📄 Upload a PDF file:", type=["pdf"])

    # Text area for user input
    user_question = st.text_area("💭 Enter your medical concerns:")

    if st.button("🤖 Get Response"):
        if uploaded_file is not None and user_question.strip():
            try:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Load and process PDF
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()

                # Split PDF text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                docs = text_splitter.split_documents(pages)
                combined_text = " ".join([doc.page_content for doc in docs])

                st.success("✅ File uploaded and processed successfully!")

                # Prompt for model
                prompt = PromptTemplate(
                    input_variables=["user_question", "pdf_content"],
                    template="""
                    You are a helpful and empathetic medical assistant.
                    - Summarize the uploaded medical PDF content: {pdf_content}.
                    - Respond to the user's medical concern: {user_question}.
                    - Explain issues in *simpler terms* with examples.
                    - Provide *empathetic and supportive* responses.
                    - Focus only on *medical and mental health* related topics.
                    """
                )

                # Model and parser
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                parser = StrOutputParser()

                # Chain
                chain = prompt | model | parser
                response = chain.invoke({
                    "user_question": user_question,
                    "pdf_content": combined_text[:5000]  # Limit context size for efficiency
                })

                # Display response
                st.subheader("🧠 Chatbot Response:")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Error: {e}")

            finally:
                # Remove temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        else:
            st.warning("⚠ Please upload a PDF and enter your question before generating a response.")
