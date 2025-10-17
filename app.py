import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="RAG PDF Chat", page_icon="📚")
st.title("Le chat 🐈")

# Initialize session state for conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    # Create default conversation
    default_id = "conv_1"
    st.session_state.conversations[default_id] = {
        "name": "Conversation 1",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_conversation_id = default_id
if "conversation_counter" not in st.session_state:
    st.session_state.conversation_counter = 1
# Shared vectorstore for all conversations
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Get current conversation
current_conv = st.session_state.conversations[st.session_state.current_conversation_id]

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Paramètres")

    # Model selection
    model_name = st.selectbox(
        "Model IA",
        ["mistral", "llama2", "llama3", "phi"],
        index=0
    )

    st.header("💬 Conversations")

    # New conversation button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ Nouvelle conversation", use_container_width=True):
            st.session_state.conversation_counter += 1
            new_id = f"conv_{st.session_state.conversation_counter}"
            st.session_state.conversations[new_id] = {
                "name": f"Conversation {st.session_state.conversation_counter}",
                "messages": [],
                "created_at": datetime.now()
            }
            st.session_state.current_conversation_id = new_id
            st.rerun()

    # List all conversations
    st.markdown("---")
    for conv_id, conv_data in st.session_state.conversations.items():
        is_current = conv_id == st.session_state.current_conversation_id

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"{'🟢 ' if is_current else ''}{conv_data['name']}",
                key=f"select_{conv_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                if not is_current:
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()

        with col2:
            # Delete button (only if not the last conversation)
            if len(st.session_state.conversations) > 1:
                if st.button("🗑️", key=f"delete_{conv_id}", use_container_width=True):
                    del st.session_state.conversations[conv_id]
                    # If we deleted the current conversation, switch to another one
                    if conv_id == st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.header("PDF")
    uploaded_files = st.file_uploader(
        "Choix PDF",
        type="pdf",
        accept_multiple_files=True
    )

    process_button = st.button("Traiter PDFs")

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Nettoyer conversation", type="secondary", use_container_width=True):
            current_conv["messages"] = []
            st.sidebar.success("Conversation nettoyée !")
            st.rerun()
    with col2:
        if st.button("🗑️ Supprimer PDFs", type="secondary", use_container_width=True):
            st.session_state.vectorstore = None
            st.sidebar.success("PDFs supprimés !")
            st.rerun()

# Initialize session state (for backward compatibility)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDFs
if process_button and uploaded_files:
    with st.spinner("Traitement PDFs..."):
        try:
            all_docs = []

            # Load PDFs
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)

                # Clean up temp file
                os.unlink(tmp_path)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)

            # Create embeddings and vectorstore (shared across all conversations)
            embeddings = OllamaEmbeddings(model=model_name)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            st.sidebar.success(f"{len(uploaded_files)} PDF(s) traités avec {len(splits)} chunks!")

        except Exception as e:
            st.sidebar.error(f"Error traitement PDFs: {str(e)}")

# Display chat messages
for message in current_conv["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez une question sur vos PDFs"):
    if st.session_state.vectorstore is None:
        st.warning("Veuillez déposer des PDFs avant de poser une question.")
    else:
        # Add user message
        current_conv["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Reflexion..."):
                try:
                    # Create RAG chain
                    llm = OllamaLLM(model=model_name, temperature=0)

                    template = """Tu es un assistant qui répond UNIQUEMENT en utilisant les informations fournies dans le contexte ci-dessous.
                    Tu NE DOIS PAS utiliser tes connaissances générales ou chercher des informations sur internet.

                    RÈGLES IMPORTANTES:
                    - Réponds UNIQUEMENT si la réponse se trouve explicitement dans le contexte fourni
                    - Si l'information n'est pas dans le contexte, réponds EXACTEMENT: "Je n'ai pas trouvé cette information dans les documents fournis. Un ticket sera créé pour traiter votre demande."
                    - Ne devine JAMAIS et n'invente JAMAIS d'informations
                    - Réponds TOUJOURS en français, quelle que soit la langue de la question

                    Contexte fourni par les PDFs: {context}

                    Question: {question}

                    Réponse:"""

                    prompt_template = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        chain_type_kwargs={"prompt": prompt_template}
                    )

                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]

                    st.markdown(answer)
                    current_conv["messages"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error response: {str(e)}"
                    st.error(error_msg)
                    current_conv["messages"].append({"role": "assistant", "content": error_msg})

# Info section
with st.expander("ℹ️ Comment utiliser"):
    st.markdown("""
    1. **Install Ollama**: Download from https://ollama.ai
    2. **Pull a model**: Run `ollama pull llama2` in terminal
    3. **Upload PDFs**: Use the sidebar to upload one or more PDF files
    4. **Process**: Click "Process PDFs" to index the documents
    5. **Chat**: Ask questions about your PDFs in the chat interface

    **Requirements**:
    ```bash
    pip install streamlit langchain chromadb pypdf ollama langchain-community
    ```

    **Run**:
    ```bash
    streamlit run app.py
    ```
    """)