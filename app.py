import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import os

# Page config
st.set_page_config(page_title="RAG PDF Chat", page_icon="📚")
st.title("Le chat 🐈")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Paramètres")

    # Model selection
    model_name = st.selectbox(
        "Model IA",
        ["mistral", "llama2", "llama3", "phi"],
        index=0
    )

    st.header("PDF")
    uploaded_files = st.file_uploader(
        "Choix PDF",
        type="pdf",
        accept_multiple_files=True

    )

    process_button = st.button("Traiter PDFs")

# Initialize session state
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

            # Create embeddings and vectorstore
            embeddings = OllamaEmbeddings(model=model_name)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            st.sidebar.success(f"{len(uploaded_files)} PDF(s) traités avec {len(splits)} chunks!")

        except Exception as e:
            st.sidebar.error(f"Error traitement PDFs: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez une question sur vos PDFs"):
    if st.session_state.vectorstore is None:
        st.warning("Veuillez déposer des PDFs avant de poser une question.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Reflexion..."):
                try:
                    # Create RAG chain
                    llm = Ollama(model=model_name, temperature=0)

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
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

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