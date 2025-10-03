import streamlit as st
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# Set up the local embedding model (must match the one used during ingestion)
HF_CACHE = "C:/Users/ethan/.cache/huggingface/hub"

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=HF_CACHE
)
Settings.embed_model = embed_model

# Use a powerful model optimized for GPU (RTX 4090)
llm = HuggingFaceLLM(
    model_name="microsoft/DialoGPT-medium",  # Much better conversational model
    tokenizer_name="microsoft/DialoGPT-medium",
    context_window=2048,   # Larger context for better responses
    max_new_tokens=512,   # Longer, more detailed responses
    generate_kwargs={
        "temperature": 0.7, 
        "do_sample": True,
        "top_p": 0.9,  # Better response quality
        "repetition_penalty": 1.1
    },
    device_map="auto"  # Automatically use GPU when available
)
Settings.llm = llm

# Cache the index loading to avoid reloading on every Streamlit rerun
@st.cache_resource
def get_index(persist_dir="storage"):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        st.error("Please run ingest.py first to create the vector database.")
        st.stop()

# Load the persisted index
try:
    index = get_index()
except Exception as e:
    st.error(f"Error loading index: {str(e)}")
    st.stop()

# Streamlit app title
st.title("Offline RAG Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the chat engine in session state (persists memory across interactions)
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question",  # Condenses question with chat history for better context
        verbose=True,
        similarity_top_k=3  # Retrieve top 3 relevant chunks
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about the company documents:"):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the chat engine (which handles memory internally)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                st.markdown(str(response))
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                response = "I'm sorry, I encountered an error while processing your question. Please try again."
                st.markdown(response)

    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})

# Optional: Button to reset chat
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.chat_engine.reset()  # Reset chat engine memory