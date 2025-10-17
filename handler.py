import json
import os
import logging
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_handler.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables to cache the models and index
embed_model = None
llm = None
index = None

def initialize_models():
    """Initialize the embedding model and LLM. Called once when the handler starts."""
    global embed_model, llm
    
    logger.info("🚀 Starting model initialization...")
    
    # Use the same embedding model and cache path as used during ingestion
    HF_CACHE = "C:/Users/ethan/.cache/huggingface/hub"
    logger.info(f"📦 Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"📁 Using cache folder: {HF_CACHE}")
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=HF_CACHE
    )
    Settings.embed_model = embed_model
    logger.info("✅ Embedding model initialized successfully")
    
    # Use a reliable causal LM model that works well with RAG
    logger.info("🤖 Initializing LLM: microsoft/DialoGPT-medium")
    logger.info("⚙️ Model configuration:")
    logger.info("   - Context window: 1024")
    logger.info("   - Max new tokens: 128")
    logger.info("   - Temperature: 0.7")
    logger.info("   - Device map: cpu (stable)")
    
    llm = HuggingFaceLLM(
        model_name="microsoft/DialoGPT-medium",  # Reliable, smaller model
        tokenizer_name="microsoft/DialoGPT-medium",
        context_window=1024,  # Smaller context to avoid issues
        max_new_tokens=128,  # Shorter responses to stay within limits
        generate_kwargs={
            "temperature": 0.7,  # Balanced creativity
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        device_map="cpu",  # Stable CPU execution
        model_kwargs={
            "torch_dtype": "auto",
            "trust_remote_code": True
        }
    )
    Settings.llm = llm
    logger.info("✅ LLM initialized successfully")
    logger.info("🎉 All models loaded and ready!")

def load_vector_index(storage_dir="storage"):
    """Load the existing vector index from storage."""
    global index
    
    logger.info(f"📚 Loading vector index from: {storage_dir}")
    
    try:
        if not os.path.exists(storage_dir):
            error_msg = f"Storage directory '{storage_dir}' not found. Please run ingest.py first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("🔍 Checking storage directory contents...")
        storage_files = os.listdir(storage_dir)
        logger.info(f"📁 Found {len(storage_files)} files in storage: {storage_files}")
        
        logger.info("🔄 Creating storage context...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        
        logger.info("📖 Loading vector index from storage...")
        index = load_index_from_storage(storage_context)
        
        logger.info(f"✅ Successfully loaded vector index from {storage_dir}")
        logger.info("🎯 Vector database ready for queries!")
        return index
        
    except Exception as e:
        logger.error(f"❌ Error loading vector index: {str(e)}")
        raise

def query_index(question, similarity_top_k=5):
    """Query the vector index and return the response."""
    if index is None:
        logger.error("❌ Vector index not loaded. Please initialize first.")
        raise RuntimeError("Index not loaded. Please initialize first.")
    
    logger.info(f"🤔 Processing question: '{question}'")
    logger.info(f"🔍 Retrieving top {similarity_top_k} similar documents...")
    
    try:
        # Create a query engine for better Q&A responses (no chat memory needed for API)
        logger.info("⚙️ Creating query engine...")
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            verbose=True,
            response_mode="compact"  # More focused responses
        )
        
        # Use a simple, short prompt to avoid context window issues
        formatted_question = question
        
        logger.info("🔍 Searching vector database for relevant documents...")
        logger.info("🤖 Generating response using LLM...")
        
        # Query the index
        response = query_engine.query(formatted_question)
        
        response_text = str(response)
        logger.info(f"✅ Generated response of {len(response_text)} characters")
        logger.info(f"📝 Response preview: {response_text[:200]}...")
        
        return response_text
        
    except Exception as e:
        logger.error(f"❌ Error querying index: {str(e)}")
        raise

def handler(event):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "question": "some user question"
        }
    }
    
    Returns:
    {
        "output": "answer text here"
    }
    """
    logger.info("🚀 Handler function called")
    logger.info(f"📥 Received event: {str(event)[:200]}...")
    
    try:
        # Initialize models on first run
        if embed_model is None or llm is None:
            logger.info("🔄 Models not initialized, starting initialization...")
            initialize_models()
        else:
            logger.info("✅ Models already initialized, skipping initialization")
        
        # Load index on first run
        if index is None:
            logger.info("🔄 Vector index not loaded, loading now...")
            load_vector_index()
        else:
            logger.info("✅ Vector index already loaded, skipping loading")
        
        # Parse the input
        if isinstance(event, str):
            logger.info("🔄 Parsing JSON string input...")
            event = json.loads(event)
        
        # Extract the question from the input
        if "input" not in event:
            logger.error("❌ Missing 'input' field in request")
            return {
                "error": "Missing 'input' field in request",
                "output": "Invalid request format. Expected: {\"input\": {\"question\": \"your question\"}}"
            }
        
        question = event["input"].get("question")
        if not question:
            logger.error("❌ Missing 'question' field in input")
            return {
                "error": "Missing 'question' field in input",
                "output": "Please provide a question in the 'question' field."
            }
        
        logger.info(f"🎯 Question extracted: '{question}'")
        
        # Query the index
        logger.info("🔍 Starting query processing...")
        answer = query_index(question)
        
        logger.info(f"✅ Query completed successfully!")
        logger.info(f"📊 Response statistics:")
        logger.info(f"   - Response length: {len(answer)} characters")
        logger.info(f"   - Response words: {len(answer.split())} words")
        
        result = {"output": answer}
        logger.info("🎉 Handler execution completed successfully!")
        
        return result
        
    except FileNotFoundError as e:
        error_msg = f"Vector database not found: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "output": "The vector database is not available. Please ensure ingest.py has been run to create the index."
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.exception("Full traceback:")
        return {
            "error": error_msg,
            "output": "I'm sorry, I encountered an error while processing your question. Please try again."
        }

# For local testing
if __name__ == "__main__":
    logger.info("🧪 Starting local test of handler...")
    
    # Test the handler with a sample question
    test_event = {
        "input": {
            "question": "What is the DuraFlex system?"
        }
    }
    
    logger.info("🚀 Running test query...")
    result = handler(test_event)
    
    logger.info("📊 Test Results:")
    logger.info(json.dumps(result, indent=2))
    
    print("\n" + "="*50)
    print("TEST COMPLETE - Check rag_handler.log for detailed logs")
    print("="*50)
