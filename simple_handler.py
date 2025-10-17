import json
import os
import logging
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from sentence_transformers import CrossEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to cache the models and index
index = None
llm = None
reranker = None

def initialize_models():
    """Initialize embedding model, LLM, and reranker for hybrid approach."""
    global llm, reranker
    
    # Initialize embedding model (same as before)
    HF_CACHE = "C:/Users/ethan/.cache/huggingface/hub"
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=HF_CACHE
    )
    Settings.embed_model = embed_model
    logger.info("SUCCESS: Embedding model initialized")
    
    # Initialize gpt2 for more controllable responses
    logger.info("Initializing gpt2 for controllable natural responses...")
    llm = HuggingFaceLLM(
        model_name="gpt2",  # More controllable than distilgpt2
        tokenizer_name="gpt2",
        context_window=1024,  # Larger context for better responses
        max_new_tokens=300,  # Shorter to avoid hallucination
        generate_kwargs={
            "temperature": 0.2,  # Lower for more factual responses
            "do_sample": True,
            "top_p": 0.8,  # More focused
            "repetition_penalty": 1.2,
            "top_k": 40  # More controlled generation
        },
        device_map="cpu",  # Stable CPU execution
        model_kwargs={
            "torch_dtype": "auto",
            "trust_remote_code": True
        }
    )
    logger.info("SUCCESS: LLM initialized for natural responses")
    
    # Initialize cross-encoder reranker
    logger.info("Initializing cross-encoder reranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("SUCCESS: Reranker initialized")

def load_vector_index(storage_dir="storage"):
    """Load the existing vector index from storage."""
    global index
    
    logger.info(f"Loading vector index from: {storage_dir}")
    
    try:
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Storage directory '{storage_dir}' not found. Please run ingest.py first.")
        
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        logger.info("SUCCESS: Successfully loaded vector index")
        return index
        
    except Exception as e:
        logger.error(f"ERROR: Error loading vector index: {str(e)}")
        raise

def extract_and_clean_facts(nodes):
    """Extract and clean key facts from document nodes with aggressive filtering."""
    facts = []
    
    for node in nodes:
        # Clean up the text by removing metadata
        text = str(node.text)
        
        # Remove page numbers and headers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Light filtering - only remove obvious metadata
            if (line and 
                not line.startswith('Page') and 
                not line.startswith('©') and
                not line.startswith('Memjet Confidential') and
                not line.startswith('Modified on:') and
                not line.isdigit() and
                len(line) > 15):  # Shorter minimum length
                cleaned_lines.append(line)
        
        if cleaned_lines:
            # Take the first few relevant lines instead of just one
            clean_text = ' '.join(cleaned_lines[:3])  # Take up to 3 lines
            
            # Limit to reasonable length but not too short
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."
            
            facts.append({
                'text': clean_text,
                'score': node.score
            })
    
    return facts

def apply_reranker(question, nodes, top_k=5, similarity_threshold=0.5):
    """Apply cross-encoder reranker to get top-k most relevant chunks."""
    global reranker
    
    if reranker is None:
        logger.warning("Reranker not initialized, returning original nodes")
        return nodes[:top_k]
    
    logger.info("Applying cross-encoder reranker...")
    
    # Prepare pairs for reranking
    pairs = []
    for node in nodes:
        chunk_text = str(node.text)[:500]  # Limit text length for reranker
        pairs.append([question, chunk_text])
    
    # Get reranker scores
    try:
        rerank_scores = reranker.predict(pairs)
        
        # Combine nodes with their rerank scores
        scored_nodes = list(zip(nodes, rerank_scores))
        
        # Filter by similarity threshold and sort by rerank score
        filtered_nodes = [
            node for node, score in scored_nodes 
            if score > similarity_threshold
        ]
        
        # Sort by rerank score (descending) and take top-k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, score in scored_nodes[:top_k]]
        
        logger.info(f"Reranked {len(nodes)} chunks to {len(top_nodes)} high-quality chunks")
        return top_nodes
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}, returning original nodes")
        return nodes[:top_k]

def merge_adjacent_chunks(nodes):
    """Merge adjacent chunks from the same document."""
    if not nodes:
        return nodes
    
    merged_chunks = []
    current_chunk = None
    
    for node in nodes:
        # Get document metadata
        doc_id = getattr(node, 'doc_id', None) or str(hash(str(node.text)[:100]))
        
        if current_chunk is None:
            # Start new chunk
            current_chunk = {
                'text': str(node.text),
                'doc_id': doc_id,
                'score': node.score,
                'metadata': getattr(node, 'metadata', {})
            }
        elif current_chunk['doc_id'] == doc_id:
            # Same document, merge with current chunk
            current_chunk['text'] += '\n\n' + str(node.text)
            # Keep the higher score
            current_chunk['score'] = max(current_chunk['score'], node.score)
        else:
            # Different document, save current and start new
            merged_chunks.append(current_chunk)
            current_chunk = {
                'text': str(node.text),
                'doc_id': doc_id,
                'score': node.score,
                'metadata': getattr(node, 'metadata', {})
            }
    
    # Add the last chunk
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    # Convert back to node-like objects for compatibility
    merged_nodes = []
    for chunk in merged_chunks:
        # Create a simple object with the required attributes
        class MergedNode:
            def __init__(self, text, score, metadata):
                self.text = text
                self.score = score
                self.metadata = metadata
                self.doc_id = chunk['doc_id']
        
        merged_nodes.append(MergedNode(chunk['text'], chunk['score'], chunk['metadata']))
    
    logger.info(f"Merged {len(nodes)} chunks into {len(merged_nodes)} final chunks")
    return merged_nodes

def summarize_documents(nodes):
    """Summarize each document into 2-3 bullet points."""
    summarized_docs = []
    
    for i, node in enumerate(nodes, 1):
        # Clean the document text
        text = str(node.text)
        
        # Remove metadata and headers
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith('Page') and 
                not line.startswith('©') and
                not line.startswith('Memjet Confidential') and
                not line.startswith('Modified on:') and
                not line.isdigit() and
                len(line) > 20):
                clean_lines.append(line)
        
        if clean_lines:
            # Take the most relevant content (first 800 characters)
            content = ' '.join(clean_lines)[:800]
            
            # Create a summary prompt
            summary_prompt = f"""Summarize the following DuraFlex documentation excerpt into 2-3 key bullet points. Focus on the most important technical information:

{content}

Key points:"""
            
            # Skip LLM summarization to avoid hallucination - use direct extraction instead
            key_points = extract_key_points(content)
            summarized_docs.append({
                'summary': key_points,
                'score': node.score,
                'source': f"Document {i}"
            })
    
    return summarized_docs

def extract_key_points(content):
    """Extract key points directly from content without LLM."""
    # Split into sentences
    sentences = content.replace('!', '.').replace('?', '.').split('.')
    key_points = []
    
    # Look for sentences that contain relevant technical information
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip very short or very long sentences
        if len(sentence) < 30 or len(sentence) > 300:
            continue
            
        # Skip sentences that are just metadata
        if any(skip_word in sentence.lower() for skip_word in [
            'page', 'memjet confidential', 'modified on', 'figure', 'table'
        ]):
            continue
            
        # Prioritize sentences with technical specifications
        if any(tech_word in sentence.lower() for tech_word in [
            'intel', 'amd', 'memory', 'gb', 'ghz', 'cpu', 'processor', 'ram', 'storage',
            'windows', 'system', 'requirement', 'minimum', 'recommended', 'specification'
        ]):
            key_points.append(f"• {sentence}.")
            
        # Stop after finding 2-3 good points
        if len(key_points) >= 3:
            break
    
    # If no technical sentences found, take the first few reasonable sentences
    if not key_points:
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 30 and len(sentence) < 300:
                key_points.append(f"• {sentence}.")
    
    return '\n'.join(key_points) if key_points else f"• {content[:200]}..."

def merge_summaries(summarized_docs):
    """Merge document summaries into a single context block."""
    if not summarized_docs:
        return "No relevant information found."
    
    context_parts = []
    
    for doc in summarized_docs:
        context_parts.append(f"**{doc['source']}** (Relevance: {doc['score']:.2f})\n{doc['summary']}")
    
    # Add double newline between each document for better readability
    return '\n\n'.join(context_parts)

def generate_conversational_response(question, merged_context):
    """Generate a conversational, human-like response using LLM with strict grounding."""
    
    prompt = f"""Answer this question using ONLY the information provided in the documentation below. Do not add any information not present in the documents.

Question: {question}

Documentation:
{merged_context}

Answer using only the facts from the documentation above:"""

    try:
        logger.info("Generating conversational response with LLM...")
        response = llm.complete(prompt)
        response_text = str(response).strip()
        
        # Always use fallback for reliability - LLM consistently hallucinates
        logger.info("Using reliable fallback response to avoid hallucination")
        return generate_fallback_response(question, merged_context)
            
    except Exception as e:
        logger.error(f"Failed to generate conversational response: {e}")
        return generate_fallback_response(question, merged_context)

def validate_response(response, context):
    """Validate that the response is grounded in the context and not hallucinated."""
    # Check for common hallucination patterns
    hallucination_indicators = [
        "customers", "USB cable", "internet", "download", "purchase", 
        "computer setup", "software development", "glitches", "cost savings"
    ]
    
    response_lower = response.lower()
    context_lower = context.lower()
    
    # If response contains hallucination indicators not in context, reject it
    for indicator in hallucination_indicators:
        if indicator in response_lower and indicator not in context_lower:
            logger.warning(f"Detected hallucination indicator: {indicator}")
            return False
    
    # Check if response length is reasonable
    if len(response) > 2000:  # Too long, likely rambling
        return False
    
    return True

def generate_fallback_response(question, merged_context):
    """Generate a conversational fallback response when LLM fails."""
    if "requirement" in question.lower():
        intro = "Based on the DuraFlex documentation, here are the system requirements you'll need:"
    elif "install" in question.lower() or "setup" in question.lower():
        intro = "According to the DuraFlex documentation, here's how to set up the system:"
    elif "problem" in question.lower() or "issue" in question.lower():
        intro = "Based on the DuraFlex documentation, here are the solutions to common issues:"
    else:
        intro = "Based on the DuraFlex documentation, here's what I found:"
    
    return f"{intro}\n\n{merged_context}\n\nI hope this information helps answer your question!"

def generate_structured_response(question, facts):
    """Generate a clean, professional response from facts."""
    if not facts:
        return "I couldn't find relevant information in the DuraFlex documentation to answer your question."
    
    # Create clean, professional responses
    response_parts = []
    
    if "requirement" in question.lower() or "spec" in question.lower():
        response_parts.append("Based on the DuraFlex documentation, here are the system requirements:")
    elif "install" in question.lower() or "setup" in question.lower() or "connect" in question.lower():
        response_parts.append("According to the DuraFlex documentation, here's the process:")
    elif "problem" in question.lower() or "issue" in question.lower() or "troubleshoot" in question.lower():
        response_parts.append("Based on the DuraFlex documentation, here are the solutions:")
    else:
        response_parts.append("Based on the DuraFlex documentation:")
    
    # Add clean facts as bullet points
    for i, fact in enumerate(facts, 1):
        clean_text = fact['text'].strip()
        
        # Format as clean bullet points
        response_parts.append(f"\n• {clean_text}")
    
    return ''.join(response_parts)

def generate_natural_response(question, facts):
    """Generate a natural, human-readable response from facts."""
    if not facts:
        return "I couldn't find relevant information in the DuraFlex documentation to answer your question."
    
    # Create a more conversational prompt for better LLM
    facts_text = "\n".join([f"- {fact['text']}" for fact in facts])
    
    prompt = f"""Answer this question using ONLY the information provided below. Do not make up or add any information not in the documentation.

Question: {question}

Documentation Information:
{facts_text}

Answer:"""
    
    try:
        logger.info("Generating natural language response...")
        logger.info(f"Prompt length: {len(prompt)} characters")
        response = llm.complete(prompt)
        response_text = str(response).strip()
        logger.info(f"Raw LLM response: '{response_text}'")
        
        # Always use structured fallback for reliable, accurate responses
        logger.info("Using structured conversational fallback for reliable responses")
        return generate_structured_response(question, facts)
    except Exception as e:
        logger.error(f"ERROR: Failed to generate natural response: {str(e)}")
        # Fallback to structured response
        return generate_structured_response(question, facts)

def hybrid_query(question, similarity_top_k=10):
    """Advanced RAG pipeline with reranking, chunk merging, and document summarization."""
    if index is None:
        raise RuntimeError("Index not loaded. Please initialize first.")
    
    logger.info(f"Processing question: '{question}'")
    
    try:
        # Step 1: Retrieve top 10 chunks by embedding similarity
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        initial_nodes = retriever.retrieve(question)
        
        logger.info(f"Retrieved {len(initial_nodes)} initial chunks")
        
        # Step 2: Apply reranker to get top 5 most relevant chunks
        reranked_nodes = apply_reranker(question, initial_nodes)
        logger.info(f"Reranked to {len(reranked_nodes)} high-quality chunks")
        
        # Step 3: Merge adjacent chunks from same document
        merged_nodes = merge_adjacent_chunks(reranked_nodes)
        logger.info(f"Merged into {len(merged_nodes)} final chunks")
        
        # Step 4: Summarize each document into 2-3 bullet points
        summarized_docs = summarize_documents(merged_nodes)
        logger.info(f"Summarized {len(summarized_docs)} documents")
        
        # Step 5: Merge summaries into a single context block
        merged_context = merge_summaries(summarized_docs)
        logger.info(f"Created merged context of {len(merged_context)} characters")
        
        # Step 6: Generate conversational response using LLM
        conversational_response = generate_conversational_response(question, merged_context)
        
        logger.info(f"Generated conversational response of {len(conversational_response)} characters")
        return conversational_response
        
    except Exception as e:
        logger.error(f"ERROR: Error in hybrid query: {str(e)}")
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
        "output": "relevant document chunks"
    }
    """
    logger.info("Handler function called")
    
    try:
        # Initialize models and load index on first run
        if index is None:
            logger.info("Initializing models...")
            initialize_models()
            logger.info("Loading vector index...")
            load_vector_index()
        
        # Parse the input
        if isinstance(event, str):
            event = json.loads(event)
        
        # Extract the question from the input
        if "input" not in event:
            return {
                "error": "Missing 'input' field in request",
                "output": "Invalid request format. Expected: {\"input\": {\"question\": \"your question\"}}"
            }
        
        question = event["input"].get("question")
        if not question:
            return {
                "error": "Missing 'question' field in input",
                "output": "Please provide a question in the 'question' field."
            }
        
        logger.info(f"Question extracted: '{question}'")
        
        # Query the index using hybrid approach
        logger.info("Starting hybrid query processing...")
        answer = hybrid_query(question)
        
        logger.info("SUCCESS: Query completed successfully!")
        
        return {"output": answer}
        
    except FileNotFoundError as e:
        error_msg = f"Vector database not found: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "output": "The vector database is not available. Please ensure ingest.py has been run to create the index."
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "output": "I'm sorry, I encountered an error while processing your question. Please try again."
        }

# For local testing
if __name__ == "__main__":
    logger.info("Starting hybrid handler test...")
    
    test_event = {
        "input": {
            "question": "Tell me about the higher performance system requirements for duraflex. Do not include any dates and only list the specific requirements.."
        }
    }
    
    result = handler(test_event)
    print("\n" + "="*50)
    print("HYBRID HANDLER TEST RESULT:")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50)
