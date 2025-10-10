"""
Mock RAG system for UI development without GPU
Provides realistic responses without requiring actual models or vector index
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import random

@dataclass
class MockIntent:
    """Mock intent classification result"""
    intent_type: str = "troubleshooting"
    confidence: float = 0.85
    keywords: List[str] = field(default_factory=lambda: ["troubleshoot", "fix", "issue"])
    requires_subqueries: bool = False

@dataclass  
class MockSource:
    """Mock source document"""
    file_name: str
    page_number: int
    content: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class MockResponse:
    """Mock structured response matching StructuredResponse from orchestrator.py"""
    answer: str
    sources: List[MockSource]
    confidence: float
    intent: MockIntent
    reasoning: str
    
class MockEliteRAGQuery:
    """Mock RAG system for UI testing - mimics EliteRAGQuery interface"""
    
    def __init__(self, cache_dir=None):
        self.initialized = False
        self.cache_dir = cache_dir
        
        # Sample DuraFlex documents for realistic responses
        self.sample_docs = [
            "DuraFlex Troubleshooting Guide V4.05_30May2022.pdf",
            "DuraFlex Operations Guide V4.06_21Sep2022.pdf",
            "DuraFlex Installation and Commissioning Guide V5.01_06Apr2023.pdf",
            "DuraFlex Electrical Databook and Design Guide V4.03_02Aug2021.pdf",
            "DuraFlex Service and Repair Guide V1.02_17Sep2021.pdf",
            "print_quality_artefacts_reference_guide.pdf",
            "DuraFlex Printhead Cradle Repair Guide_V1.1_03Oct2025.pdf",
        ]
        
    def initialize(self, storage_dir=None):
        """Simulate initialization delay"""
        print("🤖 [MOCK MODE] Initializing mock RAG system...")
        time.sleep(1)  # Simulate loading
        self.initialized = True
        print("✅ [MOCK MODE] Mock RAG system ready")
        
    def query(
        self, 
        query_text: str, 
        top_k: int = 10, 
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True
    ) -> MockResponse:
        """Return mock response based on query keywords"""
        
        # Simulate query processing time
        time.sleep(random.uniform(0.3, 0.7))
        
        # Determine intent based on keywords
        intent_type = self._classify_intent(query_text)
        
        # Generate contextual answer
        answer = self._generate_answer(query_text, intent_type)
        
        # Create mock sources
        sources = self._generate_sources(query_text, top_k, metadata_filters)
        
        # Calculate confidence
        confidence = random.uniform(0.75, 0.95)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query_text, intent_type, top_k, alpha, len(sources))
        
        return MockResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            intent=MockIntent(
                intent_type=intent_type,
                confidence=random.uniform(0.80, 0.95),
                keywords=self._extract_keywords(query_text),
                requires_subqueries=len(query_text.split()) > 10
            ),
            reasoning=reasoning
        )
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent based on keywords"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'steps', 'procedure', 'install', 'setup']):
            return 'procedural'
        elif any(word in query_lower for word in ['fix', 'repair', 'troubleshoot', 'error', 'problem', 'issue']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
            return 'comparison'
        else:
            return 'lookup'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple keyword extraction
        stopwords = {'what', 'is', 'the', 'how', 'to', 'a', 'an', 'and', 'or', 'in', 'on', 'for'}
        words = [w.lower().strip('?,!.') for w in query.split() if w.lower() not in stopwords]
        return words[:5]  # Return top 5 keywords
    
    def _generate_answer(self, query: str, intent_type: str) -> str:
        """Generate contextual mock answer"""
        
        # Extract main topic from query
        keywords = self._extract_keywords(query)
        topic = ' '.join(keywords[:3]) if keywords else 'the system'
        
        if intent_type == 'troubleshooting':
            return f"""According to DuraFlex Troubleshooting Guide V4.05_30May2022.pdf [1]:

**[MOCK RESPONSE]** For issues related to {topic}, follow these troubleshooting steps:

1. First, verify all connections are secure and properly seated
2. Check the system logs for any error codes or warnings
3. Ensure temperature is within the optimal range (28-32°C)
4. Run a diagnostic test using the built-in system tools

According to DuraFlex Operations Guide V4.06_21Sep2022.pdf [2]:

If the issue persists, perform a system reset following the procedure in Section 4.2. Make sure to document all steps taken for support reference.

**Note:** This is a mock response for UI development. Real responses will be generated from the actual knowledge base when running on GPU."""

        elif intent_type == 'procedural':
            return f"""According to DuraFlex Installation and Commissioning Guide V5.01_06Apr2023.pdf [1]:

**[MOCK RESPONSE]** To {query.lower().replace('how to ', '').replace('?', '')}, follow these steps:

**Step 1: Preparation**
- Gather necessary tools and materials
- Review safety procedures in Section 2.1 [1]
- Ensure system is powered down

**Step 2: Main Procedure**
- Follow the detailed instructions in Section 5.3 [2]
- Verify each step before proceeding
- Document any deviations from standard procedure

**Step 3: Verification**
- Run system diagnostics
- Confirm all parameters are within specification
- Complete the installation checklist

**Note:** This is a mock response for UI development. Real responses will be generated from the actual knowledge base."""

        elif intent_type == 'definition':
            return f"""According to DuraFlex Operations Guide V4.06_21Sep2022.pdf [1]:

**[MOCK RESPONSE]** {topic.title()} refers to a critical component of the DuraFlex printing system. 

Key characteristics include:
- Primary function: Controls and regulates system operation
- Operating parameters: Temperature 28-32°C, Pressure 2-4 bar
- Maintenance interval: Every 500 operating hours [2]

For detailed specifications, refer to the Electrical Databook Section 3.4 [3].

**Note:** This is a mock response for UI development. Real definitions will be sourced from actual technical documentation."""

        elif intent_type == 'comparison':
            return f"""According to DuraFlex Service and Repair Guide V1.02_17Sep2021.pdf [1]:

**[MOCK RESPONSE]** Comparing the options for {topic}:

**Option A:**
- Advantages: Higher reliability, lower maintenance
- Disadvantages: Higher initial cost
- Best for: High-volume production environments

**Option B:**
- Advantages: Lower cost, easier setup
- Disadvantages: More frequent maintenance required
- Best for: Low to medium volume applications

Refer to Section 7.2 for detailed comparison charts [2].

**Note:** This is a mock response for UI development."""

        else:  # lookup
            return f"""According to DuraFlex technical documentation [1]:

**[MOCK RESPONSE]** The {topic} specifications are:

- Operating Range: 28-32°C
- Voltage: 220-240V AC
- Power Consumption: 150-200W
- Dimensions: 450 × 300 × 200 mm
- Weight: 12.5 kg

For complete specifications, see the technical datasheet Section 4 [2].

**Note:** This is a mock response for UI development. Real specifications will be retrieved from actual documents."""
    
    def _generate_sources(
        self, 
        query: str, 
        top_k: int,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[MockSource]:
        """Generate mock source documents"""
        
        num_sources = min(random.randint(2, 5), top_k)
        sources = []
        
        # Determine content type filter
        content_type_filter = metadata_filters.get('content_type') if metadata_filters else None
        
        for i in range(num_sources):
            doc = random.choice(self.sample_docs)
            page = random.randint(1, 100)
            score = random.uniform(0.65, 0.95) - (i * 0.05)  # Decreasing scores
            
            # Determine content type
            if content_type_filter:
                content_type = content_type_filter
            else:
                content_type = random.choice(['text', 'text', 'text', 'table', 'caption'])
            
            sources.append(MockSource(
                file_name=doc,
                page_number=page,
                content=f"[MOCK] Relevant content from {doc} page {page} about {query[:50]}...",
                score=score,
                metadata={
                    'content_type': content_type,
                    'file_name': doc,
                    'page': page,
                }
            ))
        
        return sources
    
    def _generate_reasoning(
        self, 
        query: str, 
        intent_type: str, 
        top_k: int, 
        alpha: float,
        num_sources: int
    ) -> str:
        """Generate mock reasoning explanation"""
        
        avg_relevance = random.uniform(0.75, 0.92)
        
        return f"""**[MOCK REASONING]**

Retrieved {random.randint(6, 12)} relevant document chunks using hybrid search (dense embeddings + BM25).

Query intent classified as: **{intent_type}** (confidence: {random.randint(80, 95)}%)

Search strategy:
- Hybrid search weight (alpha): {alpha}
- Chunks requested: {top_k}
- Dynamic windowing: {'enabled' if random.choice([True, False]) else 'disabled'}

Prioritized {num_sources} sources based on:
- Relevance score threshold: {avg_relevance:.3f}
- Content recency
- Metadata reliability

Average relevance score: {avg_relevance:.3f}

**Note:** This is mock reasoning for UI development. Real reasoning will reflect actual RAG processing."""
    
    def format_response(self, response: MockResponse) -> str:
        """Format response for display (matches EliteRAGQuery interface)"""
        
        output = "=" * 80 + "\n"
        output += "ANSWER:\n"
        output += "=" * 80 + "\n"
        output += response.answer + "\n\n"
        
        output += "=" * 80 + "\n"
        output += "REASONING SUMMARY:\n"
        output += "=" * 80 + "\n"
        output += response.reasoning + "\n\n"
        
        output += "=" * 80 + "\n"
        output += "SOURCE SUMMARY:\n"
        output += "=" * 80 + "\n"
        for i, source in enumerate(response.sources, 1):
            output += f"[{i}] {source.file_name} (page: {source.page_number}, score: {source.score:.3f})\n"
        
        output += f"\nConfidence: {response.confidence:.0%} | Intent: {response.intent.intent_type}\n"
        output += "=" * 80 + "\n"
        
        return output


# For testing
if __name__ == "__main__":
    print("Testing Mock RAG System\n")
    
    rag = MockEliteRAGQuery()
    rag.initialize()
    
    test_queries = [
        "How to troubleshoot print quality issues?",
        "What is the PPU voltage specification?",
        "Compare inkjet vs thermal printing",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.query(query)
        print(rag.format_response(response))

