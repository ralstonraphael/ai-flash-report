"""
Query engine module for handling LLM interactions and response generation.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
import asyncio
from concurrent.futures import TimeoutError
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.schema import Document as LangChainDocument

from src.config import OPENAI_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 1  # seconds

class QueryIntent(Enum):
    """Supported query intent types."""
    SUMMARY = "summary"
    SPECIFIC_QUESTION = "specific_question"
    DATA_EXTRACTION = "data_extraction"
    NEWS_CHECK = "news_check"

class QueryTimeoutError(Exception):
    """Raised when a query takes too long to complete."""
    pass

class QueryEngine:
    """Handles LLM interactions and response generation."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0,
                 timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize query engine with LLM model.
        
        Args:
            model_name: Name of OpenAI model to use
            temperature: Sampling temperature (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            request_timeout=timeout,
            max_retries=MAX_RETRIES
        )
        self.output_parser = StrOutputParser()
        self.timeout = timeout
        
        # Initialize prompt templates
        self._init_prompts()

    def _init_prompts(self):
        """Initialize different prompt templates for each query type."""
        self.intent_prompt = ChatPromptTemplate.from_template("""
            You are an intent classifier for a competitive intelligence system.
            Determine what kind of response the user is expecting from the query below.

            Options:
            - summary
            - specific_question
            - data_extraction
            - news_check

            Query: {query}

            Respond with only one of the options above.
        """)

        self.summary_prompt = ChatPromptTemplate.from_template("""
            Create a comprehensive and detailed summary of the provided context.
            Include all significant points, changes, updates, and strategic implications.
            
            Structure your response in multiple sections:
            1. Executive Overview (2-3 paragraphs)
            2. Key Changes and Updates (detailed bullet points)
            3. Strategic Implications (thorough analysis)
            4. Future Outlook (insights and predictions)
            
            For each section, provide rich detail and supporting information.
            Use specific examples and data points from the context when available.

            Context:
            {context}

            Query: {query}

            Aim for a thorough and insightful analysis that covers all important aspects.
        """)

        self.qa_prompt = ChatPromptTemplate.from_template("""
            Provide a comprehensive and detailed answer using the provided context.
            If the answer cannot be found in the context, say so clearly.
            
            Structure your response to include:
            1. Direct answer to the question (detailed explanation)
            2. Supporting evidence and examples from the context
            3. Related implications and considerations
            4. Additional relevant insights
            
            Context:
            {context}

            Question: {query}

            Provide a thorough analysis that gives the full picture.
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Extract and analyze all relevant metrics and data points.
            Provide detailed context and interpretation for each metric.
            
            Structure your response as follows:
            1. Key Performance Indicators
               • Detailed breakdown of each metric
               • Historical trends when available
               • Context and implications
            
            2. Market and Competition Metrics
               • Market share data
               • Competitive positioning
               • Industry benchmarks
            
            3. Financial and Operational Metrics
               • Detailed financial analysis
               • Operational performance indicators
               • Growth metrics
            
            Context:
            {context}

            Data to extract: {query}

            Format each section with detailed bullet points and provide analysis for each metric.
            Include trends, comparisons, and strategic implications where possible.
        """)

    async def _execute_with_timeout(self, func, *args, **kwargs):
        """Execute a function with timeout."""
        try:
            # Create a task for the function
            task = asyncio.create_task(func(*args, **kwargs))
            
            # Wait for the task to complete with timeout
            result = await asyncio.wait_for(task, timeout=self.timeout)
            return result
        
        except asyncio.TimeoutError:
            # Cancel the task if it times out
            task.cancel()
            raise QueryTimeoutError(f"Operation timed out after {self.timeout} seconds")
        
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            raise

    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the intent of a user query.
        
        Args:
            query: User's input query
            
        Returns:
            Classified QueryIntent
        """
        start_time = time.time()
        logger.info(f"Classifying intent for query: {query}")
        
        try:
            chain = self.intent_prompt | self.llm | self.output_parser
            result = chain.invoke({"query": query})
            
            logger.info(f"Intent classification completed in {time.time() - start_time:.2f} seconds")
            return QueryIntent(result.lower())
            
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            raise

    def generate_section_content(self, section_type: str, context: List[str]) -> str:
        """
        Generate detailed content for a specific report section.
        
        Args:
            section_type: Type of section (summary, overview, etc.)
            context: Retrieved context chunks
            
        Returns:
            Comprehensive section content
        """
        prompts = {
            "executive_summary": """
                Generate a detailed executive summary that covers all key aspects.
                Include comprehensive analysis of changes, trends, and implications.
                
                Structure your response:
                1. Overview (2-3 detailed paragraphs)
                2. Key Developments (comprehensive bullet points)
                3. Strategic Analysis (thorough examination)
                4. Recommendations (detailed action items)
                
                Context: {context}
                
                Provide rich detail and supporting evidence for each point.
            """,
            
            "company_overview": """
                Provide a comprehensive company overview that includes:
                1. Detailed business model analysis
                2. Historical development and milestones
                3. Current market position and strategy
                4. Organizational structure and leadership
                5. Core competencies and capabilities
                
                Context: {context}
                
                Include specific examples and data points where available.
            """,
            
            "core_offerings": """
                Provide a detailed analysis of products/services including:
                1. Comprehensive product/service portfolio
                2. Recent developments and innovations
                3. Market reception and performance
                4. Competitive advantages
                5. Future roadmap and potential
                
                Context: {context}
                
                Include specific features, benefits, and market impact.
            """,
            
            "market_position": """
                Deliver a thorough analysis of market position including:
                1. Detailed market share analysis
                2. Competitive landscape evaluation
                3. Key differentiators and advantages
                4. Market trends and dynamics
                5. Growth opportunities and challenges
                
                Context: {context}
                
                Support each point with specific data and examples.
            """,
            
            "strategic_insights": """
                Provide comprehensive strategic insights including:
                1. Detailed SWOT analysis
                2. Market opportunities and threats
                3. Competitive advantages and challenges
                4. Growth strategies and recommendations
                5. Risk analysis and mitigation
                
                Context: {context}
                
                Include specific, actionable recommendations with supporting rationale.
            """
        }
        
        prompt = ChatPromptTemplate.from_template(prompts.get(section_type, self.summary_prompt))
        chain = prompt | self.llm | self.output_parser
        
        return chain.invoke({
            "context": "\n\n---\n\n".join(context)
        })

    def generate_response(self, query: str, context: List[str],
                         intent: Optional[QueryIntent] = None,
                         progress_callback: Optional[callable] = None) -> str:
        """
        Generate a response using the appropriate prompt template.
        
        Args:
            query: User's input query
            context: Retrieved context chunks
            intent: Optional pre-classified intent
            progress_callback: Optional callback function to report progress
            
        Returns:
            Generated response string
        """
        start_time = time.time()
        
        try:
            # Classify intent if not provided
            if intent is None:
                if progress_callback:
                    progress_callback("Classifying query intent...")
                intent = self.classify_intent(query)
            
            # Select appropriate prompt template
            if progress_callback:
                progress_callback("Preparing response generation...")
            
            if intent == QueryIntent.SUMMARY:
                prompt = self.summary_prompt
            elif intent == QueryIntent.SPECIFIC_QUESTION:
                prompt = self.qa_prompt
            elif intent == QueryIntent.DATA_EXTRACTION:
                prompt = self.extraction_prompt
            else:
                prompt = self.qa_prompt  # Default to QA prompt
            
            # Join context chunks with separators
            formatted_context = "\n\n---\n\n".join(context)
            
            # Generate response
            if progress_callback:
                progress_callback("Generating response...")
            
            chain = prompt | self.llm | self.output_parser
            response = chain.invoke({
                "context": formatted_context,
                "query": query
            })
            
            logger.info(f"Response generation completed in {time.time() - start_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise

    def evaluate_response(self, query: str, response: str,
                         context: List[str],
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response.
        
        Args:
            query: Original query
            response: Generated response
            context: Source context used
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback("Evaluating response quality...")
            
            # Create custom evaluator for more accurate scoring
            eval_prompt = ChatPromptTemplate.from_template("""
                You are an expert evaluator for competitive intelligence responses.
                Evaluate the following response based on multiple criteria.
                Provide a quick, focused evaluation.
                
                Original Query: {query}
                Generated Response: {response}
                
                Score each criterion from 0.0 to 1.0:
                1. Relevance
                2. Accuracy
                3. Completeness
                4. Coherence
                5. Conciseness
                
                Format as JSON: {{"relevance": {{"score": float}}, "accuracy": {{"score": float}}, ...}}
            """)
            
            # Generate evaluation
            chain = eval_prompt | self.llm | self.output_parser
            eval_result = chain.invoke({
                "query": query,
                "response": response
            })
            
            # Parse evaluation result
            import json
            metrics = json.loads(eval_result)
            
            logger.info(f"Response evaluation completed in {time.time() - start_time:.2f} seconds")
            return metrics
            
        except Exception as e:
            logger.error(f"Response evaluation failed: {str(e)}")
            return {
                "error": str(e),
                "relevance": {"score": 0.0},
                "accuracy": {"score": 0.0},
                "completeness": {"score": 0.0},
                "coherence": {"score": 0.0},
                "conciseness": {"score": 0.0}
            } 