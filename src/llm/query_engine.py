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
            Create a very concise summary focusing only on the most critical points.
            Limit to 2-3 short paragraphs maximum.
            Focus on key changes, updates, and strategic implications.

            Context:
            {context}

            Query: {query}

            Keep the response brief and focused on essential information only.
        """)

        self.qa_prompt = ChatPromptTemplate.from_template("""
            Provide a direct, concise answer using only the provided context.
            If the answer cannot be found in the context, say so clearly.
            Limit response to 1-2 short paragraphs.

            Context:
            {context}

            Question: {query}

            Focus on key facts and implications only.
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Extract only the most important metrics and data points.
            Format each point concisely on a new line.
            Limit to 5-7 key metrics maximum.
            
            Context:
            {context}

            Data to extract: {query}

            Format as:
            • Metric: Value (Time Period)
            
            Focus on metrics that provide strategic insights.
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
        Generate content for a specific report section with length constraints.
        
        Args:
            section_type: Type of section (summary, overview, etc.)
            context: Retrieved context chunks
            
        Returns:
            Concise section content
        """
        prompts = {
            "executive_summary": """
                Generate a very concise executive summary (2-3 sentences).
                Focus on the most important changes and implications.
                
                Context: {context}
                
                Keep it extremely brief but impactful.
            """,
            
            "company_overview": """
                Provide a brief company overview (2-3 bullet points).
                Include only essential business model and strategic information.
                
                Context: {context}
                
                Format as short bullet points.
            """,
            
            "core_offerings": """
                List main products/services (3-4 bullet points maximum).
                Focus on recent changes and key features.
                
                Context: {context}
                
                Keep each point to one line.
            """,
            
            "market_position": """
                Summarize market position in 2-3 key points.
                Include only critical competitive information.
                
                Context: {context}
                
                Focus on differentiators and trends.
            """,
            
            "strategic_insights": """
                Provide 3-4 key strategic insights or recommendations.
                Make each point actionable and specific.
                
                Context: {context}
                
                Format as brief bullet points.
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