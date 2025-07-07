"""
Query engine module for handling LLM interactions and response generation.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.schema import Document as LangChainDocument

from src.config import OPENAI_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Supported query intent types."""
    SUMMARY = "summary"
    SPECIFIC_QUESTION = "specific_question"
    DATA_EXTRACTION = "data_extraction"
    NEWS_CHECK = "news_check"


class QueryEngine:
    """Handles LLM interactions and response generation."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0):
        """
        Initialize query engine with LLM model.
        
        Args:
            model_name: Name of OpenAI model to use
            temperature: Sampling temperature (0 to 1)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()
        
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
            Create a concise summary of the following competitive intelligence information.
            Focus on key insights, trends, and strategic implications.

            Context:
            {context}

            Query: {query}

            Provide a clear, well-structured summary that addresses the query.
        """)

        self.qa_prompt = ChatPromptTemplate.from_template("""
            Answer the following question using only the provided context.
            If the answer cannot be found in the context, say so clearly.

            Context:
            {context}

            Question: {query}

            Provide a direct, factual answer with specific references where possible.
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Extract specific data points or metrics from the following context.
            Format the output clearly with appropriate headers and structure.

            Context:
            {context}

            Data to extract: {query}

            Extract and organize the requested information systematically.
        """)

    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the intent of a user query.
        
        Args:
            query: User's input query
            
        Returns:
            Classified QueryIntent
        """
        chain = self.intent_prompt | self.llm | self.output_parser
        result = chain.invoke({"query": query})
        return QueryIntent(result.lower())

    def generate_response(self, query: str, context: List[str],
                         intent: Optional[QueryIntent] = None) -> str:
        """
        Generate a response using the appropriate prompt template.
        
        Args:
            query: User's input query
            context: Retrieved context chunks
            intent: Optional pre-classified intent
            
        Returns:
            Generated response string
        """
        if intent is None:
            intent = self.classify_intent(query)
            
        # Select appropriate prompt template
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
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({
            "context": formatted_context,
            "query": query
        })

    def evaluate_response(self, query: str, response: str,
                         context: List[str]) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response.
        
        Args:
            query: Original query
            response: Generated response
            context: Source context used
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Initialize evaluators
            qa_evaluator = load_evaluator("qa")
            criteria_evaluator = load_evaluator(
                "criteria",
                criteria={
                    "relevance": "Does the response directly address the query?",
                    "accuracy": "Is the response factually accurate based on the context?",
                    "completeness": "Does the response cover all key points from the context?",
                    "coherence": "Is the response well-structured and easy to understand?"
                }
            )
            
            # Run evaluations
            qa_result = qa_evaluator.evaluate_strings(
                prediction=response,
                input=query,
                reference="\n".join(context)
            )
            
            criteria_result = criteria_evaluator.evaluate_strings(
                prediction=response,
                input=query,
                reference="\n".join(context)
            )
            
            # Combine results
            metrics = {
                "qa_score": qa_result.get("score", 0.0),
                "criteria_scores": {
                    k: v for k, v in criteria_result.items()
                    if k in ["relevance", "accuracy", "completeness", "coherence"]
                },
                "overall_score": qa_result.get("score", 0.0) * 0.5 + 
                                sum(v for k, v in criteria_result.items() 
                                    if k in ["relevance", "accuracy", "completeness", "coherence"]) / 8.0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {
                "error": str(e),
                "qa_score": 0.0,
                "overall_score": 0.0
            } 