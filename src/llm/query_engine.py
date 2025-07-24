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
        
        # Company-specific context for Norstella
        self.company_context = """
        Norstella is a global healthcare technology company formed through the combination of multiple industry-leading companies. 
        Norstella provides technology-enabled solutions, analytics, and insights to help pharmaceutical, biotechnology, and medical device companies 
        accelerate the development and commercialization of their products. The company serves as a strategic partner throughout the product lifecycle, 
        from early-stage research and development through commercialization and market access.
        """
        
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
            
            CRITICAL REQUIREMENTS:
            1. Write 4-6 full paragraphs with complete sentences - NO abbreviations or trailing dots
            2. Use the actual company name from the context - NEVER use "The company" or generic terms
            3. Include specific details, numbers, dates, and examples from the context
            4. Write in full, complete sentences with proper conclusions
            5. Provide thorough analysis and insights
            6. Each paragraph should be 4-6 sentences long
            
            Structure your response as follows:
            
            **Executive Overview**
            Write 2-3 detailed paragraphs covering the main developments and their significance. Include specific company names, dates, financial figures, and strategic implications.
            
            **Key Changes and Updates**
            Write 1-2 paragraphs detailing specific changes, launches, acquisitions, or strategic moves. Be specific about what changed, when, and why it matters.
            
            **Strategic Implications**
            Write 1-2 paragraphs analyzing what these developments mean for the company's future, competitive position, and market opportunities.
            
            **Future Outlook**
            Write 1 paragraph providing insights and predictions based on the information provided.

            Context:
            {context}

            Query: {query}

            Remember: Write complete, full sentences. Use specific company names. Provide detailed analysis with supporting evidence.
        """)

        self.qa_prompt = ChatPromptTemplate.from_template("""
            Provide a comprehensive and detailed answer using the provided context.
            
            CRITICAL REQUIREMENTS:
            1. Write 3-5 full paragraphs with complete sentences - NO abbreviations or trailing dots
            2. Use the actual company name from the context - NEVER use "The company" or generic terms
            3. Include specific details, numbers, dates, and examples from the context
            4. Write in full, complete sentences with proper conclusions
            5. Provide thorough analysis and supporting evidence
            
            Structure your response to include:
            
            **Direct Answer**
            Provide a detailed, complete answer to the question in 1-2 full paragraphs. Include specific company names, figures, and context.
            
            **Supporting Evidence and Examples**
            Write 1-2 paragraphs providing specific examples, data points, and evidence from the context that support your answer.
            
            **Related Implications and Considerations**
            Write 1 paragraph discussing the broader implications and what this means for the company's strategy or market position.
            
            **Additional Relevant Insights**
            Write 1 paragraph with additional context or related information that adds value to the answer.
            
            Context:
            {context}

            Question: {query}

            Remember: Write complete, full sentences. Use specific company names. Provide detailed analysis with supporting evidence.
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Extract and analyze all relevant metrics and data points from the context.
            
            CRITICAL REQUIREMENTS:
            1. Write 4-6 full paragraphs with complete sentences - NO abbreviations or trailing dots
            2. Use the actual company name from the context - NEVER use "The company" or generic terms
            3. Include specific details, numbers, dates, and context for each metric
            4. Write in full, complete sentences with proper analysis
            5. Provide interpretation and significance for each data point
            
            Structure your response as follows:
            
            **Key Performance Indicators**
            Write 1-2 paragraphs providing detailed breakdown of each financial and operational metric. Include historical context, trends, and what each number means for the business. Use specific company names and provide full analysis.
            
            **Market and Competition Metrics**
            Write 1-2 paragraphs covering market share data, competitive positioning, and industry benchmarks. Explain how the company compares to competitors and what the market dynamics mean.
            
            **Financial and Operational Analysis**
            Write 1-2 paragraphs with detailed financial analysis, operational performance indicators, and growth metrics. Include year-over-year comparisons and trend analysis.
            
            **Strategic Implications of the Data**
            Write 1 paragraph analyzing what these metrics collectively tell us about the company's strategic position and future prospects.
            
            Context:
            {context}

            Data to extract: {query}

            Remember: Write complete, full sentences. Use specific company names. Provide detailed analysis and interpretation for each metric.
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
                Generate a comprehensive executive summary with complete, substantive content.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 full paragraphs (minimum 100 words each)
                2. Use the actual company name from the context - NEVER use "The company" or generic terms
                3. Include specific details, numbers, dates, and examples from the context
                4. Write in complete sentences with proper conclusions
                5. Do NOT use markdown headers or bullet points
                6. Write as flowing narrative paragraphs
                
                Write a comprehensive executive summary covering:
                
                First paragraph: Current strategic position and recent major developments with specific company names, dates, and financial figures.
                
                Second paragraph: Key business developments, launches, acquisitions, partnerships, or strategic moves with detailed explanations of their significance.
                
                Third paragraph: Market impact and competitive implications of these developments.
                
                Fourth paragraph: Future outlook and strategic implications for the company's growth prospects.
                
                Context: {context}
                
                Remember: Write complete paragraphs without headers. Use specific company names and detailed supporting evidence.
            """,
            
            "company_overview": """
                Provide a comprehensive company overview with detailed analysis in paragraph form.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 full paragraphs (minimum 100 words each)
                2. Use the actual company name from the context - NEVER use "The company" or generic terms
                3. Include specific details about business model, operations, and strategy
                4. Write in complete sentences with proper conclusions
                5. Do NOT use markdown headers or bullet points
                6. Write as flowing narrative paragraphs
                
                Write a comprehensive company overview covering:
                
                First paragraph: Business model, core operations, and how the company generates revenue with specific details about business segments.
                
                Second paragraph: Historical development, key milestones, and recent strategic developments that have shaped the current position.
                
                Third paragraph: Current market position, competitive advantages, and strategic direction with specific examples.
                
                Fourth paragraph: Operational structure, key capabilities, and how the company differentiates itself in the market.
                
                Context: {context}
                
                Remember: Write complete paragraphs without headers. Use specific company names and detailed examples.
            """,
            
            "core_offerings": """
                Provide a detailed analysis of the company's products and services in paragraph form.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 full paragraphs (minimum 100 words each)
                2. Use the actual company name from the context - NEVER use "The company" or generic terms
                3. Include specific details about products, features, and market reception
                4. Write in complete sentences with proper conclusions
                5. Do NOT use markdown headers or bullet points
                6. Write as flowing narrative paragraphs
                
                Write a detailed analysis covering:
                
                First paragraph: Primary products and services portfolio with specific product names, features, and how they serve different customer segments.
                
                Second paragraph: Recent product launches, updates, innovations, or service enhancements with detailed explanations of new features and capabilities.
                
                Third paragraph: Market reception, customer feedback, and adoption rates for key products and services.
                
                Fourth paragraph: Competitive advantages these offerings provide and how they position the company in the market.
                
                Context: {context}
                
                Remember: Write complete paragraphs without headers. Use specific product names and detailed market analysis.
            """,
            
            "market_position": """
                Deliver a thorough analysis of the company's market position and competitive landscape in paragraph form.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 full paragraphs (minimum 100 words each)
                2. Use the actual company name from the context - NEVER use "The company" or generic terms
                3. Include specific details about market share, competitors, and positioning
                4. Write in complete sentences with proper conclusions
                5. Do NOT use markdown headers or bullet points
                6. Write as flowing narrative paragraphs
                
                Write a thorough market analysis covering:
                
                First paragraph: Market share, competitive standing, and how the company compares to key competitors with specific market data and competitor names.
                
                Second paragraph: Key differentiators, unique value propositions, and sustainable competitive advantages that set the company apart.
                
                Third paragraph: Current market trends, industry dynamics, and how these affect the competitive landscape.
                
                Fourth paragraph: The company's positioning strategy and how it's adapting to capitalize on market opportunities or respond to challenges.
                
                Context: {context}
                
                Remember: Write complete paragraphs without headers. Support analysis with specific data and competitor examples.
            """,
            
            "strategic_insights": """
                Provide comprehensive strategic insights and recommendations in paragraph form.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 full paragraphs (minimum 100 words each)
                2. Use the actual company name from the context - NEVER use "The company" or generic terms
                3. Include specific, actionable recommendations with supporting rationale
                4. Write in complete sentences with proper conclusions
                5. Do NOT use markdown headers or bullet points
                6. Write as flowing narrative paragraphs
                
                Write comprehensive strategic insights covering:
                
                First paragraph: Key strategic strengths and market opportunities with specific examples and explanations of how these can be leveraged for growth.
                
                Second paragraph: Primary challenges, competitive threats, and risk factors that could impact performance or strategic objectives.
                
                Third paragraph: Strategic recommendations for capitalizing on opportunities, including specific actions and their expected benefits.
                
                Fourth paragraph: Risk mitigation strategies and recommendations for addressing identified challenges and threats.
                
                Context: {context}
                
                Remember: Write complete paragraphs without headers. Include detailed, actionable recommendations with supporting analysis.
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
            
            # Join context chunks with separators and add company context
            formatted_context = self.company_context + "\n\n---\n\n" + "\n\n---\n\n".join(context)
            
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