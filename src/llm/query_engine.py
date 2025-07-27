"""
Query engine for generating structured responses from document context.
"""
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import asyncio
import logging
import time
from enum import Enum

from langchain.schema import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30  # seconds

class QueryIntent(Enum):
    """Supported query intent types."""
    SUMMARY = "summary"
    SPECIFIC_QUESTION = "specific_question"
    DATA_EXTRACTION = "data_extraction"
    NEWS_CHECK = "news_check"

class QueryTimeoutError(Exception):
    """Raised when a query operation times out."""
    pass

class QueryEngine:
    """Handles different types of queries with structured response generation."""
    
    def __init__(self, model_name: str = OPENAI_MODEL or "gpt-4", temperature: float = 0,
                 timeout: int = DEFAULT_TIMEOUT):
        """Initialize query engine with model and timeout settings."""
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
        
        # Initialize prompts
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
            Create a focused business summary using the provided context.
            
            CRITICAL REQUIREMENTS:
            1. Use actual company names from context - NEVER "The company" or generic terms
            2. Write 2-5 complete sentences per section - NO MORE
            3. Include specific numbers, dates, and strategic facts only
            4. Write in direct, factual language - avoid flowery descriptions
            5. Each section should stand alone with clear information
            
            Structure your response as follows:
            
            **Executive Overview**
            Write 2-4 sentences covering the most significant recent development and why it matters strategically. Include specific company names and financial figures.
            
            **Key Changes and Updates**
            Write 2-3 sentences detailing the most important change, launch, or strategic move. State what changed, when, and the business impact.
            
            **Strategic Implications**
            Write 2-4 sentences analyzing what this means for competitive position and market opportunities. Focus on actionable insights.
            
            **Future Outlook**
            Write 2-3 sentences providing specific predictions or expected outcomes based on the evidence provided.

            Context:
            {context}

            Query: {query}

            Remember: Keep each section to 2-5 sentences maximum. Use specific company names and concrete facts.
        """)

        self.qa_prompt = ChatPromptTemplate.from_template("""
            Provide a focused answer using the provided context.
            
            CRITICAL REQUIREMENTS:
            1. Use actual company names from context - NEVER "The company" or generic terms
            2. Write 2-5 complete sentences per section - NO MORE
            3. Include specific numbers, dates, and examples from context
            4. Write in direct, factual language with clear evidence
            5. Each section should provide distinct information
            
            Structure your response as follows:
            
            **Direct Answer**
            Write 2-4 sentences providing a clear, specific answer to the question. Include company names, figures, and direct context.
            
            **Supporting Evidence**
            Write 2-3 sentences providing specific examples and data points that support your answer.
            
            **Strategic Context**
            Write 2-4 sentences explaining what this means for the company's strategy or market position.
            
            **Additional Insights**
            Write 2-3 sentences with related information that adds value to understanding the answer.
            
            Context:
            {context}

            Question: {query}

            Remember: Keep each section to 2-5 sentences maximum. Use specific company names and concrete evidence.
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Extract and analyze key metrics and data points from the context.
            
            CRITICAL REQUIREMENTS:
            1. Use actual company names from context - NEVER "The company" or generic terms
            2. Write 2-5 complete sentences per section - NO MORE
            3. Include specific numbers, dates, and context for each metric
            4. Write in direct, analytical language with clear interpretation
            5. Focus on the most strategically important data points
            
            Structure your response as follows:
            
            **Key Performance Indicators**
            Write 2-4 sentences covering the most important financial and operational metrics. Include specific numbers, context, and what they mean for the business.
            
            **Market and Competition Data**
            Write 2-3 sentences covering market share, competitive positioning, and industry benchmarks. Compare to competitors where possible.
            
            **Financial and Operational Metrics**
            Write 2-4 sentences with key financial performance indicators and growth metrics. Include year-over-year comparisons and trends.
            
            **Strategic Implications**
            Write 2-3 sentences analyzing what these metrics reveal about the company's strategic position and future prospects.
            
            Context:
            {context}

            Data to extract: {query}

            Remember: Keep each section to 2-5 sentences maximum. Focus on metrics that drive strategic decisions.
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
                Write a strategic executive summary for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company names from context - NEVER "The company" or generic terms
                3. NO essay structure, NO conclusions, NO "In summary", NO "Overall"
                4. Each paragraph tells its own focused story with specific facts
                5. Direct, factual language - avoid academic or flowery writing
                6. Include specific numbers, dates, and strategic developments only
                
                Paragraph 1: State the most critical recent development that changes the competitive game. Include specific company name, date, financial figure, and why this matters strategically.
                
                Paragraph 2: Identify the single most important strategic implication or risk that executives must address. Be specific about impact and urgency.
                
                Context: {context}
                
                Write like you're briefing a CEO who has 30 seconds to read this.
            """,
            
            "company_overview": """
                Write a strategic company overview for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company names from context - NEVER "The company" or generic terms
                3. NO essay structure, NO conclusions, NO "In summary", NO "To conclude"
                4. Each paragraph tells its own focused story with specific facts
                5. Direct, factual language - avoid generic business speak
                6. Focus only on elements that create competitive advantage
                
                Paragraph 1: State the core business model and primary revenue engine. Name specific business segments and quantify their contribution to competitive position.
                
                Paragraph 2: Highlight the most significant recent strategic move that reshapes how they compete. Include specifics on what changed and strategic impact.
                
                Context: {context}
                
                Write like you're explaining to a competitor analysis team.
            """,
            
            "core_offerings": """
                Write a strategic offerings analysis for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company and product names from context - be specific
                3. NO essay structure, NO conclusions, NO "In conclusion", NO "Overall"
                4. Each paragraph tells its own focused story with specific facts
                5. Direct, factual language - avoid marketing speak
                6. Focus only on offerings that drive strategic advantage
                
                Paragraph 1: Name the 2-3 most strategically important products/services and state exactly how they create competitive differentiation. Include specific features or capabilities.
                
                Paragraph 2: Identify the most significant recent product development and quantify its strategic impact on market position or growth potential.
                
                Context: {context}
                
                Write like you're briefing a product strategy team.
            """,
            
            "market_position": """
                Write a strategic market position analysis for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company and competitor names from context - be specific
                3. NO essay structure, NO conclusions, NO "In summary", NO "Therefore"
                4. Each paragraph tells its own focused story with specific facts
                5. Direct, factual language - avoid consultant speak
                6. Include specific market share numbers or competitive metrics
                
                Paragraph 1: State current market position versus named competitors. Include specific market share, ranking, or competitive metrics that matter strategically.
                
                Paragraph 2: Identify the most significant competitive threat or opportunity facing them right now. Name the competitor and quantify the strategic risk or opportunity.
                
                Context: {context}
                
                Write like you're briefing a competitive intelligence team.
            """,
            
            "strategic_insights": """
                Write strategic insights for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company names from context - be specific
                3. NO essay structure, NO conclusions, NO "In conclusion", NO "To summarize"
                4. Each paragraph tells its own focused story with specific recommendations
                5. Direct, actionable language - avoid generic strategy speak
                6. Provide specific strategic recommendations, not broad observations
                
                Paragraph 1: State the single most critical strategic opportunity they should pursue immediately. Be specific about the opportunity, timeline, and potential impact.
                
                Paragraph 2: Identify the most significant strategic risk they must address. Name the specific threat, timeline, and recommended response.
                
                Context: {context}
                
                Write like you're giving direct recommendations to the CEO.
            """,
            
            "executive_summary_new": """
                Write an executive summary for the left column of a flash report.
                
                CRITICAL REQUIREMENTS:
                1. Write 3-4 concise bullet points with 1-2 sentences each
                2. Use actual company names from context - be specific
                3. Focus on the most critical recent developments
                4. Include specific numbers, dates, and strategic facts
                5. Each bullet should be a standalone strategic insight
                
                Format as bullet points:
                • [Most critical development with specific facts]
                • [Key strategic implication with numbers]
                • [Important competitive move or market change]
                • [Critical risk or opportunity executives must address]
                
                Context: {context}
                
                Write like you're briefing a CEO who has 30 seconds to read this.
            """,
            
            "key_takeaways": """
                Write key takeaways for the right column of a flash report.
                
                CRITICAL REQUIREMENTS:
                1. Write 4-5 concise bullet points with 1-2 sentences each
                2. Use actual company names from context - be specific
                3. Focus on actionable insights and strategic implications
                4. Include specific metrics, trends, and competitive dynamics
                5. Each bullet should provide a clear strategic insight
                
                Format as bullet points:
                • [Key performance metric or trend]
                • [Strategic initiative or investment]
                • [Market position or competitive advantage]
                • [Risk factor or challenge]
                • [Future outlook or prediction]
                
                Context: {context}
                
                Write like you're briefing a strategy team.
            """,
            
            "financial_highlights": """
                Write financial highlights for the bottom section of a flash report.
                
                CRITICAL REQUIREMENTS:
                1. Write 3-4 concise bullet points with 1-2 sentences each
                2. Use actual company names and specific financial figures
                3. Focus on revenue, earnings, and key financial metrics
                4. Include year-over-year comparisons and growth rates
                5. Highlight the most significant financial developments
                
                Format as bullet points:
                • [Revenue performance with specific numbers]
                • [Earnings or profitability metrics]
                • [Key financial ratios or trends]
                • [Cash flow or balance sheet highlights]
                
                Context: {context}
                
                Write like you're briefing a CFO.
            """,
            
            "strategic_insights": """
                Write strategic insights for a flash report - NOT an essay.
                
                CRITICAL REQUIREMENTS:
                1. Write EXACTLY 2 paragraphs (75-90 words each) - NO MORE, NO LESS
                2. Use actual company names from context - be specific
                3. NO essay structure, NO conclusions, NO "In conclusion", NO "To summarize"
                4. Each paragraph tells its own focused story with specific recommendations
                5. Direct, actionable language - avoid generic strategy speak
                6. Provide specific strategic recommendations, not broad observations
                
                Paragraph 1: State the single most critical strategic opportunity they should pursue immediately. Be specific about the opportunity, timeline, and potential impact.
                
                Paragraph 2: Identify the most significant strategic risk they must address. Name the specific threat, timeline, and recommended response.
                
                Context: {context}
                
                Write like you're giving direct recommendations to the CEO.
            """
        }
        
        prompt_text = prompts.get(section_type, prompts["executive_summary"])
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | self.output_parser
        
        return chain.invoke({
            "context": "\n\n---\n\n".join(context)
        })

    def generate_response(self, query: str, context: List[str],
                         intent: Optional[QueryIntent] = None,
                         progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate a response using the appropriate prompt based on intent."""
        try:
            if not intent:
                intent = self.classify_intent(query)
            
            if progress_callback:
                progress_callback(f"Generating {intent.value} response...")
            
            if intent == QueryIntent.SUMMARY:
                prompt = self.summary_prompt
            elif intent == QueryIntent.SPECIFIC_QUESTION:
                prompt = self.qa_prompt
            elif intent == QueryIntent.DATA_EXTRACTION:
                prompt = self.extraction_prompt
            else:
                prompt = self.summary_prompt
            
            chain = prompt | self.llm | self.output_parser
            
            return chain.invoke({
                "query": query,
                "context": "\n\n".join(context)
            })
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
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