"""
Enhanced conversational query engine with advanced intent detection and chat capabilities.
"""
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import time
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document as LangChainDocument

from src.config import OPENAI_MODEL
from src.llm.query_engine import QueryTimeoutError

# Set up logging
logger = logging.getLogger(__name__)

class ConversationIntent(Enum):
    """Enhanced conversation intent types for sophisticated interactions."""
    # Analysis Intents
    QUICK_SUMMARY = "quick_summary"
    DEEP_ANALYSIS = "deep_analysis"
    SPECIFIC_QUESTION = "specific_question"
    DATA_EXTRACTION = "data_extraction"
    COMPARISON = "comparison"
    
    # Conversational Intents
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    SMALL_TALK = "small_talk"
    
    # Strategic Intents
    STRATEGIC_INSIGHT = "strategic_insight"
    RECOMMENDATION = "recommendation"
    RISK_ANALYSIS = "risk_analysis"
    OPPORTUNITY_ANALYSIS = "opportunity_analysis"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    
    # Financial Intents
    FINANCIAL_ANALYSIS = "financial_analysis"
    METRICS_EXTRACTION = "metrics_extraction"
    TREND_ANALYSIS = "trend_analysis"
    
    # Creative Intents
    BRAINSTORMING = "brainstorming"
    SCENARIO_PLANNING = "scenario_planning"
    HYPOTHETICAL = "hypothetical"

class ConversationTone(Enum):
    """Different conversation tones based on user preference."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EXECUTIVE = "executive"

class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(self, role: str, content: str, intent: Optional[ConversationIntent] = None,
                 timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.intent = intent
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'intent': self.intent.value if self.intent else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class ConversationalEngine:
    """Enhanced conversational query engine with memory and advanced intent detection."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.3,
                 timeout: int = 45, conversation_tone: ConversationTone = ConversationTone.PROFESSIONAL):
        """
        Initialize conversational engine.
        
        Args:
            model_name: OpenAI model to use
            temperature: Creativity level (0.0-1.0)
            timeout: Timeout in seconds
            conversation_tone: Tone of conversation
        """
        self.llm = ChatOpenAI(  # type: ignore
            model=model_name,
            temperature=temperature,
            max_retries=3
        )
        self.output_parser = StrOutputParser()
        self.timeout = timeout
        self.conversation_tone = conversation_tone
        
        # Conversation memory
        self.chat_history: List[ChatMessage] = []
        self.context_memory: Dict[str, Any] = {}
        
        # Company context
        self.company_context = """
        Norstella is a global healthcare technology company formed through the combination of multiple industry-leading companies. 
        Norstella provides technology-enabled solutions, analytics, and insights to help pharmaceutical, biotechnology, and medical device companies 
        accelerate the development and commercialization of their products. The company serves as a strategic partner throughout the product lifecycle, 
        from early-stage research and development through commercialization and market access.
        """
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize sophisticated prompt templates for different intents and tones."""
        
        # Advanced intent classification
        self.intent_prompt = ChatPromptTemplate.from_template("""
            You are an advanced intent classifier for a conversational AI assistant specializing in business intelligence and document analysis.
            
            Analyze the user's message and determine the most appropriate intent from these options:
            
            **Analysis Intents:**
            - quick_summary: User wants a brief overview or summary
            - deep_analysis: User wants detailed, comprehensive analysis
            - specific_question: User has a specific factual question
            - data_extraction: User wants specific data points or metrics
            - comparison: User wants to compare things
            
            **Conversational Intents:**
            - greeting: User is saying hello or starting conversation
            - clarification: User is asking for clarification or explanation
            - follow_up: User is following up on a previous response
            - small_talk: User is making casual conversation
            
            **Strategic Intents:**
            - strategic_insight: User wants strategic recommendations or insights
            - recommendation: User wants specific recommendations or advice
            - risk_analysis: User wants to understand risks or challenges
            - opportunity_analysis: User wants to identify opportunities
            - competitive_analysis: User wants competitive intelligence
            
            **Financial Intents:**
            - financial_analysis: User wants financial performance analysis
            - metrics_extraction: User wants specific financial or operational metrics
            - trend_analysis: User wants to understand trends over time
            
            **Creative Intents:**
            - brainstorming: User wants to brainstorm ideas or solutions
            - scenario_planning: User wants to explore different scenarios
            - hypothetical: User is asking "what if" questions
            
            Consider the conversation history and context. Look for keywords, question types, and user patterns.
            
            Recent conversation history:
            {chat_history}
            
            Current message: {query}
            
            Respond with ONLY the intent name (e.g., "deep_analysis").
        """)
        
        # Conversational prompts based on tone
        self._init_tone_prompts()
        
        # Intent-specific prompts
        self._init_intent_prompts()
    
    def _init_tone_prompts(self):
        """Initialize tone-specific conversation styles."""
        
        tone_styles = {
            ConversationTone.PROFESSIONAL: "professional, precise, and authoritative",
            ConversationTone.FRIENDLY: "warm, approachable, and conversational",
            ConversationTone.ANALYTICAL: "data-driven, methodical, and thorough",
            ConversationTone.CREATIVE: "innovative, exploratory, and imaginative",
            ConversationTone.EXECUTIVE: "concise, strategic, and high-level"
        }
        
        self.tone_style = tone_styles[self.conversation_tone]
    
    def _init_intent_prompts(self):
        """Initialize prompt templates for each intent type."""
        
        # Greeting prompt
        self.greeting_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} AI assistant specializing in business intelligence and document analysis.
            
            The user is greeting you or starting a conversation. Respond in a {tone_style} manner and:
            1. Acknowledge their greeting warmly
            2. Briefly introduce your capabilities (document analysis, insights, recommendations)
            3. Ask how you can help them today
            4. Keep it concise but engaging
            
            Chat history: {chat_history}
            User message: {query}
            
            Respond as a helpful AI assistant.
        """)
        
        # Quick summary prompt
        self.quick_summary_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} business analyst. Provide a quick, concise summary based on the context.
            
            REQUIREMENTS:
            - Write 2-3 focused paragraphs
            - Highlight only the most important points
            - Use bullet points for key takeaways if helpful
            - Be direct and actionable
            - Reference specific companies, numbers, and dates when available
            
            Chat history: {chat_history}
            Context: {context}
            User question: {query}
            
            Provide a concise summary that directly addresses their question.
        """)
        
        # Deep analysis prompt
        self.deep_analysis_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} senior business analyst. Provide comprehensive, detailed analysis.
            
            REQUIREMENTS:
            - Write 4-6 detailed paragraphs
            - Include multiple perspectives and implications
            - Provide evidence-based insights
            - Reference specific data points, trends, and examples
            - Include strategic implications and recommendations
            - Use structured analysis (situation, implications, recommendations)
            
            Structure your response:
            **Current Situation**
            **Key Insights & Analysis** 
            **Strategic Implications**
            **Recommendations**
            
            Chat history: {chat_history}
            Context: {context}
            User question: {query}
            
            Provide thorough, expert-level analysis.
        """)
        
        # Strategic insight prompt
        self.strategic_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} strategic consultant. Provide strategic insights and recommendations.
            
            REQUIREMENTS:
            - Focus on strategic implications and competitive advantages
            - Identify opportunities, risks, and key success factors
            - Provide actionable recommendations
            - Consider market dynamics and competitive landscape
            - Think long-term and big picture
            
            Chat history: {chat_history}
            Context: {context}
            User question: {query}
            
            Provide strategic insights that would be valuable to executives and decision-makers.
        """)
        
        # Financial analysis prompt
        self.financial_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} financial analyst. Focus on financial performance and metrics.
            
            REQUIREMENTS:
            - Extract and analyze key financial metrics
            - Identify trends, ratios, and performance indicators
            - Compare to industry standards where possible
            - Highlight significant changes or anomalies
            - Provide financial health assessment
            
            Chat history: {chat_history}
            Context: {context}
            User question: {query}
            
            Provide detailed financial analysis and insights.
        """)
        
        # Follow-up prompt
        self.followup_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} AI assistant. The user is following up on a previous conversation.
            
            Consider the conversation history and provide:
            - Clarification or additional details if requested
            - Related insights that might be helpful
            - Deeper analysis if they want more information
            - Connections to other relevant topics
            
            Chat history: {chat_history}
            Context: {context}
            User follow-up: {query}
            
            Build on the previous conversation naturally and helpfully.
        """)
        
        # Brainstorming prompt
        self.brainstorm_prompt = ChatPromptTemplate.from_template("""
            You are a {tone_style} innovation consultant. Help brainstorm creative solutions and ideas.
            
            REQUIREMENTS:
            - Generate multiple creative options or approaches
            - Think outside the box while staying practical
            - Build on the available information
            - Encourage exploration of possibilities
            - Provide structured thinking frameworks when helpful
            
            Chat history: {chat_history}
            Context: {context}
            User request: {query}
            
            Help them explore creative solutions and innovative approaches.
        """)
    
    def classify_intent(self, query: str) -> ConversationIntent:
        """Classify user intent using conversation history and advanced detection."""
        
        try:
            # Get recent chat history for context
            recent_history = self._format_chat_history(limit=5)
            
            chain = self.intent_prompt | self.llm | self.output_parser
            result = chain.invoke({
                "query": query,
                "chat_history": recent_history
            })
            
            # Clean and validate result
            intent_str = result.strip().lower()
            try:
                return ConversationIntent(intent_str)
            except ValueError:
                # Fallback to specific_question if intent not recognized
                logger.warning(f"Unknown intent '{intent_str}', defaulting to specific_question")
                return ConversationIntent.SPECIFIC_QUESTION
                
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            return ConversationIntent.SPECIFIC_QUESTION
    
    def _format_chat_history(self, limit: int = 10) -> str:
        """Format recent chat history for context."""
        
        if not self.chat_history:
            return "No previous conversation."
        
        recent_messages = self.chat_history[-limit:]
        formatted = []
        
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def generate_response(self, query: str, context: List[str], 
                         progress_callback: Optional[callable] = None) -> Tuple[str, ConversationIntent]:
        """
        Generate conversational response with intent detection.
        
        Args:
            query: User's message
            context: RAG context from documents
            progress_callback: Progress update function
            
        Returns:
            Tuple of (response, detected_intent)
        """
        start_time = time.time()
        
        try:
            # Classify intent
            if progress_callback:
                progress_callback("🧠 Understanding your request...")
            
            intent = self.classify_intent(query)
            logger.info(f"Detected intent: {intent.value}")
            
            # Select appropriate prompt
            if progress_callback:
                progress_callback("🎯 Choosing response style...")
            
            prompt = self._select_prompt(intent)
            
            # Prepare context
            formatted_context = self._prepare_context(context)
            chat_history = self._format_chat_history()
            
            # Generate response
            if progress_callback:
                progress_callback("✍️ Generating response...")
            
            chain = prompt | self.llm | self.output_parser  # type: ignore
            response = chain.invoke({
                "query": query,
                "context": formatted_context,
                "chat_history": chat_history,
                "tone_style": self.tone_style
            })
            
            # Store in conversation history
            self._add_to_history(query, response, intent)
            
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            return response, intent
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise
    
    def _select_prompt(self, intent: ConversationIntent) -> ChatPromptTemplate:
        """Select the appropriate prompt template based on intent."""
        
        prompt_map = {
            ConversationIntent.GREETING: self.greeting_prompt,
            ConversationIntent.QUICK_SUMMARY: self.quick_summary_prompt,
            ConversationIntent.DEEP_ANALYSIS: self.deep_analysis_prompt,
            ConversationIntent.STRATEGIC_INSIGHT: self.strategic_prompt,
            ConversationIntent.RECOMMENDATION: self.strategic_prompt,
            ConversationIntent.FINANCIAL_ANALYSIS: self.financial_prompt,
            ConversationIntent.METRICS_EXTRACTION: self.financial_prompt,
            ConversationIntent.FOLLOW_UP: self.followup_prompt,
            ConversationIntent.CLARIFICATION: self.followup_prompt,
            ConversationIntent.BRAINSTORMING: self.brainstorm_prompt,
            ConversationIntent.SCENARIO_PLANNING: self.brainstorm_prompt,
        }
        
        # Default to deep analysis for most analytical intents
        return prompt_map.get(intent, self.deep_analysis_prompt)
    
    def _prepare_context(self, context: List[str]) -> str:
        """Prepare and format context for the prompt."""
        
        if not context:
            return "No specific document context available."
        
        # Add company context
        formatted_context = self.company_context + "\n\n---\n\n"
        formatted_context += "\n\n---\n\n".join(context)
        
        return formatted_context
    
    def _add_to_history(self, user_message: str, assistant_response: str, intent: ConversationIntent):
        """Add messages to conversation history."""
        
        # Add user message
        self.chat_history.append(ChatMessage(
            role="user",
            content=user_message,
            intent=intent
        ))
        
        # Add assistant response
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=assistant_response,
            metadata={"intent": intent.value}
        ))
        
        # Keep history manageable (last 20 messages)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far."""
        
        if not self.chat_history:
            return "No conversation history yet."
        
        topics = []
        for msg in self.chat_history:
            if msg.role == "user" and msg.intent:
                topics.append(msg.intent.value.replace("_", " ").title())
        
        return f"We've discussed: {', '.join(set(topics))}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
        self.context_memory = {}
        logger.info("Conversation history cleared")
    
    def set_tone(self, tone: ConversationTone):
        """Change conversation tone dynamically."""
        self.conversation_tone = tone
        self._init_tone_prompts()
        logger.info(f"Conversation tone changed to: {tone.value}")

    def get_suggested_followups(self) -> List[str]:
        """Generate suggested follow-up questions based on conversation."""
        
        if not self.chat_history:
            return [
                "Can you give me a quick summary of the key points?",
                "What are the most important strategic insights?",
                "How does this compare to industry trends?"
            ]
        
        last_intent = self.chat_history[-1].intent if self.chat_history else None
        
        followup_suggestions = {
            ConversationIntent.QUICK_SUMMARY: [
                "Can you provide more detailed analysis?",
                "What are the strategic implications?",
                "What should we be most concerned about?"
            ],
            ConversationIntent.DEEP_ANALYSIS: [
                "What are the key action items?",
                "How does this compare to competitors?",
                "What are the biggest risks?"
            ],
            ConversationIntent.FINANCIAL_ANALYSIS: [
                "What's driving these financial trends?",
                "How does this compare to previous periods?",
                "What should investors be watching?"
            ]
        }
        
        default_suggestions = [
            "Can you dive deeper into this topic?",
            "What are the implications?",
            "What should we focus on next?"
        ]
        
        if last_intent:
            return followup_suggestions.get(last_intent, default_suggestions)
        else:
            return default_suggestions 