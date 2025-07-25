"""
Enhanced conversational query engine with advanced RAG and intent-based retrieval strategies.
"""
from typing import List, Dict, Any, Optional, Tuple, Callable
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
    """Enhanced conversation intent types for sophisticated RAG interactions."""
    # Analysis Intents (require different RAG strategies)
    QUICK_SUMMARY = "quick_summary"           # Use broad context, summarization focus
    DEEP_ANALYSIS = "deep_analysis"           # Use ensemble retrieval, detailed context
    SPECIFIC_QUESTION = "specific_question"    # Use precise retrieval, focused context
    DATA_EXTRACTION = "data_extraction"       # Use filtered retrieval, structured focus
    COMPARISON = "comparison"                 # Use comparative retrieval, multiple contexts
    
    # Conversational Intents (lighter RAG needs)
    GREETING = "greeting"                     # Minimal context, conversational
    CLARIFICATION = "clarification"           # Previous context + targeted retrieval
    FOLLOW_UP = "follow_up"                  # Conversation history + contextual retrieval
    SMALL_TALK = "small_talk"                # No RAG needed
    
    # Strategic Intents (require comprehensive RAG)
    STRATEGIC_INSIGHT = "strategic_insight"   # Use ensemble + filtering, strategic focus
    RECOMMENDATION = "recommendation"         # Use comprehensive retrieval, actionable focus
    RISK_ANALYSIS = "risk_analysis"          # Use filtered retrieval, risk-focused context
    OPPORTUNITY_ANALYSIS = "opportunity_analysis"  # Use broad retrieval, opportunity focus
    COMPETITIVE_ANALYSIS = "competitive_analysis"  # Use comparative + ensemble retrieval
    
    # Financial Intents (require structured RAG)
    FINANCIAL_ANALYSIS = "financial_analysis"     # Use filtered + structured retrieval
    METRICS_EXTRACTION = "metrics_extraction"     # Use precise + structured retrieval
    TREND_ANALYSIS = "trend_analysis"            # Use temporal + comparative retrieval
    
    # Creative Intents (require diverse RAG)
    BRAINSTORMING = "brainstorming"          # Use diverse retrieval, creative prompting
    SCENARIO_PLANNING = "scenario_planning"   # Use comprehensive + hypothetical context
    HYPOTHETICAL = "hypothetical"            # Use diverse + contextual retrieval

class ConversationTone(Enum):
    """Different conversation tones based on user preference."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EXECUTIVE = "executive"

class RAGStrategy(Enum):
    """Different RAG retrieval strategies based on intent."""
    PRECISE = "precise"           # Few, highly relevant chunks
    BROAD = "broad"              # Many chunks, comprehensive coverage
    ENSEMBLE = "ensemble"        # Multiple retrieval methods combined
    FILTERED = "filtered"        # Context filtered by specific criteria
    COMPARATIVE = "comparative"   # Multiple contexts for comparison
    STRUCTURED = "structured"    # Organized, categorized context
    TEMPORAL = "temporal"        # Time-aware context retrieval
    DIVERSE = "diverse"          # Varied perspectives and contexts

class ChatMessage:
    """Represents a single chat message with enhanced metadata."""
    
    def __init__(self, role: str, content: str, intent: Optional[ConversationIntent] = None,
                 timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None,
                 rag_strategy: Optional[RAGStrategy] = None, context_used: int = 0):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.intent = intent
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.rag_strategy = rag_strategy
        self.context_used = context_used
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'intent': self.intent.value if self.intent else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'rag_strategy': self.rag_strategy.value if self.rag_strategy else None,
            'context_used': self.context_used
        }

class ConversationalEngine:
    """Enhanced conversational query engine with advanced RAG strategies and conversation memory."""
    
    def __init__(self, model_name: str = OPENAI_MODEL or "gpt-4", temperature: float = 0.3,
                 timeout: int = 45, conversation_tone: ConversationTone = ConversationTone.PROFESSIONAL):
        """
        Initialize conversational engine with advanced RAG capabilities.
        
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
        
        # Enhanced conversation memory
        self.chat_history: List[ChatMessage] = []
        self.context_memory: Dict[str, Any] = {}
        self.conversation_themes: List[str] = []  # Track conversation topics
        
        # RAG strategy mapping
        self.intent_to_rag_strategy = {
            ConversationIntent.QUICK_SUMMARY: RAGStrategy.BROAD,
            ConversationIntent.DEEP_ANALYSIS: RAGStrategy.ENSEMBLE,
            ConversationIntent.SPECIFIC_QUESTION: RAGStrategy.PRECISE,
            ConversationIntent.DATA_EXTRACTION: RAGStrategy.STRUCTURED,
            ConversationIntent.COMPARISON: RAGStrategy.COMPARATIVE,
            ConversationIntent.STRATEGIC_INSIGHT: RAGStrategy.ENSEMBLE,
            ConversationIntent.RECOMMENDATION: RAGStrategy.FILTERED,
            ConversationIntent.RISK_ANALYSIS: RAGStrategy.FILTERED,
            ConversationIntent.OPPORTUNITY_ANALYSIS: RAGStrategy.BROAD,
            ConversationIntent.COMPETITIVE_ANALYSIS: RAGStrategy.COMPARATIVE,
            ConversationIntent.FINANCIAL_ANALYSIS: RAGStrategy.STRUCTURED,
            ConversationIntent.METRICS_EXTRACTION: RAGStrategy.PRECISE,
            ConversationIntent.TREND_ANALYSIS: RAGStrategy.TEMPORAL,
            ConversationIntent.BRAINSTORMING: RAGStrategy.DIVERSE,
            ConversationIntent.SCENARIO_PLANNING: RAGStrategy.ENSEMBLE,
            ConversationIntent.HYPOTHETICAL: RAGStrategy.DIVERSE,
            ConversationIntent.FOLLOW_UP: RAGStrategy.PRECISE,
            ConversationIntent.CLARIFICATION: RAGStrategy.PRECISE,
        }
        
        # Company context
        self.company_context = """
        Norstella is a global healthcare technology company formed through the combination of multiple industry-leading companies. 
        Norstella provides technology-enabled solutions, analytics, and insights to help pharmaceutical, biotechnology, and medical device companies 
        accelerate the development and commercialization of their products. The company serves as a strategic partner throughout the product lifecycle, 
        from early-stage research and development through commercialization and market access.
        """
        
        # Initialize enhanced prompts
        self._init_enhanced_prompts()
    
    def _init_enhanced_prompts(self):
        """Initialize enhanced prompt templates with conversational RAG focus."""
        
        # Advanced intent classification with RAG strategy selection
        self.intent_prompt = ChatPromptTemplate.from_template("""
            You are an advanced conversational AI intent classifier specialized in business document analysis.
            
            Analyze the user's message and determine the most appropriate intent. Consider:
            - The type of analysis requested
            - The conversational context from chat history
            - The complexity and scope of the question
            - Whether this builds on previous discussion
            
            **Analysis Intents:**
            - quick_summary: User wants a brief, high-level overview
            - deep_analysis: User wants comprehensive, detailed analysis
            - specific_question: User has a targeted, factual question
            - data_extraction: User wants specific data points or metrics
            - comparison: User wants to compare different aspects
            
            **Conversational Intents:**
            - greeting: User is starting conversation or being social
            - clarification: User needs clarification on previous response
            - follow_up: User is building on previous conversation
            - small_talk: User is making casual conversation
            
            **Strategic Intents:**
            - strategic_insight: User wants strategic analysis and recommendations
            - recommendation: User wants specific actionable advice
            - risk_analysis: User wants to understand risks and challenges
            - opportunity_analysis: User wants to identify opportunities
            - competitive_analysis: User wants competitive intelligence
            
            **Financial Intents:**
            - financial_analysis: User wants financial performance analysis
            - metrics_extraction: User wants specific financial/operational metrics
            - trend_analysis: User wants to understand trends over time
            
            **Creative Intents:**
            - brainstorming: User wants to explore ideas and possibilities
            - scenario_planning: User wants to explore different scenarios
            - hypothetical: User is asking "what if" questions
            
            Recent conversation context:
            {chat_history}
            
            User's current message: {query}
            
            Respond with ONLY the intent name (e.g., "deep_analysis").
        """)
        
        # Initialize tone-specific styles
        self._init_conversational_styles()
        
        # Initialize RAG-aware prompts
        self._init_rag_aware_prompts()
    
    def _init_conversational_styles(self):
        """Initialize enhanced conversational styles for each tone."""
        
        self.tone_styles = {
            ConversationTone.PROFESSIONAL: {
                'style': "professional, authoritative, and precise",
                'approach': "structured analysis with clear conclusions",
                'language': "formal business language with industry terminology"
            },
            ConversationTone.FRIENDLY: {
                'style': "warm, approachable, and conversational",
                'approach': "accessible explanations with relatable examples",
                'language': "friendly, clear language that builds rapport"
            },
            ConversationTone.ANALYTICAL: {
                'style': "data-driven, methodical, and thorough",
                'approach': "systematic analysis with detailed evidence",
                'language': "precise, analytical terminology with quantitative focus"
            },
            ConversationTone.CREATIVE: {
                'style': "innovative, exploratory, and imaginative",
                'approach': "creative thinking with alternative perspectives",
                'language': "engaging, dynamic language that inspires ideas"
            },
            ConversationTone.EXECUTIVE: {
                'style': "concise, strategic, and high-level",
                'approach': "executive summary format with key insights",
                'language': "confident, decisive language focused on outcomes"
            }
        }
    
    def _init_rag_aware_prompts(self):
        """Initialize RAG-aware prompt templates for different strategies."""
        
        # Enhanced conversational prompt with RAG strategy awareness
        self.conversational_prompt = ChatPromptTemplate.from_template("""
            You are a {style} AI business analyst having a natural conversation about business documents.
            
            **Your conversational approach:** {approach}
            **Your language style:** {language}
            
            **Context Strategy:** This response uses {rag_strategy} retrieval to provide {context_description}.
            
            **Conversation History:**
            {chat_history}
            
            **Current Context from Documents:**
            {context}
            
            **User's Question:** {query}
            
            **Instructions:**
            - Respond in a {style} manner that feels like a natural conversation
            - Reference the conversation history when relevant to show continuity
            - Use the document context to provide accurate, evidence-based responses
            - Acknowledge the user's question directly and personally
            - {specific_instructions}
            
            Provide a conversational response that builds on our discussion:
        """)
        
        # Context descriptions for different RAG strategies
        self.rag_context_descriptions = {
            RAGStrategy.PRECISE: "focused, highly relevant information",
            RAGStrategy.BROAD: "comprehensive coverage from multiple perspectives",
            RAGStrategy.ENSEMBLE: "combined insights from multiple retrieval methods",
            RAGStrategy.FILTERED: "targeted information filtered for specific criteria",
            RAGStrategy.COMPARATIVE: "comparative analysis across different aspects",
            RAGStrategy.STRUCTURED: "organized, categorized information",
            RAGStrategy.TEMPORAL: "time-aware insights showing progression",
            RAGStrategy.DIVERSE: "varied perspectives and creative angles"
        }
        
        # Specific instructions for different intents
        self.intent_instructions = {
            ConversationIntent.QUICK_SUMMARY: "Provide a concise overview that captures the key points without overwhelming detail",
            ConversationIntent.DEEP_ANALYSIS: "Dive deep into the details with comprehensive analysis and multiple perspectives",
            ConversationIntent.SPECIFIC_QUESTION: "Answer directly and precisely while providing supporting context",
            ConversationIntent.STRATEGIC_INSIGHT: "Focus on strategic implications and actionable recommendations",
            ConversationIntent.FINANCIAL_ANALYSIS: "Emphasize financial metrics, trends, and business impact",
            ConversationIntent.FOLLOW_UP: "Build naturally on our previous discussion while adding new insights",
            ConversationIntent.CLARIFICATION: "Clarify the previous point with additional detail and examples",
            ConversationIntent.BRAINSTORMING: "Explore creative possibilities and encourage innovative thinking"
        }
    
    def get_rag_strategy(self, intent: ConversationIntent) -> RAGStrategy:
        """Get the appropriate RAG strategy for the given intent."""
        return self.intent_to_rag_strategy.get(intent, RAGStrategy.PRECISE)
    
    def apply_rag_strategy(self, vectorstore, query: str, intent: ConversationIntent, 
                         rag_strategy: RAGStrategy) -> List[str]:
        """Apply the appropriate RAG strategy to retrieve context."""
        
        try:
            if rag_strategy == RAGStrategy.PRECISE:
                # Focused retrieval with fewer, highly relevant chunks
                context = vectorstore.query_collection(query, k=4)
                
            elif rag_strategy == RAGStrategy.BROAD:
                # Comprehensive retrieval with more chunks
                context = vectorstore.query_collection(query, k=8)
                
            elif rag_strategy == RAGStrategy.ENSEMBLE:
                # Combine multiple retrieval approaches
                # Primary query
                primary_context = vectorstore.query_collection(query, k=5)
                # Related terms query
                expanded_query = f"{query} analysis insights strategic implications"
                secondary_context = vectorstore.query_collection(expanded_query, k=3)
                context = primary_context + secondary_context
                
            elif rag_strategy == RAGStrategy.FILTERED:
                # Filter context based on intent-specific keywords
                filter_keywords = self._get_filter_keywords(intent)
                filtered_query = f"{query} {' '.join(filter_keywords)}"
                context = vectorstore.query_collection(filtered_query, k=6)
                
            elif rag_strategy == RAGStrategy.COMPARATIVE:
                # Retrieve context for comparison
                context = vectorstore.query_collection(query, k=6)
                # Add comparative terms
                comparative_query = f"{query} compare comparison versus differences"
                comparative_context = vectorstore.query_collection(comparative_query, k=3)
                context.extend(comparative_context)
                
            elif rag_strategy == RAGStrategy.STRUCTURED:
                # Organize context by categories
                context = vectorstore.query_collection(query, k=6)
                # Could add categorization logic here in future
                
            elif rag_strategy == RAGStrategy.TEMPORAL:
                # Time-aware retrieval (enhance with temporal keywords)
                temporal_query = f"{query} trends changes over time period timeline"
                context = vectorstore.query_collection(temporal_query, k=6)
                
            elif rag_strategy == RAGStrategy.DIVERSE:
                # Diverse perspective retrieval
                base_context = vectorstore.query_collection(query, k=4)
                # Add alternative perspective queries
                alt_query = f"{query} alternative perspective different angle"
                alt_context = vectorstore.query_collection(alt_query, k=3)
                context = base_context + alt_context
                
            else:
                # Default to precise strategy
                context = vectorstore.query_collection(query, k=5)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_context = []
            for item in context:
                if item not in seen:
                    seen.add(item)
                    unique_context.append(item)
            
            return unique_context[:10]  # Limit to top 10 for performance
            
        except Exception as e:
            logger.error(f"RAG strategy {rag_strategy} failed: {str(e)}")
            # Fallback to simple retrieval
            return vectorstore.query_collection(query, k=5)
    
    def _get_filter_keywords(self, intent: ConversationIntent) -> List[str]:
        """Get filter keywords based on intent."""
        
        filter_map = {
            ConversationIntent.FINANCIAL_ANALYSIS: ["revenue", "profit", "cost", "financial", "earnings", "budget"],
            ConversationIntent.RISK_ANALYSIS: ["risk", "challenge", "threat", "problem", "issue", "concern"],
            ConversationIntent.OPPORTUNITY_ANALYSIS: ["opportunity", "growth", "potential", "advantage", "benefit"],
            ConversationIntent.STRATEGIC_INSIGHT: ["strategy", "strategic", "plan", "direction", "goal", "objective"],
            ConversationIntent.COMPETITIVE_ANALYSIS: ["competitor", "competitive", "market", "competition", "rival"],
            ConversationIntent.METRICS_EXTRACTION: ["metric", "KPI", "measurement", "data", "number", "percentage"]
        }
        
        return filter_map.get(intent, ["analysis", "insight", "information"])
    
    def classify_intent(self, query: str) -> ConversationIntent:
        """Enhanced intent classification with conversation awareness."""
        
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
                classified_intent = ConversationIntent(intent_str)
                
                # Update conversation themes
                self._update_conversation_themes(classified_intent)
                
                return classified_intent
            except ValueError:
                # Enhanced fallback logic based on keywords
                return self._fallback_intent_classification(query)
                
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            return ConversationIntent.SPECIFIC_QUESTION
    
    def _fallback_intent_classification(self, query: str) -> ConversationIntent:
        """Enhanced fallback intent classification using keyword analysis."""
        
        query_lower = query.lower()
        
        # Strategic keywords
        if any(word in query_lower for word in ['strategy', 'strategic', 'recommend', 'should', 'advice']):
            return ConversationIntent.STRATEGIC_INSIGHT
        
        # Financial keywords
        if any(word in query_lower for word in ['financial', 'revenue', 'profit', 'cost', 'budget', 'money']):
            return ConversationIntent.FINANCIAL_ANALYSIS
        
        # Summary keywords
        if any(word in query_lower for word in ['summary', 'overview', 'highlights', 'key points', 'brief']):
            return ConversationIntent.QUICK_SUMMARY
        
        # Analysis keywords
        if any(word in query_lower for word in ['analyze', 'analysis', 'deep', 'detailed', 'comprehensive']):
            return ConversationIntent.DEEP_ANALYSIS
        
        # Question words
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return ConversationIntent.SPECIFIC_QUESTION
        
        # Default
        return ConversationIntent.SPECIFIC_QUESTION
    
    def _update_conversation_themes(self, intent: ConversationIntent):
        """Update conversation themes based on intents."""
        
        theme_map = {
            ConversationIntent.FINANCIAL_ANALYSIS: "Financial Performance",
            ConversationIntent.STRATEGIC_INSIGHT: "Strategic Planning",
            ConversationIntent.RISK_ANALYSIS: "Risk Management",
            ConversationIntent.COMPETITIVE_ANALYSIS: "Competitive Intelligence",
            ConversationIntent.DEEP_ANALYSIS: "Detailed Analysis"
        }
        
        theme = theme_map.get(intent)
        if theme and theme not in self.conversation_themes:
            self.conversation_themes.append(theme)
            if len(self.conversation_themes) > 5:  # Keep recent themes
                self.conversation_themes.pop(0)
    
    def _format_chat_history(self, limit: int = 10) -> str:
        """Format recent chat history with enhanced context."""
        
        if not self.chat_history:
            return "No previous conversation."
        
        recent_messages = self.chat_history[-limit:]
        formatted = []
        
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "AI Assistant"
            intent_info = f" ({msg.intent.value})" if msg.intent else ""
            rag_info = f" [using {msg.rag_strategy.value} retrieval]" if msg.rag_strategy else ""
            formatted.append(f"{role}{intent_info}: {msg.content}{rag_info}")
        
        return "\n".join(formatted)
    
    def generate_response(self, query: str, vectorstore, 
                         progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, ConversationIntent]:
        """
        Generate enhanced conversational response with adaptive RAG strategies.
        
        Args:
            query: User's message
            vectorstore: Vector store for RAG context retrieval
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
            rag_strategy = self.get_rag_strategy(intent)
            
            logger.info(f"Detected intent: {intent.value}, RAG strategy: {rag_strategy.value}")
            
            # Apply RAG strategy
            if progress_callback:
                progress_callback(f"🔍 Gathering context using {rag_strategy.value} retrieval...")
            
            context = self.apply_rag_strategy(vectorstore, query, intent, rag_strategy)
            
            # Generate conversational response
            if progress_callback:
                progress_callback("✍️ Crafting conversational response...")
            
            response = self._generate_conversational_response(query, context, intent, rag_strategy)
            
            # Store in enhanced conversation history
            self._add_to_history(query, response, intent, rag_strategy, len(context))
            
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            return response, intent
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise
    
    def _generate_conversational_response(self, query: str, context: List[str], 
                                        intent: ConversationIntent, rag_strategy: RAGStrategy) -> str:
        """Generate conversational response using enhanced RAG context."""
        
        # Get tone-specific style
        tone_style = self.tone_styles[self.conversation_tone]
        
        # Prepare context
        formatted_context = self._prepare_enhanced_context(context)
        chat_history = self._format_chat_history(limit=6)
        
        # Get context description and specific instructions
        context_description = self.rag_context_descriptions.get(rag_strategy, "relevant information")
        specific_instructions = self.intent_instructions.get(intent, "Provide a helpful and informative response")
        
        chain = self.conversational_prompt | self.llm | self.output_parser  # type: ignore
        response = chain.invoke({
            "style": tone_style['style'],
            "approach": tone_style['approach'],
            "language": tone_style['language'],
            "rag_strategy": rag_strategy.value,
            "context_description": context_description,
            "chat_history": chat_history,
            "context": formatted_context,
            "query": query,
            "specific_instructions": specific_instructions
        })
        
        return response
    
    def _prepare_enhanced_context(self, context: List[str]) -> str:
        """Prepare and format context with enhanced organization."""
        
        if not context:
            return "No specific document context available for this query."
        
        # Add company context
        formatted_context = f"**Company Background:**\n{self.company_context}\n\n"
        
        # Add conversation themes if available
        if self.conversation_themes:
            formatted_context += f"**Conversation Themes:** {', '.join(self.conversation_themes)}\n\n"
        
        # Add document context
        formatted_context += "**Relevant Document Context:**\n"
        for i, chunk in enumerate(context, 1):
            formatted_context += f"\n--- Context {i} ---\n{chunk}\n"
        
        return formatted_context
    
    def _add_to_history(self, user_message: str, assistant_response: str, 
                       intent: ConversationIntent, rag_strategy: RAGStrategy, context_count: int):
        """Add messages to enhanced conversation history."""
        
        # Add user message
        self.chat_history.append(ChatMessage(
            role="user",
            content=user_message,
            intent=intent
        ))
        
        # Add assistant response with enhanced metadata
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=assistant_response,
            intent=intent,
            rag_strategy=rag_strategy,
            context_used=context_count,
            metadata={
                "intent": intent.value,
                "rag_strategy": rag_strategy.value,
                "context_count": context_count,
                "conversation_themes": self.conversation_themes.copy()
            }
        ))
        
        # Keep history manageable (last 20 messages)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
    
    def get_conversation_summary(self) -> str:
        """Get an enhanced summary of the conversation."""
        
        if not self.chat_history:
            return "No conversation history yet."
        
        if self.conversation_themes:
            return f"We've discussed: {', '.join(self.conversation_themes)}"
        else:
            intents = [msg.intent.value.replace("_", " ").title() 
                      for msg in self.chat_history if msg.role == "user" and msg.intent]
            unique_intents = list(dict.fromkeys(intents))  # Preserve order, remove duplicates
            return f"Topics covered: {', '.join(unique_intents[:5])}"
    
    def clear_history(self):
        """Clear conversation history and themes."""
        self.chat_history = []
        self.context_memory = {}
        self.conversation_themes = []
        logger.info("Enhanced conversation history cleared")
    
    def set_tone(self, tone: ConversationTone):
        """Change conversation tone dynamically."""
        self.conversation_tone = tone
        logger.info(f"Conversation tone changed to: {tone.value}")

    def get_suggested_followups(self) -> List[str]:
        """Generate contextual follow-up suggestions based on conversation and themes."""
        
        if not self.chat_history:
            return [
                "What are the key strategic insights?",
                "Can you analyze the financial performance?",
                "What are the main risks and opportunities?"
            ]
        
        # Get suggestions based on themes and recent intents
        last_intent = self.chat_history[-1].intent if self.chat_history else None
        
        theme_suggestions = {
            "Financial Performance": [
                "What's driving these financial trends?",
                "How do these metrics compare to industry benchmarks?",
                "What are the key financial risks?"
            ],
            "Strategic Planning": [
                "What are the implementation priorities?",
                "How should we measure success?",
                "What are the potential roadblocks?"
            ],
            "Competitive Intelligence": [
                "How do we differentiate from competitors?",
                "What are our competitive advantages?",
                "Where are the market gaps?"
            ]
        }
        
        # Return theme-based suggestions if available
        if self.conversation_themes:
            latest_theme = self.conversation_themes[-1]
            return theme_suggestions.get(latest_theme, [
                "Can you dive deeper into this analysis?",
                "What are the strategic implications?",
                "What should we focus on next?"
            ])
        
        # Fallback suggestions
        return [
            "What are the key takeaways?",
            "How does this impact our strategy?",
            "What are the next steps?"
        ] 