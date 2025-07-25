"""
Enhanced conversational analysis section for Streamlit app.
Transforms the basic analysis tab into an intelligent chatbot interface.
"""
import streamlit as st
import logging
from typing import List, Dict, Any
from datetime import datetime

from src.llm.conversational_engine import ConversationalEngine, ConversationTone, ConversationIntent

logger = logging.getLogger(__name__)

def init_conversational_analysis():
    """Initialize conversational analysis in session state."""
    
    if 'conv_engine' not in st.session_state:
        # Get user preference for conversation tone
        tone = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL)
        st.session_state.conv_engine = ConversationalEngine(
            temperature=0.3,
            conversation_tone=tone
        )
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = []

def render_chat_message(message: Dict[str, Any], is_user: bool = True):
    """Render a single chat message with enhanced styling."""
    
    if is_user:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.write(message['content'])
            if 'timestamp' in message:
                st.caption(f"_{message['timestamp']}_")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message['content'])
            
            # Show detected intent if available
            if 'intent' in message and message['intent']:
                intent_display = message['intent'].replace('_', ' ').title()
                st.caption(f"💡 _{intent_display}_ • {message.get('timestamp', '')}")

def render_conversation_controls():
    """Render conversation control buttons and settings."""
    
    with st.sidebar:
        st.subheader("💬 Conversation Settings")
        
        # Conversation tone selector
        tone_options = {
            "Professional": ConversationTone.PROFESSIONAL,
            "Friendly": ConversationTone.FRIENDLY, 
            "Analytical": ConversationTone.ANALYTICAL,
            "Creative": ConversationTone.CREATIVE,
            "Executive": ConversationTone.EXECUTIVE
        }
        
        current_tone = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL)
        selected_tone = st.selectbox(
            "Conversation Style",
            options=list(tone_options.keys()),
            index=list(tone_options.values()).index(current_tone),
            help="Choose how you'd like the AI to communicate"
        )
        
        # Update tone if changed
        new_tone = tone_options[selected_tone]
        if new_tone != st.session_state.get('conversation_tone'):
            st.session_state.conversation_tone = new_tone
            if 'conv_engine' in st.session_state:
                st.session_state.conv_engine.set_tone(new_tone)
            st.rerun()
        
        st.divider()
        
        # Conversation stats
        if st.session_state.chat_messages:
            st.write("📊 **Conversation Stats**")
            message_count = len(st.session_state.chat_messages)
            st.metric("Messages", message_count)
            
            # Show conversation summary
            if 'conv_engine' in st.session_state:
                summary = st.session_state.conv_engine.get_conversation_summary()
                st.caption(summary)
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Chat", use_container_width=True, help="Start a fresh conversation"):
                st.session_state.chat_messages = []
                if 'conv_engine' in st.session_state:
                    st.session_state.conv_engine.clear_history()
                st.rerun()
        
        with col2:
            if st.button("📥 Export", use_container_width=True, help="Export conversation"):
                export_conversation()

def render_suggested_prompts():
    """Render contextual suggested prompts."""
    
    if not st.session_state.get('documents_loaded', False):
        return
    
    # Get suggestions based on conversation
    suggestions = []
    if 'conv_engine' in st.session_state and st.session_state.chat_messages:
        suggestions = st.session_state.conv_engine.get_suggested_followups()
    else:
        # Default suggestions for new conversations
        suggestions = [
            "Give me a quick summary of the key highlights",
            "What are the most significant strategic insights?", 
            "Analyze the financial performance and trends",
            "What are the biggest opportunities and risks?",
            "Compare this to industry benchmarks"
        ]
    
    if suggestions:
        st.write("💡 **Suggested Questions:**")
        cols = st.columns(min(len(suggestions), 3))
        
        for i, suggestion in enumerate(suggestions[:6]):  # Limit to 6 suggestions
            col_idx = i % len(cols)
            with cols[col_idx]:
                if st.button(
                    suggestion, 
                    key=f"suggestion_{i}",
                    use_container_width=True,
                    help="Click to use this question"
                ):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

def process_user_message(user_input: str) -> Dict[str, Any]:
    """Process user message and generate AI response."""
    
    try:
        # Get RAG context for the query
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            with st.status("Processing your message...", expanded=False) as status:
                status.write("🔍 Finding relevant information...")
                
                # Get context from vectorstore
                vs = st.session_state.vectorstore
                context = vs.query_collection(user_input, k=8)  # Get more context for conversation
                
                status.write("🧠 Understanding your request...")
                
                # Generate response using conversational engine
                engine = st.session_state.conv_engine
                response, intent = engine.generate_response(
                    query=user_input,
                    context=context,
                    progress_callback=lambda msg: status.write(msg)
                )
                
                status.write("✅ Response ready!")
        
        progress_placeholder.empty()
        
        return {
            'content': response,
            'intent': intent.value,
            'timestamp': datetime.now().strftime("%H:%M"),
            'context_used': len(context)
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {
            'content': f"I encountered an error processing your message: {str(e)}. Please try rephrasing your question.",
            'intent': 'error',
            'timestamp': datetime.now().strftime("%H:%M"),
            'context_used': 0
        }

def export_conversation():
    """Export conversation to downloadable format."""
    
    if not st.session_state.chat_messages:
        st.warning("No conversation to export")
        return
    
    # Format conversation for export
    export_text = f"# AI Flash Report Conversation\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    for i, msg in enumerate(st.session_state.chat_messages):
        role = "👤 You" if msg.get('is_user', True) else "🤖 AI Assistant"
        content = msg['content']
        timestamp = msg.get('timestamp', '')
        
        export_text += f"## {role} ({timestamp})\n{content}\n\n"
    
    # Offer download
    st.download_button(
        label="📄 Download Conversation",
        data=export_text,
        file_name=f"flash_report_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )

def enhanced_analysis_section():
    """Main enhanced analysis section with conversational interface."""
    
    # Check if documents are loaded
    if not st.session_state.get('documents_loaded', False):
        st.info("⚠️ Please upload and process documents first in the Upload tab")
        return
    
    # Initialize conversational components
    init_conversational_analysis()
    
    # Header
    st.subheader("💬 AI Assistant Chat")
    st.caption("Ask questions about your documents in natural language. I can provide summaries, deep analysis, strategic insights, and more!")
    
    # Render conversation controls in sidebar
    render_conversation_controls()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        if st.session_state.chat_messages:
            for message in st.session_state.chat_messages:
                render_chat_message(message, is_user=message.get('is_user', True))
        else:
            # Welcome message for new conversations
            with st.chat_message("assistant", avatar="🤖"):
                tone_name = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL).value
                st.write(f"""
                Hello! I'm your AI business analyst assistant. I'm ready to help you analyze your documents in a {tone_name} manner.
                
                I can help you with:
                - **Quick summaries** of key points
                - **Deep analysis** with strategic insights  
                - **Financial analysis** and metrics extraction
                - **Competitive intelligence** and market insights
                - **Strategic recommendations** and next steps
                
                What would you like to explore first?
                """)
    
    # Suggested prompts
    with st.expander("💡 Suggested Questions", expanded=not bool(st.session_state.chat_messages)):
        render_suggested_prompts()
    
    # Chat input
    chat_input_container = st.container()
    
    with chat_input_container:
        # Check for suggested query
        default_query = st.session_state.pop('suggested_query', '')
        
        # Chat input
        user_input = st.chat_input(
            "Ask me anything about your documents...",
            key="chat_input"
        )
        
        # Use suggested query if available
        if default_query and not user_input:
            user_input = default_query
        
        # Process user input
        if user_input:
            # Add user message to chat
            user_message = {
                'content': user_input,
                'is_user': True,
                'timestamp': datetime.now().strftime("%H:%M")
            }
            st.session_state.chat_messages.append(user_message)
            
            # Generate AI response
            ai_response = process_user_message(user_input)
            ai_response['is_user'] = False
            st.session_state.chat_messages.append(ai_response)
            
            # Rerun to update the display
            st.rerun()
    
    # Analytics section (collapsible)
    if st.session_state.chat_messages:
        with st.expander("📊 Conversation Analytics", expanded=False):
            render_conversation_analytics()

def render_conversation_analytics():
    """Render analytics about the conversation."""
    
    if not st.session_state.chat_messages:
        return
    
    # Calculate stats
    total_messages = len(st.session_state.chat_messages)
    user_messages = [msg for msg in st.session_state.chat_messages if msg.get('is_user', True)]
    ai_messages = [msg for msg in st.session_state.chat_messages if not msg.get('is_user', True)]
    
    # Intent analysis
    intents = [msg.get('intent') for msg in ai_messages if msg.get('intent')]
    intent_counts = {}
    for intent in intents:
        if intent:
            display_intent = intent.replace('_', ' ').title()
            intent_counts[display_intent] = intent_counts.get(display_intent, 0) + 1
    
    # Display analytics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", total_messages)
        st.metric("Your Questions", len(user_messages))
    
    with col2:
        st.metric("AI Responses", len(ai_messages))
        avg_context = sum(msg.get('context_used', 0) for msg in ai_messages) / len(ai_messages) if ai_messages else 0
        st.metric("Avg Context Used", f"{avg_context:.1f}")
    
    with col3:
        if intent_counts:
            st.write("**Most Common Topics:**")
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.write(f"• {intent}: {count}")
    
    # Timeline (simplified)
    if len(st.session_state.chat_messages) > 2:
        st.write("**Conversation Timeline:**")
        timeline_data = []
        for i, msg in enumerate(st.session_state.chat_messages):
            role = "You" if msg.get('is_user') else "AI"
            timeline_data.append({
                'Message': i + 1,
                'Role': role,
                'Time': msg.get('timestamp', ''),
                'Type': msg.get('intent', '').replace('_', ' ').title() if not msg.get('is_user') else 'Question'
            })
        
        # Display as a simple table
        st.dataframe(timeline_data, use_container_width=True) 