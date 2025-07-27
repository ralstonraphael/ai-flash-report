"""
Enhanced conversational analysis section for Streamlit app.
Clean ChatGPT-like interface for document analysis.
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
        tone = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL)
        st.session_state.conv_engine = ConversationalEngine(
            temperature=0.3,
            conversation_tone=tone
        )
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

def render_minimal_sidebar():
    """Render minimal sidebar controls."""
    
    with st.sidebar:
        st.markdown("### 🔬 AI Analysis")
        
        # Simple tone selector
        tone_options = {
            "Professional": ConversationTone.PROFESSIONAL,
            "Friendly": ConversationTone.FRIENDLY, 
            "Analytical": ConversationTone.ANALYTICAL,
            "Executive": ConversationTone.EXECUTIVE
        }
        
        current_tone = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL)
        selected_tone = st.selectbox(
            "Conversation Style",
            options=list(tone_options.keys()),
            index=list(tone_options.values()).index(current_tone),
            help="How should I communicate?"
        )
        
        # Update tone if changed
        new_tone = tone_options[selected_tone]
        if new_tone != st.session_state.get('conversation_tone'):
            st.session_state.conversation_tone = new_tone
            if 'conv_engine' in st.session_state:
                st.session_state.conv_engine.set_tone(new_tone)
            st.rerun()
        
        st.divider()
        
        # Simple stats
        if st.session_state.chat_messages:
            message_count = len([msg for msg in st.session_state.chat_messages if msg.get('is_user', True)])
            st.metric("Questions Asked", message_count)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            if 'conv_engine' in st.session_state:
                st.session_state.conv_engine.clear_history()
            st.rerun()

def render_welcome_message():
    """Render welcome message for new chats."""
    
    with st.chat_message("assistant", avatar="🔬"):
        tone_name = st.session_state.get('conversation_tone', ConversationTone.PROFESSIONAL).value
        st.write(f"""
        **Norstella AI Ready** • *{tone_name} mode*
        
        I've analyzed your documents. Ask me about:
        • Key insights & summaries
        • Strategic analysis  
        • Financial metrics
        • Market intelligence
        
        What would you like to know?
        """)

def render_suggested_starters():
    """Render simple suggested starter questions."""
    
    if st.session_state.chat_messages:
        return  # Only show on empty chat
        
    st.markdown("**💡 Quick starts:**")
    
    suggestions = [
        "Key highlights?",
        "Strategic analysis?", 
        "Financial metrics?",
        "Risks & opportunities?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"starter_{i}", use_container_width=True):
                # Add the suggestion as a user message
                user_message = {
                    'content': suggestion,
                    'is_user': True,
                    'timestamp': datetime.now().strftime("%H:%M")
                }
                st.session_state.chat_messages.append(user_message)
                
                # Generate AI response
                ai_response = process_user_message(suggestion)
                ai_response['is_user'] = False
                st.session_state.chat_messages.append(ai_response)
                
                st.rerun()

def process_user_message(user_input: str) -> Dict[str, Any]:
    """Process user message and generate AI response."""
    
    try:
        # Show processing status
        with st.status("Thinking...", expanded=False) as status:
            status.write("🔍 Searching documents...")
            
            # Get context from vectorstore
            vs = st.session_state.vectorstore
            context = vs.query_collection(user_input, k=6)
            
            status.write("🧠 Generating response...")
            
            # Generate response using conversational engine with advanced RAG
            engine = st.session_state.conv_engine
            response, intent = engine.generate_response(
                query=user_input,
                vectorstore=vs,
                progress_callback=lambda msg: status.write(msg)
            )
            
            status.write("✅ Done!")
        
        return {
            'content': response,
            'intent': intent.value,
            'timestamp': datetime.now().strftime("%H:%M"),
            'context_used': len(context)
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {
            'content': f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            'intent': 'error',
            'timestamp': datetime.now().strftime("%H:%M"),
            'context_used': 0
        }

def enhanced_analysis_section():
    """Main enhanced analysis section with ChatGPT-like interface."""
    
    # Check if documents are loaded
    if not st.session_state.get('documents_loaded', False):
        st.error("⚠️ Please upload and process documents first in the Upload tab")
        return
    
    # Initialize conversational components
    init_conversational_analysis()
    
    # Clean header
    st.markdown("### 💬 Chat with your documents")
    st.markdown("---")
    
    # Render minimal sidebar
    render_minimal_sidebar()
    
    # Chat container with fixed height for scrolling
    chat_container = st.container()
    
    with chat_container:
        # Show welcome message if no chat history
        if not st.session_state.chat_messages:
            render_welcome_message()
        else:
            # Display conversation history
            for message in st.session_state.chat_messages:
                if message.get('is_user', True):
                    with st.chat_message("user", avatar="👤"):
                        st.write(message['content'])
                        st.caption(f"_{message.get('timestamp', '')}_")
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message['content'])
                        # Show intent subtly
                        if 'intent' in message and message['intent']:
                            intent_display = message['intent'].replace('_', ' ').title()
                            st.caption(f"💡 _{intent_display}_ • {message.get('timestamp', '')}")
    
    # Suggested starters (only for empty chat)
    if not st.session_state.chat_messages:
        st.markdown("---")
        render_suggested_starters()
    
    # Chat input at bottom
    st.markdown("---")
    
    # Chat input with better UX
    user_input = st.chat_input(
        "Ask me anything about your documents...",
        key="main_chat_input"
    )
    
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