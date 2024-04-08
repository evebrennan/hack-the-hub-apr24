import os

import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Initialise conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

conversation_history = st.session_state['conversation_history']

def create_context(selected_archetype):
  """
  This function creates context based on the selected archetype.
  """
  context = ""
  if selected_archetype == "Medical Professional":
    context = """I have access to a vast medical knowledge base and can provide information on symptoms, diagnoses, and treatment options. However, I cannot diagnose or treat any medical conditions. Please consult a licensed physician for any medical concerns."""
  elif selected_archetype == "Spiritual Advisor":
    context = """I can offer guidance and support from a spiritual perspective. I can share wisdom from various traditions and help you explore your inner self."""
  elif selected_archetype == "Old Age Wisdom":
    context = """I have access to a lifetime of experiences and knowledge. I can offer practical advice and a listening ear."""
  return context

# Function to interact with OpenAI's completion endpoint
def ask_gpt(question, selected_archetype):
    # Add query to the conversation history
    conversation_history.append({"user": question})

    # To do: make these dynamic
    context = create_context(selected_archetype)

    # Create prompt
    prompt_template = f"""
        The following is a conversation between a person with a mental health concern and an AI. The AI provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Answers must have less than 300 words. Use the following pieces of context to answer the question at the end.

        {context}

        User: {question}
        {selected_archetype}:"""
    
    # Send prompt to model
    response = client.chat.completions.create(model="gpt-4-0125-preview",  
        messages=[{"content":prompt_template.format(question), "role":"user"}],
        temperature=0.7, max_tokens=300)

    # Add response to the conversation history
    conversation_history.append({"assistant": response.choices[0].message.content.strip()})
    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("Artificial Wisdom")

    st.write("An alternative approach to mental health.")

    # Archetype selection dropdown
    archetype_options = ["-", "Medical Professional", "Spiritual Advisor", "Old Age Wisdom"]
    selected_archetype = st.selectbox("Who would you like to talk to?", archetype_options)

    st.info(create_context(selected_archetype))

    for message in conversation_history:
        if message.get("user"):
            with st.chat_message("user"):
                st.markdown(message["user"])
        if message.get("assistant"):
            with st.chat_message("assistant"):
                st.markdown(message["assistant"])

    # Initialize enter_pressed in session state (default False)
    if 'enter_pressed' not in st.session_state:
        st.session_state['enter_pressed'] = False

    # User input box using chat_input
    user_input = st.chat_input(placeholder="How are you?", key="user_input")

    # Update enter_pressed based on button click or Enter key press
    st.session_state.enter_pressed = st.session_state.get('enter_pressed', False)

    if user_input is not None and user_input.strip() != "":  # Check for empty input
        # Display user input
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

        # Get response from GPT
        response = ask_gpt(user_input, selected_archetype)
        
        # Display response
        message_placeholder.markdown(response)

if __name__ == "__main__":
    main()