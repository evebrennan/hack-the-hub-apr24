import time
import os

import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Initialise conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

conversation_history = st.session_state['conversation_history']

# Function to interact with OpenAI's completion endpoint
def ask_gpt(question):
    conversation_history.append({"user": question})
    response = client.chat.completions.create(model="gpt-4-0125-preview",  # Consider using "text-davinci-003" for a more informative response
        messages=[{"content":question, "role":"user"}],
        temperature=0.7,
        max_tokens=150)
    conversation_history.append({"assistant": response.choices[0].message.content.strip()})
    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("Chat with GPT")

    st.write("Welcome! Ask me anything.")

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
    user_input = st.chat_input(placeholder="Ask me anything", key="user_input")

    # Update enter_pressed based on button click or Enter key press
    st.session_state.enter_pressed = st.session_state.get('enter_pressed', False)

    if user_input is not None and user_input.strip() != "":  # Check for empty input
        # Display user input
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

        # Get response from GPT
        response = ask_gpt(user_input)
        
        # Display response
        #  Simulate stream of response with milliseconds delay
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

if __name__ == "__main__":
    main()