import streamlit as st
import sys
import argparse
from agentProfessor import create_chatbot, my_agent

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_pdf", type = str, help="Path to data file", default="../Agentic_AI_A_Comprehensive_Survey_of_Technologies_.pdf")
parser.add_argument("--email_to_send", type = str, help="Recipient email", default="ymil@ciklum.com")
args = parser.parse_args()

chatbot = create_chatbot(args.path_to_pdf, args.email_to_send)

st.title("Assistant for creating lecture notes")
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ProfessorBot. How can I assist you with your lecture notes today? Notes will be sent to your email"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer..."):
            result_text = my_agent(chatbot, input)
            st.write(result_text) 
    message = {"role": "assistant", "content": result_text}
    st.session_state.messages.append(message)