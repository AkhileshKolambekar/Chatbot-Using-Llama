import streamlit as st
import time
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
from transformers import pipeline

# Import the model
llm = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id = "model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type = 'llama'
    )

# Import transformer to convert english to hindi
hindi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

st.set_page_config(page_title='AgriBot',layout='wide')

st.subheader('TechChat 🤖',divider = 'rainbow')

language = st.selectbox('Select Language',options = ['English','Hindi'],index = None)

convo = st.container(height = 400,border=True)

with convo:
    if language:

        # A method to store the conversation for context
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if language == 'English':
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            elif language == 'Hindi':
                if message['role'] == 'user':
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                elif message['role'] == 'assistant':
                    with st.chat_message(message["role"]):
                        translated = hindi(message["content"])
                        st.markdown(translated[0]['translation_text'])
            
        #Function to Generate Response
        def generate_response(user_prompt, context):

            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n","\n<<SYS>>\n\n"
            SYSTEM_PROMPT = B_SYS + """\
                        You are a helpful assistant who is an expert in technology. 
                        You do not respond as 'User' or pretend to be 'User'. 
                        You only respond once as 'Assistant' to the prompt.
                        You should provide precise response.
                        """ + E_SYS

            if len(context)>2:
                template = B_INST+SYSTEM_PROMPT+"""Prompt:{user_prompt}\n Context:{context}"""+E_INST 
                prompt = PromptTemplate(input_variables=['user_prompt','context'],template=template)
                response = llm(prompt.format(user_prompt = user_prompt,context = context))
            else:
                template = B_INST+SYSTEM_PROMPT+"{user_prompt}"+E_INST 
                prompt = PromptTemplate(input_variables=['user_prompt'],template=template)
                response = llm(prompt.format(user_prompt = user_prompt))

            return response

user_prompt = st.chat_input("Type your message here...")

with convo:
    if user_prompt:

            #Display User Message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            #Store the User Message for Context
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            #Display the Response
            with st.chat_message("assistant"):
                response = generate_response(user_prompt,st.session_state.messages)
                if language == 'English':
                    st.markdown(response)
                elif language == 'Hindi':
                    translated = hindi(response)
                    st.markdown(translated[0]['translation_text'])
            
            #Store the Response for Context
            st.session_state.messages.append({"role":'assistant',"content":response})