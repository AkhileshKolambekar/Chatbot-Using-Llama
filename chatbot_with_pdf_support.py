import streamlit as st
import PyPDF2
from streamlit_pdf_viewer import pdf_viewer

from langchain_community.llms import Ollama
from transformers import AutoTokenizer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from streamlit import session_state as ss

import warnings
warnings.filterwarnings('ignore')

st.title('Chatbot')

# Load the model
with st.spinner('Loading the model'):
    if 'llm' not in ss:
        ss.llm = Ollama(model = 'llama3.1')

# Load the tokenizer
with st.spinner('Loading the tokenizer'):
    if 'tokenizer' not in ss:
        token_file = open("token_file.txt",'r')
        my_token = token_file.read()
        ss.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='meta-llama/Meta-Llama-3.1-8B',
                                                token = my_token)

EOS_TOKEN = ss.tokenizer.eos_token

# Load the embedder
with st.spinner('Loading the embedder'):
    if 'embedder' not in ss:
        ss.embedder = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2",model_kwargs = {'trust_remote_code':True})

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)


if 'pdf_file' not in ss:
    ss.pdf_files = []
    ss.pdf_file_names = []

if 'upload_complete' not in ss:
    ss.upload_complete = False

if 'vector_store' not in ss:
    ss.vector_store = None

if 'show_uploader' not in ss:
    ss.show_uploader = True

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:

    ss.pdf_files = st.file_uploader('Upload your pdf', type=('pdf'),accept_multiple_files=True)

    if len(ss.pdf_files)>0:

        submit_pdf = st.button('Done')

        if submit_pdf:
            ss.upload_complete = True

        if ss.upload_complete:
            ss.pdf_file_names = [file.name for file in ss.pdf_files]

            status = st.empty()
            
            if len(ss.pdf_file_names)==1:
                status.success('File uploaded successfully')
            else:
                status.success('Files uploaded successfully')

            with st.spinner('Extracting Text'):
                if ss.vector_store == None:
                    # Read the pdf
                    text = ""
                    for pdf in ss.pdf_files:
                        read_pdf = PyPDF2.PdfReader(pdf)
                        num_of_pages = len(read_pdf.pages)
                        
                        for page_num in range(num_of_pages):
                            page = read_pdf.pages[page_num]
                            content = page.extract_text().strip()
                            if content:
                                text += content

                    # Split the text and create a vectorstore
                    split_text = text_splitter.split_text(text)
                    ss.vector_store = FAISS.from_texts(split_text, ss.embedder)

        display_pdf = st.checkbox('Show PDF')

        if display_pdf:
            file_name = st.selectbox('Select PDF',ss.pdf_file_names)
            file_idx = ss.pdf_file_names.index(file_name)
            binary_data =ss.pdf_files[file_idx].getvalue()
            with st.container(height = 1000):
                st.write(pdf_viewer(input = binary_data))


if len(ss.pdf_files) == 0:
    ss.upload_complete = False
    ss.vector_store = None
    ss.pdf_file_names = []

message_container = st.container(height=450)

# List to store messages
if "messages" not in ss:
    ss['messages'] = [{'role':'ai','content':'How can I help you today?'}]

# Load previous chats
for message in ss.messages:
    with message_container:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

memory = ss.messages

if len(ss.pdf_files)>0 and ss.upload_complete:
    
    retriever = ss.vector_store.as_retriever(k = 3)
    
    if len(ss.messages)>2:
        prompt_template = """
        User has asked you a query from a pdf that it has provided.
        Relevant Text consists of texts from the pdf that might help you understand the query.
        Context consists of previous chats with the user.
        Provide an accurate answer to the user query inferred from the Relevant Text.
        You can use the context if necessary in order to improve the answer.
        If you cannot provide an answer then simply state that instead of coming up with a wrong answer.

        ## Context:
        {context}

        ### Relevant Text:
        {relevant_text}

        ### Query:
        {query}
        """
        final_prompt = PromptTemplate.from_template(template = prompt_template,
                                            kwargs={'input_variables':['context','relevant_text','query']})
        
        chain = (
            {'context': lambda x: memory,'relevant_text': retriever,'query': RunnablePassthrough()}
            |final_prompt
            |ss.llm
            |StrOutputParser()
        )

    else:
        prompt_template = """
        User has asked you a query from a pdf that it has provided.
        Relevant Text consists of texts from the pdf that might help you understand the query.
        Provide an accurate answer to the user query inferred from the Relevant Text.
        If you cannot provide an answer then simply state that instead of coming up with a wrong answer.

        ### Relevant Text:
        {relevant_text}

        ### Query:
        {query}
        """
        final_prompt = PromptTemplate.from_template(template = prompt_template,
                                            kwargs={'input_variables':['relevant_text','query']})
        chain = (
            {'relevant_text': retriever,'query': RunnablePassthrough()}
            |final_prompt
            |ss.llm
            |StrOutputParser()
        )

else:    

    if len(ss.messages)>2:

        prompt_template = """
        Provide an accurate answer to the user query and use the context provided if necessary.
        If you cannot provide an answer then simply state that instead of coming up with a wrong answer.

        ### Context:
        {context}

        ### Query:
        {query}
        """

        final_prompt = PromptTemplate.from_template(template = prompt_template,
                                                    kwargs={'input_variables':['context','query']})
        
        chain = (
            {'context': lambda x: memory,'query': RunnablePassthrough()}
            |final_prompt
            |ss.llm
            |StrOutputParser()
        )

    else:
        prompt_template = """
        Provide an accurate answer to the user query.
        If you cannot provide an answer then simply state that instead of coming up with a wrong answer.

        ### Query:
        {query}
        """

        final_prompt = PromptTemplate.from_template(template = prompt_template,
                                                    kwargs={'input_variables':['query']})
        chain = (
            {'query': RunnablePassthrough()}
            |final_prompt
            |ss.llm
            |StrOutputParser()
        )


user_input = st.chat_input('Enter your query')

with message_container:

    if user_input:
        st.chat_message('user').write(user_input)
        ss.messages.append({'role':'user','content':user_input})

        with st.spinner('Generating Response'):
            response = chain.invoke(user_input)

        st.chat_message('ai').write(response)
        ss.messages.append({'role':'ai','content':response})

memory = ss.messages