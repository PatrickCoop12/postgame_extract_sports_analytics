import streamlit as st
import openai, boto3
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import os
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initializing textract API

textract = boto3.client('textract', region_name='us-east-1', aws_access_key_id='AKIASF2P7IDFKIGWFHIU',aws_secret_access_key='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')

# Creating
def document_to_retriever(document, chunk_size, chunk_overlap):
    with open(document, 'rb') as file:
        img = file.read()
        bytes = bytearray(img)

    extracted_text = textract.analyze_document(Document = {'Bytes': bytes}, FeatureTypes = ['TABLES'])


    text = []
    blocks = extracted_text['Blocks']

    for block in blocks:
        if block['BlockType'] == 'WORD':
             text.append(block['Text'])
    # text formation based upon Line block type
        elif block['BlockType'] == 'LINE':
             text.append(block['Text'])

    words = " ".join(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.create_documents([words])
    print(splits)
    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(openai_api_key='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'))
    retriever = vectorstore.as_retriever()


    return retriever, words


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_response(retriever, input_text):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0,
                                                          openai_api_key='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'),
                                               retriever=retriever, verbose=True, memory=st.session_state.memory)
    result = qa({"chat_history": chat_history, "question":input_text})
    return result['answer']


st.set_page_config(page_title="PostGame Extract", page_icon=":tada:", layout="wide")
st.title(":blue[Post]:red[Game] Extract")
st.subheader('The Sports Administration Document Interaction and Analysis Tool', divider='red')
st.markdown(
"""
#### Getting Started:
- Use the file uploader to upload an image to get stared.
- Upon upload an interactive chat will appear below. You may use the chat to ask specific questions about the document
  (the tool currently only supports single page uploads).
- Two download options will also appear in the left sidebar allowing you to export images of handwritten documents as scanned pdfs. Text extracted 
  from the image is also available for export in .txt format. 
"""
)
st.subheader('', divider='blue')


left_column, right_column = st.columns(2)
st.sidebar.markdown('### About')
st.sidebar.markdown('This app has been designed to allow users the ability to capture a photo of a variety of documents relating to sports. These documents may include anything from boxscores, play-by-play sheets, and even handwritten notes. PDF files are also supported.')
file_upload = st.sidebar.file_uploader("Please Upload a File", type=['png','jpeg', 'jpg', 'pdf'])

# Initializing messages in the session state for call back to chat history during chat session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#for message in st.session_state.messages:
    #with st.chat_message(message["role"]):
        #st.markdown(message["content"])

# File upload conditional to limits code execution until after file has been uploaded. Code executed is display of
# image, but you may adjust as needed.
if file_upload is not None:
    st.markdown('#### Chatbot')
# Retriever and word extraction
    with open(file_upload.name,"wb") as f: 
      f.write(file_upload.getbuffer())

    
    retriever, words = document_to_retriever(file_upload.name, 4000, 2)
    with open(file_upload.name, mode='wb') as w:
        w.write(file_upload.getvalue())
    if '.pdf' not in file_upload.name:
        image = Image.open(file_upload.name)
        st.sidebar.image(file_upload.name, use_column_width=True)
        #right_column.image(image, caption='menu',use_column_width=True)

# Taking extracted words and making them available for download
# PDF Scan conversion
    option = st.sidebar.selectbox(
        'Export',
        ('Select', 'As .txt File','As Scan')
    )
    if option == 'As .txt File':
        text_export = words
        st.sidebar.download_button('download extracted text', text_export, file_name='extracted_text.txt')

    if option == 'As Scan':
        st.sidebar.download_button('download Scan', file_upload, file_name=file_upload.name)
# Scanner tool implementation


    #st.write(text_export)

# extracting text from image and formatting into a retriever for our LLM

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    chat_history = []

# Chat configuration and setup

    if prompt := st.chat_input("Ask me about the repository!"):
    #st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Generating an answer..."):
            with st.chat_message("assistant"):
                #message_placeholder = st.empty()
                #full_response = ""
                response = generate_response(retriever, prompt)
                st.markdown(response)
                #for response in response:
                    #full_response += response
                    #message_placeholder.markdown(full_response + "▌")
                #message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

#if prompt := st.chat_input(placeholder="What would you like to know?"):
#   st.session_state.messages.append({"role": "user", "content": prompt})
#    st.chat_message("user").write(prompt)
#    response = generate_response(retriever, prompt)
#    msg = response
#    st.session_state.messages.append(msg)
#    st.chat_message("assistant").write(msg)



#with st.chat_message("assistant"):
#    response = generate_response(retriever,st.session_state.messages)
#    st.session_state.messages.append({"role": "assistant", "content": response})
#    st.write(response)

#with st.chat_message('my_form'):
#    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')

#    if submitted:
#        generate_response(retriever,text)


