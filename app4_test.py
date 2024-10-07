__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai, boto3
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import os
import io
import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures, Direction, DirectionalFinderType
import chromadb
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()
#from streamlit_chromadb_connection import ChromadbConnection


# Calling required API keys from streamlit secrets
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['AWS_SECRET_ACCESS_KEY']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']



# Initializing extraction tools
textract = boto3.client('textract', region_name='us-east-1', aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'])
extractor = Textractor(region_name="us-east-1", kms_key_id= '4b28ef85-000d-44e3-9210-cca4c06af170')

# Creating functions needed for workflow execution:
# Text Extraction
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
    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(model = "text-embedding-3-large", openai_api_key=OPENAI_API_KEY))
    retriever = vectorstore.as_retriever()
    print(retriever)


    return vectorstore, retriever, words


# Response generation function
def generate_response(retriever, input_text):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4o", temperature=0,
                                                          openai_api_key=OPENAI_API_KEY),
                                               retriever=retriever, verbose=True, memory=st.session_state.memory)
    result = qa({"chat_history": chat_history, "question":input_text})
    return result['answer']

# Scan tool functions
# Point ordering function used in perspective transform fucntion
def order_points(pts):
   # initializing the list of coordinates to be ordered
   rect = np.zeros((4, 2), dtype = "float32")

   s = pts.sum(axis = 1)

   # top-left point will have the smallest sum
   rect[0] = pts[np.argmin(s)]

   # bottom-right point will have the largest sum
   rect[2] = pts[np.argmax(s)]

   #computing the difference between the points, the
   #top-right point will have the smallest difference,
   #whereas the bottom-left will have the largest difference
   diff = np.diff(pts, axis = 1)
   rect[1] = pts[np.argmin(diff)]
   rect[3] = pts[np.argmax(diff)]

   # returns ordered coordinates
   return rect

# Perspective transform Function
def perspective_transform(image, pts):
   # unpack the ordered coordinates individually
   rect = order_points(pts)
   (tl, tr, br, bl) = rect

   #compute the width of the new image, which will be the
   #maximum distance between bottom-right and bottom-left
   #x-coordinates or the top-right and top-left x-coordinates
   widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
   widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
   maxWidth = max(int(widthA), int(widthB))

   #compute the height of the new image, which will be the
   #maximum distance between the top-left and bottom-left y-coordinates
   heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
   heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
   maxHeight = max(int(heightA), int(heightB))

   #construct the set of destination points to obtain an overhead shot
   dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

   # compute the perspective transform matrix
   transform_matrix = cv2.getPerspectiveTransform(rect, dst)

   # Apply the transform matrix
   warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

   # return the warped image
   return warped


# Scan conversion function that utilizes two functions above
def scan_transformation(document_image):
    original_img = cv2.imread(document_image)
    copy = original_img.copy()

# The resized height in hundreds
    ratio = original_img.shape[0] / 500.0
    img_resize = imutils.resize(original_img, height=500)

    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)


    edged_img = cv2.Canny(blurred_image, 75, 200)
    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            doc = approx
            break


    p = []

    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)


    warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    T = threshold_local(warped_image, 11, offset=10, method="gaussian")
    warped = (warped_image > T).astype("uint8") * 255

    return cv2.imwrite('./'+'scan'+'.png',warped)

# Page configurations and setup

st.set_page_config(page_title="PostGame Extract", page_icon=":tada:", layout="wide")
st.title(":blue[Post]:red[Game] Extract")
st.subheader('Interactive Sports Document OCR Tool', divider='red')
with st.expander('Instructions'):    
    st.markdown(
    """
    #### Getting Started
    1. To get started with the app, please use the file uploader to upload an image of a document or pdf     
    (the tool currently only supports single page uploads).
      * If the sidebar containing the file uploader is not showing, please click the arrow in the top left corner 
                     (on mobile you may have to scroll up)
    2. Upon upload of your file:
      * An interactive chat will appear below. You may use the chat to ask specific questions about the document.
      * An image of the file will be available for view
      * Several export options will be made available by selecting an option from the drop-down on the left. After export is generated, please click the download button.
    - Export Options: 
      - For Image Files (.jpg, .jpeg, .png) 
        - Raw Text Extraction
        - Converted Scanned Image* (converts raw image of document into a digital scan). Best used for handwritten notes, and printed templates and forms containing handwriting.
        - Table Extractor to Excel (PDFs containing tabular data can be automatically converted to excel for further analysis)
      - For PDF Files
        - Raw Text Extraction
        - Table Extractor to Excel 
    ###### *please note scan conversion is only available for image files uploaded, and not available for PDFs.
    """
    )
st.subheader('', divider='blue')


left_column, right_column = st.columns(2)
st.sidebar.markdown('### About')
st.sidebar.markdown('This app has been designed to allow users the ability to capture a photo of a variety of documents relating to sports. These documents may include anything from boxscores, play-by-play sheets, and even handwritten notes. PDF files are also supported.')
file_upload = st.sidebar.file_uploader("Please Upload a File", type=['jpg','jpeg','png', 'pdf'])

# Initializing messages and memory in the session state for call back to chat history during chat session
if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#for message in st.session_state.messages:
    #with st.chat_message(message["role"]):
        #st.markdown(message["content"])

# If statement used to check for file upload to limit further code execution until after file has been uploaded.

if file_upload is not None:
    
    #try:
        #vectorstore.delete(vectorstore.get()['ids'])
    #except:
        #pass
        
    st.markdown('#### Chatbot')
    # Temporary file saving
    with open(file_upload.name,"wb") as f: 
      f.write(file_upload.getbuffer())
    
    vectorstore, retriever, words = document_to_retriever(file_upload.name, 4000, 2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    with open(file_upload.name, mode='wb') as w:
        w.write(file_upload.getvalue())

    if '.pdf' in file_upload.name:
        option = st.sidebar.selectbox(
            'Export',
            ('Select', 'As .txt File', 'As Excel')
            )
        # Extracted words export as .txt file
        if option == 'As .txt File':
            text_export = words
            st.sidebar.download_button('download extracted text', text_export, file_name='extracted_text.txt')
        # Table extraction export into excel file
        if option == 'As Excel':
            document = extractor.analyze_document(
                file_source=file_upload.name,
                features=[TextractFeatures.TABLES],
                save_image=True
                )

            document.export_tables_to_excel("download.xlsx")
            with open('download.xlsx', "rb") as fh:
                buffer = io.BytesIO(fh.read())
            st.sidebar.download_button('download excel', buffer, file_name = 'download.xlsx')
    elif '.pdf' not in file_upload.name:
        image = Image.open(file_upload.name)
        st.sidebar.image(file_upload.name, use_column_width=True)
        #right_column.image(image, caption='menu',use_column_width=True)

        option = st.sidebar.selectbox(
            'Export',
            ('Select', 'As .txt File','As Scan', 'As Excel')
        )
    # Extracted words export as .txt file
        if option == 'As .txt File':
            text_export = words
            st.sidebar.download_button('download extracted text', text_export, file_name='extracted_text.txt')
    # Converted scan image export 
        if option == 'As Scan':
            try:
                scan_transformation(file_upload.name)
                with open('scan.png', "rb") as f:
                    scan = io.BytesIO(f.read())
    # Exception raised if there is an error in converting document image to scan. Likely due to an error detecting corners/edges in document
    # User will be asked to take a better image with a contrasting background.
            except:
                st.warning('Please place document for scan conversion on surface with a greater contrast (i.e. a dark table surface)')
            st.sidebar.download_button('download Scan', scan, file_name='scan.png')
    # Table extraction export into excel file
        if option == 'As Excel':
            document = extractor.analyze_document(
                file_source=file_upload.name,
                features=[TextractFeatures.TABLES],
                save_image=True
                )

            document.export_tables_to_excel("download.xlsx")
            with open('download.xlsx', "rb") as fh:
                buffer = io.BytesIO(fh.read())
            st.sidebar.download_button('download excel', buffer, file_name = 'download.xlsx')


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    chat_history = []

    # Chat configuration and setup

    if prompt := st.chat_input("Ask me a question about your document!"):
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
    
    # Deleting all items from vectorstore to avoid document hallucination/confusion
    #vectorstore.delete(vectorstore.get()['ids'])


