#must import sqlite3 to utilize Chroma in deployment on the streamlit hosting site
#__import__('pysqlite3')
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
#!pip install docx2txt pypdf 


#FUNCTIONS
# function to load local files
# return list of langchain documents. one document per page.  This function works with pdf's or Word documents. Use an elif clause for each file extension format you want to support
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}...')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f'Loading {file}...')
        loader = TextLoader(file)
    else:
        print('Sorry.  Document format not supported.')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    #split the data into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def q_and_a(vector_store, q, k=3):
    from langchain_classic.chains import RetrievalQA
    
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    #k is the number of similar chunks retrieved. higher k costs more in chatgpt but gives better answer
    chain=RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.invoke(q)
    return answer


## embedding costs for OpenAI
def calculate_embedding_cost(texts):
# tiktoken library calculates cost of embedding in OpenAI
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    #if history key in session state exists, delete it to clear history in text box.
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('./data.jpeg', width=400)

    st.subheader('AI LLM Question and Answering System')
    st.write('This is an LLM document analysis program using AI. You can upload a document (.pdf, .txt, or .docx files) and use ChatGPT to analyze and answer questions about your document. In order to use, enter your ChatGPT API key (which is not stored or revealed to anyone) in the left-hand sidebar and upload a document to analyze. Be aware, ChatGPT charges a small amount to chunk and embed your file. Then select parameters (chunks and k-value) to determine how thoroughly to break up the document and inspect it. Then click the button. After processing your file, use the text box below to type in your questions regarding your file.')

    with st.sidebar:
        api_key = st.text_input('Input Your Gemini API Key:', type='password')
        if api_key == "":
            os.environ['GOOGLE_API_KEY'] = st.secrets["gemini_api_key"]
            
        else:
            os.environ['GOOGLE_API_KEY']=api_key

        uploaded_file = st.file_uploader('Upload Your PDF, TXT, or DOCX File For Processing:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size (between 100 and 2048):', min_value=100, max_value=2048, value=512)
        k = st.number_input('k-value (between 1 and 20): Higher k-value costs more but can give better results', min_value=1, max_value=20, value=3)
        add_data = st.button('Run File Analysis', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, processing, and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                #put vector store in session state
                st.session_state.vs = vector_store
                st.success('File uploaded, processed, and embedded successfully.')

    st.divider()

    q = st.text_input('Ask a question here about your uploaded file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            answer = q_and_a(vector_store, q, k)
            st.text_area('AI LLM Answer: ', value=answer["result"])

            st.divider()

            # create history key
            if 'history' not in st.session_state:
                st.session_state.history = ''
            #concatenate current question with its answer
            value = f'Q: {q}\nA: {answer["result"]}'
            st.session_state.history = f'{value} \n {"-"*50} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Response History', value=h, key='history', height=400)
