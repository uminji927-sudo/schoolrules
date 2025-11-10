import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    # PDF 파일 로드
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        st.error(f"❌ PDF 파일 로드 실패: {file_path} 파일을 확인해주세요. ({str(e)})")
        raise

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(pages)

@st.cache_resource
# 수정: pages를 _pages로 변경하여 Streamlit 캐싱에서 제외
def get_vectorstore(_pages):
    # 임베딩 모델 로드 (Kor-MiniLM-L6-v2 사용)
    # 다운로드에 시간이 걸릴 수 있습니다.
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # Chroma DB에 저장
    # 명신여고 관련 파일이므로 디렉토리 이름을 'mshs_db'로 변경했습니다.
    vectorstore = Chroma.from_documents(
        documents=_pages, # 수정: _pages 사용
        embedding=embeddings, 
        persist_directory="./mshs_db" 
    )
    return vectorstore

@st.cache_resource
def initialize_components(selected_model):
    # 파일 경로를 명신여고 소개 PDF로 변경
    file_path = "명신여고소개.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-2.5-flash' 모델을 사용해보세요.")
        raise
        
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrie
