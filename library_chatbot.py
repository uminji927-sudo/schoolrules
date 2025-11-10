import os
import streamlit as st
import nest_asyncio
import tempfile
from pathlib import Path

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# RAG 관련 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Gemini API 키 설정 ---
try:
    # ⚠️ 파일 경로 설정
    PDF_FILE_PATH = "명신여고소개.pdf" 
    
    # ⚠️ API 키 설정 확인
    if "GOOGLE_API_KEY" not in st.secrets:
        raise ValueError("GOOGLE_API_KEY가 Streamlit Secrets에 설정되지 않았습니다.")
    
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error(f"⚠️ 설정 오류: {str(e)}")
    st.info("💡 `st.secrets` 파일에 `GOOGLE_API_KEY`를 설정하고, `명신여고소개.pdf` 파일이 코드와 같은 위치에 있는지 확인해주세요.")
    st.stop()


# --- 2. RAG 파이프라인 구축 (캐시) ---
@st.cache_resource(show_spinner="📚 학교 소개 문서 로딩 및 학습 중...")
def get_retriever(pdf_path: str):
    """
    PDF 문서를 로드하고, 분할하고, 임베딩하여 FAISS 벡터 저장소에서 검색기(Retriever)를 생성합니다.
    """
    if not Path(pdf_path).exists():
        st.error(f"❌ '{pdf_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        st.stop()

    try:
        # 1. 문서 로드 (PyPDFLoader 사용)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2. 텍스트 분할 (RecursiveCharacterTextSplitter 사용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True
        )
        texts = text_splitter.split_documents(documents)

        # 3. 임베딩 모델 로드 (Gemini 임베딩 모델 사용)
        # embedding_model = GoogleGenerativeAIEmbeddings(model="embedding-001") # 임베딩 비용을 줄이기 위해 권장
        embedding_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

        # 4. 벡터 저장소 생성 및 저장
        # FAISS: 로컬에서 빠르고 간단하게 사용할 수 있는 벡터 저장소
        vectorstore = FAISS.from_documents(texts, embedding_model)

        # 5. 검색기(Retriever) 설정
        # k=3: 사용자 질문과 가장 관련 높은 3개의 문서를 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        return retriever

    except Exception as e:
        st.error(f"❌ RAG 파이프라인 구축 실패: {str(e)}")
        st.stop()

# --- 3. LLM 및 RAG 체인 설정 (캐시) ---
@st.cache_resource(show_spinner="🤖 챗봇 모델 로딩 중...")
def get_rag_chain(selected_model, retriever):
    """
    LLM, 프롬프트, 검색기를 결합한 RAG 체인을 생성합니다.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.3, # 사실 기반 답변을 위해 낮은 온도 설정
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 로드 실패: {str(e)}")
        st.stop()

    # 1. 시스템 프롬프트 설정 (RAG용)
    SYSTEM_PROMPT = (
        "당신은 명신여고 소개 전문가 '명신AI'입니다. "
        "항상 한국어와 친절하고 전문적인 존댓말을 사용합니다. "
        "제공된 **문맥(context)** 정보만을 사용하여 사용자의 질문에 답변해주세요. "
        "문맥에 답변할 정보가 없다면, '죄송하지만, 제가 가진 명신여고 소개 자료에는 해당 정보가 없습니다. 다른 질문을 해주세요.'라고 답변하세요. "
        "대화에 이모지를 적절히 섞어 답해주세요. 🎓\n\n"
        "**문맥(Context):**\n{context}\n\n"
    )
    
    # 2. 답변 생성 프롬프트
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"), # 대화 기록 (RAG에서도 필수)
            ("human", "{input}"),        # 사용자의 현재 입력
        ]
    )

    # 3. 문서 결합 체인 (검색된 문서를 프롬프트의 context 변수에 넣음)
    document_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    
    # 4. 검색 체인 (검색기 + 문서 결합 체인)
    # create_retrieval_chain은 {input}으로 질문을 받고, 검색기로 문서를 찾은 뒤, document_chain에 전달하여 답변을 생성합니다.
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


# --- 4. Streamlit UI 설정 ---

st.header("명신여자고등학교 소개 AI 챗봇 🎓")
st.info("명신여고소개.pdf 기반의 전문 AI입니다. 학교에 대해 무엇이든 물어보세요.")

# 채팅 기록을 Streamlit의 세션 상태(session_state)에 저장
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 모델 선택
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-2.5-pro"),
    index=0,
    help="가장 빠르고 효율적인 2.5 Flash 모델을 추천합니다."
)

# RAG 검색기 가져오기
rag_retriever = get_retriever(PDF_FILE_PATH)

# RAG 체인 가져오기
rag_chain = get_rag_chain(option, rag_retriever)

# 대화 기록을 관리하는 Runnable 생성
# LangChain RAG 체인과 대화 기록을 결합하여 컨텍스트를 유지합니다.
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- 5. 채팅 UI 로직 ---

# 첫 방문 시 환영 메시지 추가
if not chat_history.messages:
    chat_history.add_ai_message("명신여자고등학교 소개 전문 AI, 명신AI입니다! 😊 궁금한 점을 질문해주세요.")

# 이전 대화 기록 모두 출력
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 입력 받기
if prompt_message := st.chat_input("명신여고에 대해 질문하세요..."):
    # 사용자가 입력한 메시지 출력
    st.chat_message("human").write(prompt_message)
    
    # AI 응답 생성 및 출력
    with st.chat_message("ai"):
        with st.spinner("명신여고 자료에서 답변을 찾는 중..."):
            config = {"configurable": {"session_id": "any_id"}}
            
            # 체인 실행
            # RAG 체인의 결과는 딕셔너리({answer, context, input}) 형태로 반환됩니다.
            # 최종적으로 사용자에게 보여줄 것은 'answer
