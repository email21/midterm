import streamlit as st
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 기록을 저장할 리스트

def reset_chat():
    """대화 내용을 초기화하는 함수"""
    st.session_state.messages.clear()
    st.rerun()

# Streamlit 페이지
st.title("💠 Solar LLM Chatbot ")
st.write("이전 대화를 기억하는 기본 챗봇입니다.")

# 사이드바에 초기화 버튼 추가
with st.sidebar:
    st.header("옵션")
    if st.button("대화 초기화"):
        reset_chat()

# 챗 모델 초기화
try:
    chat_model = ChatUpstage(upstage_api_key=api_key)
except Exception as e:
    st.error(f"API 키 오류: {e}")
    st.stop()

# 기본 시스템 프롬프트 템플릿 생성
system_template = """당신은 친근하고 도움이 되는 AI 어시스턴트입니다. 
사용자와의 이전 대화 내용을 참고하여 자연스럽고 일관성 있는 대화를 이어가세요.
항상 정중하고 친절하게 답변해주세요."""

# 프롬프트 템플릿 생성 (이전 대화 기록을 포함)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder("chat_history"),  # 이전 대화 기록이 들어갈 자리
    ("human", "{input}")  # 현재 사용자 입력
])

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요!"):
    # 사용자 메시지를 화면에 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 사용자 메시지를 세션 상태에 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 이전 대화 기록을 LangChain 메시지 형태로 변환
        chat_history = []
        for msg in st.session_state.messages[:-1]:  # 현재 메시지 제외
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        try:
            # 프롬프트에 현재 입력과 대화 기록 전달
            messages = prompt_template.format_messages(
                chat_history=chat_history,
                input=user_input
            )
            
            # AI 모델로부터 응답 생성
            response = chat_model.invoke(messages)
            ai_response = response.content
            
            # 응답을 화면에 표시
            message_placeholder.markdown(ai_response)
            
            # AI 응답을 세션 상태에 저장
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")