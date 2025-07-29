import streamlit as st
from typing import Dict, Any, List
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from sentiment_analysis import SentimentAnalysis

# .env 파일에서 API 키 로드
load_dotenv()

MAX_CONVERSATION_TURNS = 10
system_template = """당신은 친근하고 도움이 되는 AI 어시스턴트 **chatbot**입니다.
사용자와의 이전 대화 내용을 참고하여 자연스럽고 일관성 있는 대화를 이어가세요.
항상 정중하고 친절하게 답변해주세요."""

class Chatbot:
    """대화 기억 챗봇 시스템"""
    
    def __init__(self):
        self.chat_model = None
        self.sentiment_analysis = None
        self.prompt_template = None
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 초기화"""
        try:
            # API 키 확인
            api_key = os.getenv("SOLAR_API_KEY")
            if not api_key:
                raise ValueError("SOLAR_API_KEY가 설정되지 않았습니다.")
            
            # 챗봇 모델 초기화
            self.chat_model = ChatUpstage(upstage_api_key=api_key)
            
            # 감정 분석 시스템 초기화
            self.sentiment_analysis = SentimentAnalysis()
            
            # 프롬프트 템플릿 설정
            self._setup_prompt_template()
            
            print("------ 챗봇 시스템 초기화 완료 -----")
            
        except Exception as e:
            st.error(f"시스템 초기화 오류: {e}")
            raise
    
    def _setup_prompt_template(self):
        """프롬프트 템플릿 구성"""
        assistant_name = "chatbot"
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template.format(name=assistant_name)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
    def _convert_to_langchain_messages(self, conversation: List[Dict[str, str]]) -> List:
        """세션 상태의 대화를 LangChain 메시지로 변환"""
        chat_history = []
        
        # 현재 메시지 제외하고 변환
        for msg in conversation[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        return chat_history
    
    def generate_response(self, user_input: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """AI 응답 생성"""
        try:
            # 대화 기록 변환
            chat_history = self._convert_to_langchain_messages(conversation)
            
            # 프롬프트에 현재 입력과 대화 기록 전달
            messages = self.prompt_template.format_messages(
                chat_history=chat_history,
                input=user_input
            )
            
            # AI 모델로부터 응답 생성
            response = self.chat_model.invoke(messages)
            ai_response = response.content
            
            # 감정 분석 처리
            sentiment_info = self.sentiment_analysis.process_message_sentiment(ai_response)
            
            return {
                'status': 'success',
                'response': ai_response,
                'sentiment_info': sentiment_info
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': "응답 생성 중 오류가 발생했습니다.",
                'sentiment_info': {'has_sentiment': False}
            }

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sentiment_analysis_enabled" not in st.session_state:
        st.session_state.sentiment_analysis_enabled = True
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

def reset_chat():
    """대화 내용 초기화"""
    st.session_state.messages = []
    st.session_state.conversation_count = 0
    st.rerun()

def manage_conversation_overflow():
    """대화 기록 길이 관리"""
    max_messages = MAX_CONVERSATION_TURNS * 2
    
    if len(st.session_state.messages) >= max_messages:
        st.warning(f"!!! 대화가 {MAX_CONVERSATION_TURNS}턴을 초과했습니다.")
        
        # 자동으로 오래된 대화 70% 유지
        keep_count = int(max_messages * 0.7)
        removed_count = len(st.session_state.messages) - keep_count
        st.session_state.messages = st.session_state.messages[-keep_count:]
        
        st.success(f"!!! 오래된 대화 {removed_count}개를 자동 삭제했습니다.")

def display_conversation_history():
    """대화 기록 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 감정 분석 결과 표시 (어시스턴트 메시지만)
            if (message["role"] == "assistant" and 
                message.get("sentiment_display") and
                st.session_state.sentiment_analysis_enabled):
                st.caption(f"챗봇 답변의 감정 분석: {message['sentiment_display']}")

def setup_sidebar():
    """사이드바 설정"""
    with st.sidebar:

        st.header("💬 대화 관리")
        
        if st.session_state.messages:
            turn_count = len(st.session_state.messages) // 2
            st.write(f"대화 횟수: {turn_count}/{MAX_CONVERSATION_TURNS}")
            
            if st.button("🗑️ 대화 기록 초기화", type="secondary"):
                reset_chat()
        else:
            st.write("대화 기록이 없습니다.")
        
        st.markdown("---")
        # st.header("⚙️ 설정")
        
        # # 감정 분석 토글 (현재는 항상 활성화)
        # st.session_state.sentiment_analysis_enabled = st.checkbox(
        #     "감정 분석 활성화",
        #     value=True,
        #     help="챗봇 답변에 대한 감정 분석을 표시합니다."
        # )

@st.cache_resource
def load_chatbot():
    """챗봇 시스템 로드 (캐싱)"""
    return Chatbot()

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="Solar LLM Chatbot + 감정 분석",
        page_icon="💠",
        layout="wide"
    )
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 제목
    st.title("💠 Solar LLM Chatbot + 감정 분석")
    st.write("이전 대화를 기억하고 감정 분석 기능이 포함된 챗봇입니다.")
    
    # 레이아웃 구성
    col_main, col_sidebar = st.columns([4, 1])
    
    with col_sidebar:
        setup_sidebar()
    
    with col_main:
        try:
            # 챗봇 시스템 로드
            chatbot = load_chatbot()
            
            # 대화 기록 표시
            st.header("💬 대화")
            if st.session_state.messages:
                with st.container(height=400):
                    display_conversation_history()
            else:
                st.info("💡 안녕하세요! 궁금한 것이 있으면 언제든 물어보세요.")
            
            # 사용자 입력 처리
            if user_input := st.chat_input("질문을 입력하세요!"):
                # 대화 길이 관리
                manage_conversation_overflow()
                
                # 사용자 메시지 추가
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input
                })
                
                # AI 응답 생성
                with st.spinner("답변을 생성하고 있습니다..."):
                    result = chatbot.generate_response(
                        user_input, 
                        st.session_state.messages
                    )
                
                if result['status'] == 'success':
                    # 성공적인 응답 처리
                    ai_response = result['response']
                    sentiment_info = result['sentiment_info']
                    
                    # 어시스턴트 메시지 저장
                    assistant_message = {
                        "role": "assistant",
                        "content": ai_response
                    }
                    
                    # 감정 분석 정보 추가
                    if sentiment_info['has_sentiment']:
                        assistant_message.update({
                            'sentiment': sentiment_info['sentiment'],
                            'sentiment_display': sentiment_info['sentiment_display']
                        })
                    
                    st.session_state.messages.append(assistant_message)
                    st.session_state.conversation_count += 1
                    
                else:
                    # 오류 처리
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"--- 오류: {result.get('error', '알 수 없는 오류')}"
                    })
                
                # 페이지 새로고침
                st.rerun()
        
        except Exception as e:
            st.error(f"시스템 오류: {e}")

if __name__ == "__main__":
    main()