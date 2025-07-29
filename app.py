import streamlit as st
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from transformers import pipeline
from dotenv import load_dotenv
import os

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 기록을 저장할 리스트
    st.session_state.sentiment_analysis_enabled = True

def reset_chat():
    """대화 내용을 초기화하는 함수"""
    st.session_state.messages.clear()
    st.rerun()

# 한국어 감정 분석 파이프라인 초기화
@st.cache_resource
def load_korean_sentiment_pipeline():
    """한국어 특화 감정 분석 파이프라인"""
    try:
        model_name = "nlp04/korean_sentiment_analysis_kcelectra"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"한국어 감정 분석 모델 로드 실패: {e}")
        return None

def convert_to_simple_sentiment(sentiment_result):
    """11개 세분화된 감정을 긍정/중립/부정 3개로 변환"""
    if not sentiment_result:
        return None
    
    label = sentiment_result['label']
    score = sentiment_result['score']
    
    # 감정 그룹 매핑
    positive_emotions = ['기쁨(행복한)', '고마운', '설레는(기대하는)', '사랑하는', '즐거운(신나는)']
    neutral_emotions = ['일상적인', '생각이 많은']
    negative_emotions = ['슬픔(우울한)', '힘듦(지침)', '짜증남', '걱정스러운(불안한)']
    
    # 3개 그룹으로 변환
    if label in positive_emotions:
        simple_label = "긍정"
    elif label in neutral_emotions:
        simple_label = "중립"
    elif label in negative_emotions:
        simple_label = "부정"
    else:
        simple_label = "알 수 없음"
    
    return {
        'original_label': label,
        'simple_label': simple_label,
        'score': score
    }

def analyze_korean_sentiment(text: str, pipeline_model):
    """한국어 감정 분석"""
    if pipeline_model is None:
        return None
    try:
        # 텍스트가 너무 길면 앞부분만 사용
        if len(text) > 400:
            text = text[:400]
            
        result = pipeline_model(text)
        return {
            'label': result[0]['label'], 
            'score': result[0]['score']
        }
    except Exception as e:
        st.error(f"감정 분석 중 오류: {e}")
        return None

def get_sentiment_display(sentiment_result, strict_threshold=0.7, neutral_threshold=0.55):
    """감정 분석 결과 표시 """
    converted = convert_to_simple_sentiment(sentiment_result)
    if not converted:
        return ""
    
    simple_label = converted['simple_label']
    score = converted['score']
    original_label = converted['original_label']
    
    if score >= strict_threshold:  
        confidence_level = "높음"
    elif score >= neutral_threshold: # 55-70%
        confidence_level = "보통"
    else: 
        confidence_level = "낮음" # 55% 미만, 거의 50:50에 가까워 판단이 애매함
        return f" 불확실 [ 원본: {original_label}, 신뢰도: {confidence_level}, {score*100:.1f}% ]"
    
    return f" {simple_label} [ 원본: {original_label}, 신뢰도: {confidence_level}, {score*100:.1f}% ]"

# Streamlit 페이지 설정
st.title("💠 Solar LLM Chatbot + 감정 분석")
st.write("이전 대화를 기억하고 감정 분석 기능이 포함된 챗봇입니다.")

# 사이드바에 초기화 버튼 추가
with st.sidebar:
    st.header("옵션")
    if st.button("대화 초기화"):
        reset_chat()

# 모델 로드
try:
    chat_model = ChatUpstage(upstage_api_key=api_key)
    sentiment_pipeline = load_korean_sentiment_pipeline() if st.session_state.sentiment_analysis_enabled else None
except Exception as e:
    st.error(f"모델 로드 오류: {e}")
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
         # 이미 처리된 감정 분석 결과만 표시
        if message["role"] == "assistant" and message.get("sentiment_display"):
            st.caption(f"챗봇 답변의 감정 분석: {message['sentiment_display']}")

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
            
            sentiment_display = None
            if st.session_state.sentiment_analysis_enabled and sentiment_pipeline:
                sentiment_result = analyze_korean_sentiment(ai_response, sentiment_pipeline)
                if sentiment_result:
                    sentiment_display = get_sentiment_display(sentiment_result)
                    st.caption(f"챗봇 답변의 감정 분석: {sentiment_display}")
            
            # 메시지 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "sentiment": sentiment_result
            })
            
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")