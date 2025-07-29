import streamlit as st
from transformers import pipeline
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

class SentimentAnalysis:
    """한국어 감정 분석"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self._initialize_system()
    
    def _initialize_system(self):
        """감정 분석 시스템 초기화"""
        try:
            self.sentiment_pipeline = self._load_korean_sentiment_pipeline()
            print("--- 감정 분석 시스템 초기화 완료")
        except Exception as e:
            print(f"!!!!! 감정 분석 시스템 초기화 실패: {e}")
            raise
    
    @st.cache_resource
    def _load_korean_sentiment_pipeline(_self):
        """한국어 특화 감정 분석 파이프라인 로드"""
        try:
            model_name = "nlp04/korean_sentiment_analysis_kcelectra"
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1  # CPU 사용
            )
            return sentiment_pipeline
        except Exception as e:
            st.error(f"한국어 감정 분석 모델 로드 실패: {e}")
            return None
    
    def _convert_to_simple_sentiment(self, sentiment_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """11개 세분화된 감정을 긍정/중립/부정 3개로 변환"""
        if not sentiment_result:
            return None
        
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        # 감정 그룹 매핑
        positive = ['기쁨(행복한)', '고마운', '설레는(기대하는)', '사랑하는', '즐거운(신나는)']
        neutral = ['일상적인', '생각이 많은']
        negative = ['슬픔(우울한)', '힘듦(지침)', '짜증남', '걱정스러운(불안한)']
        
        # 3개 그룹으로 변환
        if label in positive:
            simple_label = "긍정"
        elif label in neutral:
            simple_label = "중립"
        elif label in negative:
            simple_label = "부정"
        else:
            simple_label = "알 수 없음"
        
        return {
            'original_label': label,
            'simple_label': simple_label,
            'score': score
        }
    
    def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """한국어 감정 분석 실행"""
        if self.sentiment_pipeline is None:
            return None
        
        try:
            # 텍스트 길이 제한 (모델의 최대 시퀀스 길이: 512 토큰)
            if len(text) > 400:
                text = text[:400]
            
            result = self.sentiment_pipeline(text)
            return {
                'label': result[0]['label'],
                'score': result[0]['score']
            }
        except Exception as e:
            st.error(f"감정 분석 중 오류: {e}")
            return None
    
    def get_sentiment_display(self, sentiment_result: Dict[str, Any], 
                            strict_threshold: float = 0.7, 
                            neutral_threshold: float = 0.55) -> str:
        """감정 분석 결과 표시 문자열 생성"""
        converted = self._convert_to_simple_sentiment(sentiment_result)
        if not converted:
            return ""
        
        simple_label = converted['simple_label']
        score = converted['score']
        original_label = converted['original_label']
        
        # 신뢰도 레벨 결정
        if score >= strict_threshold:
            confidence_level = "높음"
        elif score >= neutral_threshold: # 55-70%
            confidence_level = "보통"
        else:
            confidence_level = "낮음" # 55% 미만, 거의 50:50에 가까워 판단이 애매함
            return f"불확실 [ 원본: {original_label}, 신뢰도: {confidence_level}, {score*100:.1f}% ]"
        
        return f"{simple_label} [ 원본: {original_label}, 신뢰도: {confidence_level}, {score*100:.1f}% ]"
    
    def process_message_sentiment(self, message_content: str) -> Dict[str, Any]:
        """메시지의 감정 분석 처리"""
        sentiment_result = self.analyze_sentiment(message_content)
        
        if sentiment_result:
            sentiment_display = self.get_sentiment_display(sentiment_result)
            return {
                'sentiment': sentiment_result,
                'sentiment_display': sentiment_display,
                'has_sentiment': True
            }
        else:
            return {
                'sentiment': None,
                'sentiment_display': None,
                'has_sentiment': False
            }
