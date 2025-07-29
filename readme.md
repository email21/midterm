# Solar LLM Chatbot + 감정 분석
1. Streamlit + LangChain 대화 기억 챗봇 (RAG X, 기본 Prompt)
2. 허깅페이스 파이프라인 활용 감정 분석 기능
3. 깃허브 업로드 및 배포

## 배포 링크
<br>https://midterm-3g6ciqiee6jnzqwygkzusa.streamlit.app/

## 설치 및 실행
```
git clone https://github.com/email21/midterm.git
pip install -r requirements.txt
streamlit run app.py
```

## 환경 설정
.env 파일에 SOLAR_API_KEY 설정
```
SOLAR_API_KEY=your_solar_api_key_here
```

## 파일
app.py                       # 메인 애플리케이션
sentiment_analysis_system.py # 감정 분석 모듈
