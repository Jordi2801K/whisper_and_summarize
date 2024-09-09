import streamlit as st
from openai import OpenAI
import PyPDF2

# # 이거는 streamlit에 배포할 때 쓰는 api key 설정
api_key = st.secrets["API_KEY"]
# OpenAI 클라이언트 설정
client = OpenAI(api_key=api_key)

# 페이지 wide로 설정
st.set_page_config(layout="wide")

st.title("음성 인식쓰 & 자료 요약하기")

# 사이드바에서 기능 선택
option = st.sidebar.radio(
    "고르시오",
    ("음성 인식", "텍스트 요약")
)

# 텍스트 다운로드 함수
def download_text(text, filename):
    st.download_button(
        label="파일 다운로드",
        data=text,
        file_name=filename,
        mime="text/plain"
    )

# 음성 파일을 텍스트로 변환하는 함수
def transcribe_with_api(audio_file):
    audio_file.seek(0)
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return response.text

# 텍스트 파일 및 PDF 파일에서 텍스트를 추출하는 함수
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":  # txt 파일 처리
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":  # pdf 파일 처리
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return None

# 텍스트 요약
def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
You are the world's foremost expert in every field and major of university education. Your task is to receive notes and lecture recordings provided by university students and summarize them with absolute precision. The summary must not contain any incorrect information, wrong concepts, or hallucinations under any circumstances.

The student will study and prepare for exams based on your summary, so the summary must be extremely accurate. You should only summarize from the content provided by the student.

When summarizing, first carefully read and analyze the entire content thoroughly. Then, group similar or related content and summarize it accordingly. Typically, similar content will be located closely together in the text.

Assign a number to each group of similar or related content and summarize each within that group. When a different topic or concept arises, assign it a new number. At the beginning of each numbered section, provide a short title describing the content. Within each section, use bullet points to make the summary clear and easy to follow.

The summary should be as detailed as possible, minimizing omissions.

After summarizing each group of related content, provide an overall summary at the end that encapsulates the entire document. Additionally, highlight the most important points or concepts that the student should focus on while studying.

Once again, when summarizing, you must only use the content provided in the document, and the summary must not contain any incorrect information or hallucinations.

You must write the response in Korean.
"""
},
            {"role": "user", "content": f"Summarize the following content strictly according to the System Prompt:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content


# 음성 인식 섹션
if option == "음성 인식":
    st.header("음성 파일을 텍스트로 변환")
    file = st.file_uploader("음성 파일 업로드", type=["wav", "mp3"])
    if file:
        with st.spinner('Whisper API로 텍스트 변환 중...'):
            transcript = transcribe_with_api(file)
            st.success("변환 완료!")
            st.text_area("변환된 텍스트 미리보기", transcript[:500], height=300)
            download_text(transcript, "transcription.txt")  # 변환된 텍스트 다운로드

# 텍스트 요약 섹션
elif option == "텍스트 요약":
    st.header("텍스트 요약")
    file = st.file_uploader("요약할 파일 업로드 (txt, pdf)", type=["txt", "pdf"])
    
    if file:
        with st.spinner('파일에서 텍스트 추출 중...'):
            text = extract_text_from_file(file)
            if text:
                st.text_area("추출된 텍스트 미리보기", text[:500], height=300)

                # 요약 버튼을 누르면 요약 실행
                if st.button("요약하기"):
                    with st.spinner('텍스트 요약 중...'):
                        summary = summarize_text(text)
                        st.success("요약 완료!")
                        st.text_area("요약된 텍스트 미리보기", summary[:200], height=200)
                        download_text(summary, "summary.txt")  # 요약된 텍스트 다운로드
            else:
                st.error("지원하지 않는 파일 형식입니다.")
