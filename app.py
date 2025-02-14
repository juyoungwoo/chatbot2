import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ✅ Groq API 키 설정
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ✅ 임베딩 모델 캐싱 (메모리 절약)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

# ✅ Google Drive API 초기화
@st.cache_resource
def init_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

# ✅ PDF 파일 목록 가져오기
def get_pdf_files(service, folder_id):
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Google Drive API 오류: {str(e)}")
        return []

# ✅ Streamlit UI 시작
st.title("📄 IPR실 매뉴얼 AI 챗봇")
st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

try:
    # ✅ Google Drive에서 PDF 목록 가져오기
    service = init_drive_service()
    FOLDER_ID = '1fThzSsDTeZA6Zs1VLGNPp6PejJJVydra'
    pdf_files = get_pdf_files(service, FOLDER_ID)

    if not pdf_files:
        st.warning("📂 매뉴얼 폴더에 PDF 파일이 없습니다.")
    else:
        st.info(f"📄 총 {len(pdf_files)}개의 매뉴얼을 분석 중...")

        # ✅ 모든 PDF 파일 처리 및 임베딩 벡터 저장
        @st.cache_resource
        def process_all_pdfs():
            all_texts = []

            for pdf in pdf_files:
                try:
                    request = service.files().get_media(fileId=pdf['id'])
                    file_content = request.execute()

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(file_content)
                        pdf_path = temp_file.name

                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata['source'] = pdf['name']

                    all_texts.extend(documents)

                    os.unlink(pdf_path)  # 사용 후 파일 삭제

                except Exception as e:
                    st.warning(f"⚠️ {pdf['name']} 처리 중 오류 발생: {str(e)}")

            # ✅ 문서 분할 최적화
            text_splitter = CharacterTextSplitter(
                chunk_size=500,  
                chunk_overlap=200,  
                separator="\n"  
            )
            split_texts = text_splitter.split_documents(all_texts)

            # ✅ 벡터 저장소 생성 및 캐싱
            embeddings = get_embeddings()
            vector_store = FAISS.from_documents(split_texts, embeddings)

            return vector_store

        # ✅ 벡터 스토어 생성
        vector_store = process_all_pdfs()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 검색 결과 최적화

        # ✅ AI 프롬프트 설정
        system_template = """
        Use the following pieces of context to answer the user's question concisely.
        If you don't know the answer, just say "I don't know", don't try to make up an answer.
        If possible, mention the document (source) the information comes from.
        ----------------
        {summaries}
        You MUST answer in Korean and in Markdown format:
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain_type_kwargs = {"prompt": prompt}

        # ✅ LLM 모델 설정 최적화
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.environ["GROQ_API_KEY"],
            max_tokens=512  # 응답 길이 제한
        )

        # ✅ QA 체인 설정
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        # ✅ 사용자 질문 입력
        query = st.text_area("📝 질문을 입력하세요:", height=100)  # 긴 질문 입력 가능
        if st.button("💬 질문하기") and query.strip():
            with st.spinner("🤖 답변 생성 중..."):
                try:
                    result = chain({"question": query}, return_only_outputs=True)
                    answer = result.get("answer", "❌ 답변을 생성할 수 없습니다.")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"⚠️ 답변 생성 중 오류 발생: {str(e)}")

except Exception as e:
    st.error(f"🚨 시스템 오류 발생: {str(e)}")
