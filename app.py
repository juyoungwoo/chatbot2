import streamlit as st
import os
import io
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
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
import tempfile

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Google Drive 설정
@st.cache_resource
def init_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

# PDF 파일 목록 가져오기
def get_pdf_files(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])

# Streamlit UI 구성
st.title("📄 IPR실 매뉴얼 AI 챗봇")
st.write("추가적인 자료 업데이트 희망시 주영 연구원 요청")

try:
    # 드라이브 서비스 초기화
    service = init_drive_service()
    FOLDER_ID = '1fThzSsDTeZA6Zs1VLGNPp6PejJJVydra'  # 구글 드라이브 폴더 ID
    
    # PDF 파일 목록 가져오기
    pdf_files = get_pdf_files(service, FOLDER_ID)
    
    if not pdf_files:
        st.warning("폴더에 PDF 파일이 없습니다.")
    else:
        st.info(f"총 {len(pdf_files)}개의 PDF 파일을 분석합니다.")
        
        # 모든 PDF 파일의 내용을 하나로 합치기
        @st.cache_resource
        def process_all_pdfs():
            all_texts = []
            for pdf in pdf_files:
                try:
                    # PDF 파일 다운로드
                    request = service.files().get_media(fileId=pdf['id'])
                    file_content = request.execute()
                    
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(file_content)
                        pdf_path = temp_file.name
                    
                    # PDF 로드 및 처리
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    # 파일 이름을 메타데이터에 추가
                    for doc in documents:
                        doc.metadata['source'] = pdf['name']
                    
                    all_texts.extend(documents)
                    
                    # 임시 파일 삭제
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    st.warning(f"{pdf['name']} 처리 중 오류 발생: {str(e)}")
            
            # 문서 분할
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            split_texts = text_splitter.split_documents(all_texts)
            
            # 벡터 저장소 생성
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(split_texts, embeddings)
            
            return vector_store
        
        # 모든 PDF 처리
        vector_store = process_all_pdfs()
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        # 프롬프트 설정
        system_template = """
        Use the following pieces of context to answer the users question shortly.
        Given the following summaries of a long document and a question.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        If possible, mention which document (source) the information comes from.
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
        
        # LLM 모델 설정
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # QA 체인 설정
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        
        # 사용자 질문 입력
        query = st.text_input("질문을 입력하세요:")
        if st.button("질문하기") and query:
            with st.spinner("답변 생성 중..."):
                result = chain({"question": query}, return_only_outputs=True)
                answer = result["answer"]
                st.markdown(answer)

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
