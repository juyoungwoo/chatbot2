import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Streamlit UI 구성
st.title("📄 PDF 기반 AI 챗봇")
st.write("PDF 내용을 학습한 AI 챗봇입니다.")

# PDF 업로드 기능
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # PDF 파일 저장
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 문서 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 벡터 저장소 생성
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 프롬프트 설정
    system_template = """
    Use the following pieces of context to answer the users question shortly.
    Given the following summaries of a long document and a question.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
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
