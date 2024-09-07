__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="세무 챗봇", page_icon="💼", layout="centered")
st.title("세무 상담 챗봇 테스트")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vectorstore_loaded" not in st.session_state:
    st.session_state["vectorstore_loaded"] = False

if "is_loading" not in st.session_state:
    st.session_state["is_loading"] = False

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

with st.spinner("챗봇 데이터베이스를 로드 중입니다. 잠시만 기다려 주세요..."):

    vectorstore_dir = "chroma_vectorstore"

    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    if os.path.exists(vectorstore_dir):

        from chromadb.config import Settings
        from chromadb import Client

        # DuckDB로 설정
        settings = Settings(persist_directory="chroma_vectorstore", chroma_db_impl="duckdb")
        client = Client(settings)

        vectorstore = Chroma(client=client, persist_directory=vectorstore_dir, embedding_function=embedding_model)
    else:
        raise NotImplementedError()

    st.session_state["vectorstore_loaded"] = True

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

if st.session_state["vectorstore_loaded"]:

    memory = st.session_state["memory"]

    retriever = vectorstore.as_retriever()

    OPENAI_KEY = st.secrets["general"]["openai_api_key"]
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_KEY,
        streaming=True
    )

    template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
    만약 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마십시오.
    당신은 한국 세법에 대해 전문가인 세무사입니다.
    친절하게 답변해 주세요.
    맥락: {context}
    질문: {question}
    이전 대화: {chat_history}
    도움되는 답변:"""
    prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=template)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer"
    )

    def stream_answer(question):
        answer_container = st.empty()
        callback_handler = StreamlitCallbackHandler(container=answer_container)
        result = qa({"question": question}, callbacks=[callback_handler])
        return result

    question_disabled = st.session_state["is_loading"]

    with st.form(key="question_form"):
        question = st.text_input(
            "질문을 입력하세요", 
            placeholder="예: 종합소득세는 어떻게 계산하나요?", 
            disabled=question_disabled
        )

        submit_button_col, reset_button_col = st.columns([1, 1])
        with submit_button_col:
            submit_button = st.form_submit_button("질문하기", use_container_width=True)
        with reset_button_col:
            reset_button = st.form_submit_button("대화 초기화", use_container_width=True)

    if submit_button and not question_disabled:
        if question:
            st.session_state["is_loading"] = True

            with st.spinner("질문에 대한 답변을 생성 중입니다. 잠시만 기다려 주세요..."):
                result = stream_answer(question)

            answer = result['answer']
            references = "\n\n".join([doc.page_content for doc in result['source_documents']])

            with st.expander("📄 참조된 문서"):
                st.write(f"**참조된 문서:** {references}")

            st.session_state["chat_history"].append((question, answer))
            st.session_state["is_loading"] = False
        else:
            st.warning("질문을 입력해주세요.")

    if reset_button:
        with st.spinner("대화 기록을 초기화하는 중입니다..."):
            st.session_state["chat_history"] = []

            success_message = st.empty() 
            success_message.success("대화 내용이 초기화되었습니다.")
            time.sleep(3) 
            success_message.empty()

    if st.session_state["chat_history"]:
        st.write("### 💬 대화 기록")
        for i, (q, a) in enumerate(st.session_state["chat_history"]):
            with st.expander(f"질문 {i + 1}: {q}"):
                st.write(f"**답변:** {a}")
else:
    st.warning("문서 및 데이터베이스 로드를 기다리고 있습니다...")
