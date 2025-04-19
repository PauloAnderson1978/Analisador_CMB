import os
import hashlib
import tempfile
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# CONFIGURAÇÃO DA PÁGINA
# =============================================
st.set_page_config(
    page_title="📑 Analisador de Regulamentos Pro",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CORREÇÃO INTEGRADA PARA O ERRO removeChild
# =============================================
def apply_removechild_fix():
    """Aplica uma correção para o erro removeChild usando state."""
    if 'removechild_patched' not in st.session_state:
        components.html("""
        <script>
        (function() {
            const patchRemoveChild = () => {
                if (!window.removeChildPatched) {
                    const originalRemoveChild = Node.prototype.removeChild;
                    Node.prototype.removeChild = function(child) {
                        if (!this.contains(child)) {
                            console.debug('[Streamlit Fix] Prevented removeChild error');
                            return child;
                        }
                        return originalRemoveChild.apply(this, arguments);
                    };
                    window.removeChildPatched = true;
                    console.log('[Streamlit Fix] removeChild patch applied');
                }
            };

            // Executa imediatamente
            patchRemoveChild();

            // Reaplica após atualizações do Streamlit (mais robusto)
            document.addEventListener('DOMContentLoaded', patchRemoveChild);
            window.addEventListener('load', patchRemoveChild);
        })();
        </script>
        """, height=0, width=0)
        st.session_state['removechild_patched'] = True

apply_removechild_fix()

# =============================================
# ESTILOS CSS 
# =============================================
st.markdown("""
<style>
    :root {
        --primary: #1B5E20;
        --secondary: #43A047;
        --light-green: #E8F5E9;
        --text-dark: #212121;
        --text-light: #FFFFFF;
        --accent-green: #66BB6A;
    }

    body {
        font-family: sans-serif;
        color: var(--text-dark) !important;
    }

    .stApp {
        background-color: #f5f5f5 !important;
    }

    h1, h2, h3 {
        color: var(--primary) !important;
    }

    .stButton>button {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s !important;
        background-color: var(--primary) !important;
        color: var(--text-light) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: var(--secondary) !important;
    }

    .response-box {
        padding: 1.5rem;
        background-color: white;
        border-radius: 12px;
        border-left: 4px solid var(--secondary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        color: var(--text-dark) !important;
    }

    .history-item {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        background-color: #F0F2F6;
        transition: all 0.2s;
        color: var(--text-dark);
    }

    .history-item:hover {
        background-color: #E1E4E8;
    }

    .history-timestamp {
        font-size: 0.8rem;
        color: #757575;
    }

    .st-emotion-cache-1y4p8pa {
        width: 100%;
        padding: 1rem;
    }

    .st-emotion-cache-16txtl3 {
        background-color: #E8F5E9 !important;
        color: var(--primary) !important;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    div[data-testid="metric-label"] {
        color: var(--secondary) !important;
    }

    div[data-testid="metric-value"] {
        color: var(--text-dark) !important;
    }

    .stCaption {
        color: var(--text-dark) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# CONFIGURAÇÃO DA API (ATUALIZADA)
# =============================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"""
    ❌ Erro na configuração da API:
    {str(e)}
    
    Por favor:
    1. Crie um arquivo .streamlit/secrets.toml
    2. Adicione: GEMINI_API_KEY = "sua_chave_aqui"
    """)
    st.stop()

# =============================================
# FUNÇÕES AUXILIARES
# =============================================
def reset_question_state():
    st.session_state.last_question = ""
    st.session_state.show_response = None
    st.session_state.question_text = ""

def clear_history():
    st.session_state.history = []

def add_to_history(question, answer):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.history = st.session_state.history[-5:]

@st.cache_resource
def get_embeddings(_api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=_api_key
    )

def process_pdf(file_path: str, _api_key: str):
    with st.status("📄 Processando documento...", expanded=True) as status:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)

            doc_hash = hashlib.sha256()
            for page in pages:
                doc_hash.update(page.page_content.encode())
            doc_hash = doc_hash.hexdigest()

            vectorstore = FAISS.from_documents(chunks, get_embeddings(_api_key))

            status.update(
                label=f"✅ Documento processado! (ID: {doc_hash[:12]}...)",
                state="complete",
                expanded=False
            )

            return vectorstore, doc_hash, len(pages), len(chunks)
        except Exception as e:
            status.update(label="❌ Falha no processamento", state="error")
            st.error(f"Erro no processamento: {str(e)}")
            return None, None, 0, 0

# =============================================
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# =============================================
if 'vectorstore' not in st.session_state:
    st.session_state.update({
        'vectorstore': None,
        'doc_hash': None,
        'page_count': 0,
        'chunk_count': 0,
        'last_question': "",
        'show_response': None,
        'history': [],
        'current_file': None,
        'question_text': ""
    })

# =============================================
# INTERFACE PRINCIPAL
# =============================================
def main():
    st.title("📑 Analisador de Leis e Regulamentos")
    st.markdown("Carregue um PDF e faça perguntas sobre o conteúdo.")

    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "📤 Carregar regulamento (PDF)",
        type="pdf",
        help="Envie um documento PDF para análise",
        key="file_uploader"
    )

    if uploaded_file and (st.session_state.current_file != uploaded_file.getvalue()):
        st.session_state.current_file = uploaded_file.getvalue()
        st.session_state.vectorstore = None

        if uploaded_file.size > 10_000_000:
            st.warning("Arquivos acima de 10MB podem demorar mais para processar.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        vectorstore, doc_hash, page_count, chunk_count = process_pdf(tmp_file_path, api_key)
        os.unlink(tmp_file_path)

        if vectorstore:
            st.session_state.update({
                'vectorstore': vectorstore,
                'doc_hash': doc_hash,
                'page_count': page_count,
                'chunk_count': chunk_count
            })

            with st.expander("📊 Resumo do documento"):
                col1, col2 = st.columns(2)
                col1.metric("Páginas", page_count)
                col2.metric("Trechos", chunk_count)
                st.caption(f"ID do documento: {doc_hash[:24]}...")

    if st.session_state.vectorstore:
        st.markdown("---")

        with st.form(key='question_form'):
            question = st.text_input(
                "💡 Faça sua pergunta sobre o regulamento:",
                value=st.session_state.question_text,
                placeholder="Ex: Quais são os requisitos para aprovação?",
                key="question_input"
            )

            col1, col2 = st.columns([4, 1])
            with col1:
                submit_button = st.form_submit_button(
                    "🔍 Analisar",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                new_question_button = st.form_submit_button(
                    "🔄 Nova Pergunta",
                    on_click=reset_question_state,
                    use_container_width=True
                )

        if submit_button and question:
            with st.spinner("🤖 Analisando pergunta..."):
                try:
                    prompt_template = """
                    Você é um especialista em análise de documentos regulatórios.
                    Responda em português (Brasil) com tom profissional.

                    Contexto:
                    {context}

                    Pergunta:
                    {question}

                    Instruções:
                    - Formate a resposta com Markdown
                    - Destaque artigos/seções com `código`
                    - Use **negrito** para pontos importantes
                    - Se não souber, diga "Não encontrado no documento"
                    """

                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGoogleGenerativeAI(
                            model="gemini-1.5-pro-latest",
                            temperature=0.3,
                            google_api_key=api_key
                        ),
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(),
                        chain_type_kwargs={"prompt": prompt}
                    )

                    result = qa_chain({"query": question})

                    if result and 'result' in result:
                        st.session_state.last_question = question
                        st.session_state.show_response = result['result']
                        st.session_state.question_text = ""
                        add_to_history(question, result['result'])
                    else:
                        st.error("Não foi possível obter uma resposta.")

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a análise: {str(e)}")

        if st.session_state.show_response:
            st.markdown(f"""
            <div class="response-box">
                <h3 style='color: #1a1a1a; margin-top: 0;'>📝 Resposta</h3>
                {st.session_state.show_response}
            </div>
            """, unsafe_allow_html=True)

            if st.button("❌ Limpar Resposta", on_click=reset_question_state):
                pass

            st.subheader("🔍 Trechos de referência")
            docs = st.session_state.vectorstore.similarity_search(
                st.session_state.last_question,
                k=3
            )

            for i, doc in enumerate(docs):
                with st.expander(f"Trecho {i+1} (Página {doc.metadata.get('page', 'N/A')})"):
                    st.write(doc.page_content)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem; background-color: #0a5c0a; color: white; border-radius: 12px;'>
            <h3 style='color: var(--text-light) !important;'>ℹ️ Como usar</h3>
            <ol style='padding-left: 1rem;'>
                <li>Carregue um PDF regulatório</li>
                <li>Espere o processamento</li>
                <li>Faça perguntas sobre o conteúdo</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📚 Histórico (Últimas 5)")

        if st.button("🧹 Limpar Todo o Histórico", use_container_width=True, on_click=clear_history):
            pass

        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.container():
                    st.markdown(f"""
                    <div class="history-item">
                        <div class="history-timestamp">{item['timestamp']}</div>
                        <p><strong>Pergunta:</strong> {item['question'][:60]}{'...' if len(item['question']) > 60 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Ver resposta", key=f"view_{i}", on_click=lambda item=item: st.session_state.update(show_response=item['answer'], last_question=item['question'])):
                        pass
        else:
            st.caption("Nenhuma pergunta no histórico")

if __name__ == "__main__":
    main()
