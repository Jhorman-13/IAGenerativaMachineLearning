import io
import platform

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Configuración de la página (Debe ser la primera instrucción)
st.set_page_config(page_title="ChatPDF - RAG", page_icon="💬", layout="wide")

# --- INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Sube un PDF y pregúntame lo que quieras sobre él."}]

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.title("⚙️ Configuración")
    try:
        st.image(Image.open("Chat_pdf.png"), width=250)
    except Exception:
        pass # Si no encuentra la imagen, simplemente no la muestra y no rompe el diseño
    
    st.markdown("---")
    # Clave de API
    api_key = st.text_input("🔑 Clave de OpenAI", type="password", help="Ingresa tu API Key para poder chatear.")
    
    # Carga del PDF
    pdf = st.file_uploader("📄 Carga tu archivo PDF", type="pdf")
    
    st.markdown("---")
    st.info(f"🐍 Versión de Python: {platform.python_version()}")
    
    # Botón para limpiar el chat
    if st.button("🗑️ Limpiar Conversación"):
        st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Sube un PDF y pregúntame lo que quieras sobre él."}]
        st.rerun()

# --- FUNCIÓN DE CACHÉ PARA RAG ---
@st.cache_resource(show_spinner=False)
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae el texto, lo divide en fragmentos y crea el índice FAISS."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base

# --- INTERFAZ PRINCIPAL DEL CHAT ---
st.title("💬 Chatbot RAG con PDF")

if not api_key:
    st.warning("👈 Por favor, ingresa tu clave de API de OpenAI en el menú lateral para comenzar.")
elif pdf is None:
    st.info("👈 Por favor, carga un archivo PDF en el menú lateral para analizarlo.")
else:
    with
