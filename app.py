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
    with st.spinner("Procesando el documento y creando la base de conocimiento..."):
        try:
            base = crear_base_conocimiento(pdf.getvalue(), api_key)
            if base is None:
                st.error("No se pudo extraer texto del PDF (¿es un PDF escaneado sin OCR?).")
                st.stop()
        except Exception as e:
            st.error(f"Error al procesar el PDF: {e}")
            st.stop()

    # Mostrar el historial del chat
    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    # Entrada de texto del usuario
    if pregunta := st.chat_input("Escribe tu pregunta sobre el documento..."):
        
        # 1. Mostrar la pregunta del usuario y guardarla en el historial
        with st.chat_message("user"):
            st.markdown(pregunta)
        st.session_state.mensajes.append({"role": "user", "content": pregunta})

        # 2. Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Buscando en el documento..."):
                try:
                    # Recuperación
                    docs = base.similarity_search(pregunta, k=4)
                    contexto = "\n\n---\n\n".join(d.page_content for d in docs)

                    # Generación
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
                    mensajes_llm = [
                        ("system", 
                         "Eres un asistente útil. Responde la pregunta usando únicamente el contexto proporcionado. "
                         "Si la respuesta no está en el contexto, dilo claramente. Responde en español y de forma conversacional."),
                        ("human", f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
                    ]
                    
                    respuesta = llm.invoke(mensajes_llm)
                    
                    # Mostrar la respuesta
                    st.markdown(respuesta.content)
                    
                    # Mostrar las fuentes de forma colapsable
                    with st.expander("📚 Ver fragmentos recuperados"):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**Fragmento {i}**")
                            st.caption(d.page_content)
                    
                    # Guardar respuesta en el historial
                    st.session_state.mensajes.append({"role": "assistant", "content": respuesta.content})
                    
                except Exception as e:
                    st.error(f"Hubo un error al generar la respuesta: {e}")
