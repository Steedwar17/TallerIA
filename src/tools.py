"""
tools.py — Herramientas (@tool) para el Agente de Triaje Inteligente.
Ruta 3: PLN con Embeddings + Agente de Triaje (IMDB Reviews).
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

try:
    from langchain_core.tools import tool
    from langchain_core.documents import Document
except ImportError:
    # Fallback decorador si langchain_core aún no está instalado en el entorno
    def tool(fn):
        fn.is_tool = True
        return fn
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from src.policies import POLITICAS_TRIAJE


def _formatear_salida_compacta(politica: dict) -> str:
    """Formatea la política en un resumen ultra compacto para optimizar tokens."""
    return f"[{politica['id']}] {politica['contenido']}"


@tool
def consultar_politicas_triaje(categoria: str, consulta: str = "") -> str:
    """Consulta las políticas internas de triaje y pautas de respuesta corporativas.

    Úsala cuando conozcas el sentimiento o categoría del ticket/reseña (ej. 'positivo',
    'negativo', o 'neutral'/'incierto') y necesites determinar la prioridad, acción a seguir
    y pauta de respuesta según los protocolos oficiales.

    Args:
        categoria: Clasificación del sentimiento o estado ('positivo', 'negativo', 'neutral' o 'incierto').
        consulta: Detalle opcional o pregunta específica sobre el caso para refinar la recuperación.

    Returns:
        Resumen compacto con: Identificador, Prioridad, Acción requerida y Pauta de respuesta recomendada.
    """
    cat_normalizada = categoria.strip().lower()
    
    # Mapeo de términos equivalentes a las categorías base
    if cat_normalizada in ["negativo", "negative", "malo", "critica", "queja"]:
        target_cat = "negativo"
    elif cat_normalizada in ["positivo", "positive", "bueno", "favorable", "promotor"]:
        target_cat = "positivo"
    else:
        target_cat = "neutral"

    # 1. Intento de recuperación semántica con RAG (Google Embeddings + InMemoryVectorStore) si hay API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "tu_clave_aqui":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_core.vectorstores import InMemoryVectorStore

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
            vector_store = InMemoryVectorStore(embeddings)

            docs = [
                Document(
                    page_content=p["contenido"],
                    metadata={"id": p["id"], "category": p["category"], "titulo": p["titulo"]},
                )
                for p in POLITICAS_TRIAJE
            ]
            vector_store.add_documents(docs)

            # Búsqueda semántica combinando categoría y consulta
            query_text = f"Categoría {target_cat}. {consulta}".strip()
            resultados = vector_store.similarity_search(query_text, k=1)
            if resultados:
                doc = resultados[0]
                pol_id = doc.metadata.get("id", "POLITICA")
                return f"[{pol_id}] {doc.page_content}"
        except Exception:
            # Si falla la llamada externa o cuota, recurre al retriever local por metadatos
            pass

    # 2. Recuperador ligero en memoria por categoría / metadatos (robusto y determinista)
    for pol in POLITICAS_TRIAJE:
        if pol["category"] == target_cat:
            return _formatear_salida_compacta(pol)

    # Fallback por defecto si no coincide
    return _formatear_salida_compacta(POLITICAS_TRIAJE[2])

