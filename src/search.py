# src/search.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS (OBRIGATÓRIAS):
- Responda SOMENTE com base no CONTEXTO.
- Se a informação não estiver explícita no CONTEXTO, responda exatamente:
  "Não tenho informações necessárias para responder sua pergunta."
- NÃO inclua anos, datas, números ou unidades que não estejam escritos no CONTEXTO.
- NÃO reescreva valores monetários: repita-os exatamente como aparecem (símbolo, pontuação e casas decimais).
- NÃO opine, NÃO interprete, NÃO resuma além do necessário.
- Responda de forma curta e direta (1 a 2 linhas no máximo).

EXEMPLOS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

AGORA RESPONDA A "PERGUNTA DO USUÁRIO"
"""

# DB
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "vector_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
COLLECTION = os.getenv("PGVECTOR_COLLECTION", "documents")
TOP_K = int(os.getenv("TOP_K", "10"))

# Providers
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Verbose e threshold
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1e9"))  # grande por padrão (não filtra)

def _get_embeddings():
    if EMBEDDINGS_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return OpenAIEmbeddings(model="text-embedding-3-small")

def _get_llm():
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    return ChatOpenAI(model="gpt-5-nano", temperature=0)

def _get_vectorstore(embeddings):
    conn_str = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=conn_str,
        use_jsonb=True,
        pre_delete_collection=False,
    )

def search_prompt(question: str):
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)
    llm = _get_llm()

    # Busca vetorial
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    trechos = []
    debug_linhas = []
    pages_usadas = []

    # Heurística: score é DISTÂNCIA (menor = melhor)
    APLICAR_THRESHOLD = os.getenv("APLICAR_THRESHOLD", "false").lower() == "true"
    DIST_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1e9"))

    for doc, score in results:
        if APLICAR_THRESHOLD and score > DIST_THRESHOLD:
            continue
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        pages_usadas.append(page)
        trechos.append(f"[p{page}] {doc.page_content}")
        debug_linhas.append(f"- p{page} | score={score:.4f} | {src}")

    # monta o contexto ANTES de checar vazio
    contexto = "\n\n---\n\n".join(trechos) if trechos else "(nenhum resultado relevante encontrado)"

    # early-return: evita chamar LLM sem contexto
    if contexto == "(nenhum resultado relevante encontrado)":
        return "Não tenho informações necessárias para responder sua pergunta.", []

    if VERBOSE:
        print("\n[DEBUG] Contexto usado:")
        for l in debug_linhas:
            print(l)
        print()

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(contexto=contexto, pergunta=question)
    resp = llm.invoke(formatted)

    return resp.content, sorted(set(pages_usadas))
