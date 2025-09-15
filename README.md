# Desafio – Ingestão e Busca Semântica com LangChain e Postgres

Este projeto implementa um sistema de **ingestão e busca semântica** em PDF utilizando **LangChain, PostgreSQL com extensão pgVector** e execução via **Docker Compose**.  
O usuário pode fazer perguntas via **linha de comando (CLI)** e receber respostas baseadas **somente no conteúdo do PDF**.

---

## 📌 Objetivo
- **Ingestão**: Ler um PDF, dividi-lo em chunks, gerar embeddings e armazenar no banco vetorial.
- **Busca**: Receber uma pergunta no terminal, vetorizar, buscar os trechos mais relevantes no banco e responder com base no contexto.

---

## ⚙️ Tecnologias
- **Linguagem**: Python
- **Framework**: LangChain
- **Banco**: PostgreSQL + pgVector (em Docker)
- **Embeddings e LLMs**:
  - OpenAI:  
    - Embeddings: `text-embedding-3-small`  
    - LLM: `gpt-5-nano`
  - Google Gemini:  
    - Embeddings: `models/embedding-001`  
    - LLM: `gemini-2.5-flash-lite`



## 📂 Estrutura do projeto
```

├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf # PDF a ser ingerido
├── src/
│ ├── ingest.py # Ingestão do PDF no banco vetorial
│ ├── search.py # Funções de busca semântica
│ └── chat.py # CLI para interação com o usuário
└── README.md # Este arquivo
```

---

## 🚀 Como executar

### 1. Criar ambiente virtual e instalar dependências
```
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Subir o banco de dados com pgVector
```
docker compose up -d
```

### 3. Configurar variáveis de ambiente
```
Copie o arquivo .env.example para .env e edite com suas chaves:
```

### Provedor de embeddings/LLM
```
EMBEDDINGS_PROVIDER=openai     # ou gemini
LLM_PROVIDER=openai            # ou gemini
```

### Config OpenAI
```
OPENAI_API_KEY=sk-xxxxx
```
### Config Google Gemini
```
GOOGLE_API_KEY=AIza-xxxxx
```

### Config Banco
```
PGHOST=localhost
PGPORT=5432
PGDATABASE=vector_db
PGUSER=postgres
PGPASSWORD=postgres
PGVECTOR_COLLECTION=documents
```

### Config de ingestão
```
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### Busca
```
TOP_K=10
```

### 4. Criar banco e extensão pgVector (caso não exista)
```
docker exec -it postgres_rag psql -U postgres -c "CREATE DATABASE vector_db;"
docker exec -it postgres_rag psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. Rodar a ingestão do PDF
```
python src/ingest.py
```

### 6. Rodar o chat (CLI)
```
python src/chat.py
```

### Exemplo de uso
Faça sua pergunta (Ctrl+C para sair):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00
FONTES: páginas 0, 2, 3, 10, 15, 20, 25, 26, 29

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
