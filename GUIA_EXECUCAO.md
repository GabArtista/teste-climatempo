# Guia de Execução e Teste Manual — Weather LLM Agent

> Leia do início ao fim antes de começar. O passo-a-passo cobre setup, execução e todos os testes manuais.

---

## PRÉ-REQUISITOS

| Requisito | Verificar com | Esperado |
|-----------|--------------|----------|
| Python 3.11+ | `python3 --version` | 3.11 ou superior |
| Ollama | `ollama --version` | instalado |
| Modelo qwen2.5:1.5b | `ollama list` | aparece na lista |
| Node.js 18+ | `node --version` | 18 ou superior |
| Internet | `curl api.open-meteo.com` | responde |

Se o modelo não estiver na lista:
```bash
ollama pull qwen2.5:1.5b
```

---

## PASSO 1 — BACKEND

### 1.1 Setup (apenas na primeira vez)
```bash
cd teste/backend

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Criar arquivo de configuração
cp .env.example .env
```

### 1.2 Iniciar o Ollama (terminal separado — deixar aberto)
```bash
ollama serve
```

### 1.3 Iniciar o backend
```bash
# Ative o venv se não estiver ativo
source .venv/bin/activate

# Inicia o servidor
uvicorn main:app --reload --port 8000
```

**Saída esperada:**
```
INFO     | Starting Weather LLM Agent v1.0.0
INFO     | Ollama model: qwen2.5:1.5b @ http://localhost:11434/v1
INFO     | Application startup complete.
INFO     | Uvicorn running on http://127.0.0.1:8000
```

---

## PASSO 2 — FRONTEND

### 2.1 Setup (apenas na primeira vez)
```bash
cd teste/frontend
npm install
```

### 2.2 Iniciar o frontend (terminal separado)
```bash
npm run dev
```

**Saída esperada:**
```
VITE v5.x.x  ready in 300 ms
➜  Local:   http://localhost:5173/
```

Abra **http://localhost:5173** no browser.

---

## PASSO 3 — TESTES MANUAIS

### 3.1 Swagger (API interativa)
Abra **http://localhost:8000/docs**

Endpoints disponíveis:
- `POST /api/v1/agent/chat` — chat com o agente
- `GET /api/v1/agent/health` — status do Ollama
- `GET /api/v1/weather/` — previsão direta (sem LLM)
- `GET /api/v1/weather/cities` — lista as 26 capitais
- `GET /api/v1/weather/data-quality` — anomalias do capitals.json

---

### 3.2 Teste via curl

**Cenário 1 — Pergunta com cidade (tool DEVE ser chamada)**
```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como está o tempo em Curitiba nos próximos 3 dias?", "history": []}'
```
✅ Esperado: `"tool_called": true`, resposta com temperaturas e datas

---

**Cenário 2 — Saudação (tool NÃO deve ser chamada)**
```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá, tudo bem?", "history": []}'
```
✅ Esperado: `"tool_called": false`, resposta conversacional

---

**Cenário 3 — Pergunta sem cidade (agente deve pedir a cidade)**
```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Vai chover amanhã?", "history": []}'
```
✅ Esperado: `"tool_called": false`, resposta perguntando qual cidade

---

**Cenário 4 — Previsão direta (bypassa o LLM)**
```bash
curl "http://localhost:8000/api/v1/weather/?city=S%C3%A3o+Paulo&days=3"
```
✅ Esperado: JSON com `forecasts` contendo 3 dias, temperaturas em °C, precipitação em mm

---

**Cenário 5 — Cidade inválida**
```bash
curl "http://localhost:8000/api/v1/weather/?city=Mordor&days=3"
```
✅ Esperado: HTTP 404 com mensagem de erro amigável

---

**Cenário 6 — Anomalia do capitals.json**
```bash
curl http://localhost:8000/api/v1/weather/data-quality
```
✅ Esperado: JSON listando "Campo Grande - Rio Grande do Norte" como entrada incorreta

---

**Cenário 7 — Multi-turn (conversa em múltiplos turnos)**
```bash
# Turno 1: perguntar sem cidade
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quero saber a previsão do tempo", "history": []}'

# Turno 2: fornecer a cidade (com histórico)
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Para Fortaleza, por favor",
    "history": [
      {"role": "user", "content": "Quero saber a previsão do tempo"},
      {"role": "assistant", "content": "Claro! Qual cidade você gostaria de consultar?"}
    ]
  }'
```
✅ Esperado no turno 2: `"tool_called": true`, previsão de Fortaleza

---

### 3.3 Teste no Frontend (http://localhost:5173)

1. **Tela inicial** — deve mostrar 4 sugestões de perguntas
2. **Clique em uma sugestão** — deve enviar a pergunta automaticamente
3. **Aguarde a resposta** — animação de loading (3 pontos piscando)
4. **Resposta com clima** — deve mostrar o WeatherCard com os dias
5. **Badge "⚡ Open-Meteo"** — aparece nas mensagens que usaram a tool
6. **Digite saudação** — resposta sem WeatherCard, badge não aparece
7. **Botão 🗑️** — limpa a conversa

---

## PASSO 4 — TESTES AUTOMATIZADOS

### Testes unitários e de feature (não precisam do Ollama)
```bash
cd teste/backend
source .venv/bin/activate
pytest tests/Unit/ tests/Feature/ -v
```
✅ Esperado: **30/30 PASSED**

### Suíte de validação F1 (precisa do Ollama rodando — ~8 minutos)
```bash
# Certifique-se que o backend está rodando em :8000
pytest tests/Validation/ -v -s
```
✅ Esperado: 1 teste PASSED, resultados salvos em `tests/Validation/results.json`

---

## PASSO 5 — DOCKER (opcional — desafio extra)

### Pré-requisito: Docker + Docker Compose instalados
```bash
docker --version
docker compose version
```

### Subir tudo com Docker
```bash
cd teste
docker compose up --build
```

> **Atenção:** O Ollama precisa estar rodando no host (fora do Docker).
> O `docker-compose.yml` já aponta para `host.docker.internal:11434`.

Acesse:
- Frontend: **http://localhost:5173**
- Backend: **http://localhost:8000/docs**

---

## TROUBLESHOOTING

| Problema | Causa | Solução |
|----------|-------|---------|
| `Connection refused :11434` | Ollama não está rodando | `ollama serve` |
| `503 Service Unavailable` | Ollama sem o modelo | `ollama pull qwen2.5:1.5b` |
| `404 Cidade não encontrada` | Nome de cidade inválido | Use uma capital brasileira |
| Resposta muito lenta (>60s) | CPU inference sem GPU | Normal — modelo pequeno em CPU |
| Frontend não conecta ao backend | CORS ou porta errada | Backend deve estar em :8000 |

---

## RESUMO DOS ENDPOINTS

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Info da API |
| POST | `/api/v1/agent/chat` | Chat com agente LLM |
| GET | `/api/v1/agent/health` | Status do Ollama |
| GET | `/api/v1/weather/` | Previsão direta |
| GET | `/api/v1/weather/cities` | Lista capitais (26) |
| GET | `/api/v1/weather/data-quality` | Anomalias capitals.json |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |
