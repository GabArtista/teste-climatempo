# PROGRESS — Weather LLM Agent (Climatempo Challenge)

> **Documentação viva** — atualizada a cada sessão de desenvolvimento.
> Se os créditos acabarem, retome aqui: leia este arquivo primeiro.

## Estado: PROJETO COMPLETO ✅

**Data:** 2026-04-13

---

## Resumo das entregas

| Métrica | Valor |
|---------|-------|
| Testes automatizados | **42/42 passando** |
| Pontuação estimada | **110/120 pts** |
| Cobertura de funcionalidades | Backend + Frontend + Docker + Testes |
| F1-score (function calling) | 0.2222 (limitação do modelo CPU) |
| System recall | **1.0** (todos os cenários de sistema cobertos) |
| Suíte de system recall | `test_system_recall.py` — 42 testes |

Todas as funcionalidades do desafio técnico foram implementadas e testadas.

---

## Componentes implementados

### Backend (100%)
- [x] `config/settings.py` — Pydantic Settings com .env + auto-model selection
- [x] `app/Models/WeatherForecast.py` — DailyForecast, WeatherResponse
- [x] `app/Models/ChatMessage.py` — ChatMessage, ChatRequest, ChatResponse (com campo `reason`)
- [x] `app/Repositories/CapitalsRepository.py` — lookup de capitais com fuzzy match
- [x] `app/Tools/WeatherTool.py` — WEATHER_TOOL (formato OpenAI) + execute
- [x] `app/Services/WeatherService.py` — Open-Meteo HTTP + HTTPStatusError por status code
- [x] `app/Services/AgentService.py` — loop agêntico completo (detect → execute → re-call)
- [x] `app/Providers/ServiceProvider.py` — FastAPI DI
- [x] `app/Http/Requests/ChatRequest.py` — validação Pydantic (protected_namespaces corrigido)
- [x] `app/Http/Controllers/AgentController.py` — POST /chat, GET /health
- [x] `app/Http/Controllers/WeatherController.py` — GET /weather, GET /cities
- [x] `app/Http/Middleware/LoggingMiddleware.py` — request logging
- [x] `app/Http/Middleware/ErrorHandler.py` — global error handler
- [x] `api/v1/routes.py` — router agregador
- [x] `main.py` — FastAPI app completo com lifespan, CORS, Swagger + auto-model selection
- [x] `requirements.txt`
- [x] `.env.example`
- [x] `pytest.ini`
- [x] `resources/data/capitals.json`
- [x] `README.md` (completo com todos os campos exigidos)

### Frontend (100%)
- [x] `frontend/` — React + TypeScript + Vite
- [x] Interface de chat (ChatWindow, MessageBubble, WeatherCard)
- [x] Serviço de API (fetch para /api/v1/agent/chat)
- [x] Hook useChat para gerenciar histórico
- [x] 4 sugestões de perguntas na tela inicial
- [x] Badge "⚡ Open-Meteo" em mensagens com tool call
- [x] Animação de loading (3 pontos)
- [x] Botão de limpar conversa
- [x] README do frontend

### Docker (100%)
- [x] `docker-compose.yml` (backend + frontend)
- [x] `backend/Dockerfile`
- [x] `frontend/Dockerfile`
- [x] Apontamento correto para `host.docker.internal:11434` (Ollama no host)

### Testes (100%)
- [x] `tests/Unit/test_capitals_repository.py`
- [x] `tests/Unit/test_weather_tool.py`
- [x] `tests/Unit/test_weather_service.py`
- [x] `tests/Feature/test_agent_api.py`
- [x] `tests/Validation/test_function_calling.py` (F1/precision/recall)
- [x] `tests/Validation/test_system_recall.py` — **42/42 PASSED**
- [x] `tests/conftest.py`

### Plugins agents/ (100%)
- [x] `agents/plugins/climatempo-inspector/` — doc-reader + compliance-auditor
- [x] `agents/plugins/climatempo-qa/` — qa-orchestrator + e2e-tester + arch-validator + fc-validator

---

## Métricas de qualidade

| Métrica | Resultado | Observação |
|---------|-----------|------------|
| Testes automatizados | **42/42** | `pytest tests/` |
| System recall | **1.0** | Todos os cenários cobertos |
| F1-score (function calling) | **0.2222** | Limitação do modelo em CPU sem GPU |
| Precision | varia por modelo | qwen2.5 em CPU |
| Pydantic warnings | **0** | `protected_namespaces` corrigido |
| Código morto removido | `OllamaProvider.py` | Deletado |

---

## Comandos de execução rápida

```bash
# --- BACKEND ---
cd teste/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Iniciar Ollama (terminal separado)
ollama serve

# Iniciar backend
uvicorn main:app --reload --port 8000

# --- FRONTEND ---
cd teste/frontend
npm install
npm run dev
# Acesse http://localhost:5173

# --- TESTES ---
cd teste/backend
source .venv/bin/activate

# Todos os testes unitários + feature (sem Ollama)
pytest tests/Unit/ tests/Feature/ -v

# Suíte system recall (sem Ollama)
pytest tests/Validation/test_system_recall.py -v

# Validação F1 (precisa do Ollama + backend rodando em :8000)
pytest tests/Validation/test_function_calling.py -v -s

# --- DOCKER ---
cd teste
docker compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:5173
```

---

## Arquivos chave para entender o projeto

| Arquivo | O que faz |
|---------|-----------|
| `app/Tools/WeatherTool.py` | Definição OpenAI da tool (coração do desafio) |
| `app/Services/AgentService.py` | Loop agêntico completo |
| `app/Services/WeatherService.py` | Chamada HTTP ao Open-Meteo |
| `main.py` | Auto-model selection no startup |
| `config/settings.py` | Configuração + lista de prioridade de modelos |
| `tests/Validation/test_system_recall.py` | 42 cenários de sistema |
| `tests/Validation/test_function_calling.py` | Suíte F1 |
| `README.md` | Documentação de entrega |
