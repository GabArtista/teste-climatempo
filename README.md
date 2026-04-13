# Teste Técnico Climatempo — AI Engineer

Agente conversacional de previsão do tempo com function calling sobre LLM local (Ollama). O sistema interpreta consultas em linguagem natural, identifica capitais brasileiras via detecção híbrida (keywords + n-gram determinístico) e busca dados reais da Open-Meteo API. O LLM é usado exclusivamente para formatação da resposta final, garantindo respostas consistentes e sem alucinações sobre dados meteorológicos.

**Repositório:** https://github.com/GabArtistas/teste-climatempo

---

## Tecnologias

| Camada | Stack |
|--------|-------|
| Backend | Python 3.12, FastAPI, Ollama (LLM local), Open-Meteo API |
| Frontend | React 18, TypeScript, Vite (desafio extra) |
| Infra | Docker, docker-compose (desafio extra) |
| Testes | pytest — 42 testes passando (unit + feature + system recall) |

---

## Quick Start

### Opção A — Docker (recomendado)

```bash
# Requer Ollama rodando localmente com o modelo configurado
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173

### Opção B — Manual (backend + frontend)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (novo terminal)
cd frontend
npm install
npm run dev
```

### Opção C — Só backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## Pré-requisitos

- **Python 3.11+**
- **Ollama** instalado e rodando com o modelo configurado em `backend/.env`
- **Node 18+** (apenas para o frontend)
- **Docker + Docker Compose** (apenas para Opção A)

Instalar Ollama: https://ollama.com — após instalar, baixe o modelo:

```bash
ollama pull llama3.2  # ou o modelo definido em backend/.env
```

---

## Estrutura do Repositório

| Diretório / Arquivo | Descrição |
|---------------------|-----------|
| `backend/` | API FastAPI, agente, lógica de detecção e integração Open-Meteo |
| `frontend/` | Interface React/TypeScript (desafio extra) |
| `docker-compose.yml` | Orquestração com healthcheck (desafio extra) |
| `backend/README.md` | Instruções detalhadas do backend, variáveis de ambiente, arquitetura |
| `GUIA_EXECUCAO.md` | Passo-a-passo completo com testes manuais |
| `RELATORIO_TECNICO.md` | Relatório técnico: arquitetura, decisões, validação F1 |

---

## Exemplo Rápido

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como está o tempo em São Paulo?"}'
```

Resposta esperada:

```json
{
  "response": "Em São Paulo, a temperatura atual é de 24°C com céu parcialmente nublado. A sensação térmica é de 26°C e a umidade relativa está em 68%.",
  "city_detected": "São Paulo",
  "weather_data": { ... }
}
```

---

## Rodando os Testes

```bash
cd backend
pytest tests/ -v
```

42 testes cobrindo: detecção de cidades (unit), integração com Open-Meteo (feature) e recall sobre as 27 capitais brasileiras (system).

---

## Documentação

- [`backend/README.md`](backend/README.md) — configuração detalhada, variáveis de ambiente, endpoints
- [`GUIA_EXECUCAO.md`](GUIA_EXECUCAO.md) — passo-a-passo completo com exemplos e testes manuais
- [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md) — arquitetura, decisões técnicas e métricas de validação
