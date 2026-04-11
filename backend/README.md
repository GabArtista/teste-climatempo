# Weather LLM Agent

Agente de IA com function calling para previsão do tempo, integrando LLM local (Ollama) com a API Open-Meteo.

## Arquitetura

```
backend/
├── main.py                        # FastAPI entry point + Swagger
├── config/settings.py             # Pydantic Settings (env vars)
├── app/
│   ├── Http/
│   │   ├── Controllers/           # Handlers HTTP (AgentController, WeatherController)
│   │   ├── Middleware/            # Logging, ErrorHandler
│   │   └── Requests/              # Validação de entrada (Pydantic)
│   ├── Models/                    # Domain models (WeatherForecast, ChatMessage)
│   ├── Providers/                 # Injeção de dependência
│   ├── Repositories/              # Acesso a dados (CapitalsRepository)
│   ├── Services/                  # Lógica de negócio (AgentService, WeatherService)
│   └── Tools/                     # Tool calling OpenAI-compatible (WeatherTool)
├── api/v1/routes.py               # Rotas versionadas
├── resources/data/capitals.json   # 27 capitais brasileiras com lat/lon
└── tests/
    ├── Unit/                      # Testes unitários (mocked)
    ├── Feature/                   # Testes de integração HTTP
    └── Validation/                # Suíte F1/precision/recall
```

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.ai) instalado e rodando
- Modelo `qwen2.5:1.5b` baixado
- Conexão com a internet (API Open-Meteo)

## Instalação

```bash
# 1. Clone e entre no diretório
cd backend

# 2. Crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env se necessário (valores padrão já funcionam)
```

## Configuração

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | URL do Ollama |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Modelo LLM |
| `OPEN_METEO_BASE_URL` | `https://api.open-meteo.com/v1` | API de previsão |
| `REQUEST_TIMEOUT` | `30` | Timeout HTTP em segundos |
| `LOG_LEVEL` | `INFO` | Nível de log |

## Executando

```bash
# 1. Inicie o Ollama (em outro terminal)
ollama serve

# 2. Certifique-se que o modelo está disponível
ollama pull qwen2.5:1.5b

# 3. Inicie o backend
uvicorn main:app --reload --port 8000
```

Acesse:
- **API interativa (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/api/v1/agent/health

## Exemplos de uso

### Chat com o agente

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Como está o tempo em São Paulo nos próximos 3 dias?",
    "history": []
  }'
```

**Resposta:**
```json
{
  "response": "🌤️ Previsão do tempo para São Paulo:\n\n📅 15/01/2024: máx 28.5°C, mín 20.1°C, sem chuva\n📅 16/01/2024: máx 30.1°C, mín 21.3°C, 5.2mm chuva\n📅 17/01/2024: máx 27.3°C, mín 19.8°C, 12.4mm chuva",
  "tool_called": true,
  "city_queried": "São Paulo"
}
```

### Consulta direta (sem LLM)

```bash
curl "http://localhost:8000/api/v1/weather/?city=Curitiba&days=3"
```

### Conversa sem ferramenta

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá, tudo bem?", "history": []}'
```

**Resposta:**
```json
{
  "response": "Olá! Estou bem, obrigado. Como posso ajudar com informações sobre o tempo?",
  "tool_called": false,
  "city_queried": null
}
```

## Executando os testes

```bash
# Testes unitários (não precisam do Ollama)
pytest tests/Unit/ -v

# Testes de feature (não precisam do Ollama — usam mocks)
pytest tests/Feature/ -v

# Validação F1 (precisa do Ollama rodando — lento ~5min)
pytest tests/Validation/ -v -s

# Todos os testes
pytest tests/ -v
```

## Validação de Precisão (F1-Score)

A suíte de validação testa se o modelo decide corretamente quando acionar a ferramenta de clima. Utiliza uma amostra de 15 prompts (8 positivos + 7 negativos) e calcula precision, recall e F1-Score.

```bash
pytest tests/Validation/test_function_calling.py -v -s
# Resultado salvo em: tests/Validation/results.json
```

## Escolha do modelo

**`qwen2.5:1.5b`** foi escolhido pelos seguintes critérios:

1. **Suporte a tool calling**: único modelo de pequeno porte com suporte nativo e confiável a function calling no formato OpenAI
2. **Tamanho adequado**: 1.0GB — compatível com 8GB RAM e execução sem GPU
3. **API compatível**: suporta o formato `tools` da API OpenAI via Ollama
4. **Alternativas descartadas**: `phi:latest` (sem tool calling), `llama3.2:3b` (2GB — muito grande para a máquina disponível)
