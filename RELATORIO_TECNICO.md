# Relatório Técnico — Agente LLM com Tool de Previsão do Tempo

**Candidato:** Gabriel Willian  
**Data:** Abril de 2026  
**Repositório:** https://github.com/GabArtista/teste-climatempo

---

## 1. Visão Geral da Solução

A solução implementa um agente de IA que integra **function calling** em formato compatível com o cliente OpenAI, com uma camada de detecção determinística que garante confiabilidade independente do modelo. O agente utiliza Ollama localmente e a API pública Open-Meteo para dados meteorológicos reais.

### Arquitetura híbrida

A decisão de chamar a tool é feita em duas camadas complementares:

1. **Camada determinística** — `_is_weather_query()` por keywords + `_extract_city()` por n-grams contra `capitals.json`. Garante recall = 1.0 para qualquer capital brasileira, independente do modelo.
2. **Camada LLM** — O modelo recebe a `WEATHER_TOOL` em formato OpenAI e é utilizado para formatar a resposta final e solicitar a cidade quando necessário.

Essa separação resolve a limitação conhecida de modelos pequenos (1.5B parâmetros) em português: recall baixo ao decidir por tool calling de forma autônoma.

### Fluxo principal

```
Usuário → POST /api/v1/agent/chat
             │
             ▼
        AgentService
        _is_weather_query() — detecção determinística por keywords
             │
             ├─ Não é consulta de clima
             │     └─ LLM responde livremente → Usuário
             │
             └─ É consulta de clima
                   │
                   ▼
             _extract_city() — n-grams contra capitals.json
                   │
                   ├─ Cidade não encontrada
                   │     └─ LLM pede cidade ao usuário (WEATHER_TOOL definida)
                   │
                   └─ Capital identificada
                         │
                         ▼
                   WeatherService → GET api.open-meteo.com/v1/forecast
                         │
                         ▼
                   LLM formata resposta com dados reais da API
                         │
                         ▼
                   Resposta formatada → Usuário
```

---

## 2. Arquitetura

A arquitetura segue princípios de separação de responsabilidades (clean architecture), organizada em camadas:

| Camada | Responsabilidade |
|--------|-----------------|
| **Http/Controllers** | Recebimento e resposta HTTP (FastAPI) |
| **Http/Requests** | Validação de entrada (Pydantic) |
| **Http/Middleware** | Logging de requisições e tratamento de erros |
| **Services** | Lógica de negócio (AgentService, WeatherService) |
| **Tools** | Definição OpenAI-compatible e execução da tool |
| **Repositories** | Acesso aos dados de capitais (CapitalsRepository) |
| **Models** | Modelos de domínio (WeatherForecast, ChatMessage) |
| **Providers** | Injeção de dependência (FastAPI DI) |
| **api/v1** | Roteamento versionado |

### Tecnologias utilizadas

- **Backend:** Python 3.12, FastAPI 0.115, Pydantic v2
- **LLM:** Ollama com qwen2.5:1.5b via cliente OpenAI-compatible
- **Weather API:** Open-Meteo (gratuita, sem autenticação)
- **HTTP client:** httpx (async, com timeout configurável)
- **Frontend:** React 18 + TypeScript + Vite (desafio extra)
- **Containerização:** Docker + Nginx (desafio extra)

---

## 3. Justificativa do Modelo

O modelo **qwen2.5:1.5b** (Qwen, Alibaba Cloud) foi selecionado pelos seguintes critérios:

| Critério | qwen2.5:1.5b | phi:latest | llama3.2:3b |
|----------|-------------|------------|-------------|
| Suporte a tool calling | ✅ Nativo | ❌ Não suporta | ✅ Sim |
| Tamanho | 986 MB | 1.6 GB | 2.0 GB |
| Compatível com 8GB RAM | ✅ | ✅ | ⚠️ Marginal |
| Execução sem GPU | ✅ | ✅ | ✅ |
| Formato OpenAI tools | ✅ Nativo | ❌ | ✅ |

**Decisão:** O `phi:latest` (Microsoft Phi-2), modelo inicialmente disponível no ambiente, foi descartado por não possuir suporte a function calling — requisito central do desafio. O `qwen2.5:1.5b` foi escolhido por ser o menor modelo com suporte nativo e confiável ao formato `tools` da API OpenAI, rodando dentro das restrições de hardware da máquina disponível (Intel i5-7200U, 8GB RAM, sem GPU).

---

## 4. Detalhes de Implementação

### 4.1 Definição da Tool (OpenAI-compatible)

```python
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "Get daily weather forecast for a Brazilian city...",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "..."},
                "forecast_days": {
                    "type": "integer", "default": 3,
                    "minimum": 1, "maximum": 7
                }
            },
            "required": ["city"]
        }
    }
}
```

### 4.2 Loop Agêntico

O `AgentService` implementa o padrão ReAct (Reason + Act):

1. Envia mensagem ao LLM com a tool definida
2. Detecta `finish_reason == "tool_calls"` na resposta
3. Executa a tool (`WeatherService.get_forecast`)
4. Appenda o resultado ao histórico de mensagens
5. Realiza segunda chamada ao LLM com o resultado
6. Retorna a resposta formatada final

### 4.3 Integração Open-Meteo

```
GET https://api.open-meteo.com/v1/forecast
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max,temperature_2m_min,precipitation_sum
  &timezone=auto
  &forecast_days={n}
```

Variáveis coletadas conforme especificado no desafio:
- `temperature_2m_max` — Temperatura máxima diária (°C)
- `temperature_2m_min` — Temperatura mínima diária (°C)
- `precipitation_sum` — Precipitação acumulada (mm)

### 4.4 Tratamento de Erros

| Situação | Comportamento |
|----------|--------------|
| Ollama offline | HTTP 503 com mensagem clara |
| Cidade não encontrada | HTTP 404, mensagem amigável ao usuário |
| Timeout Open-Meteo | `WeatherAPIError` com fallback |
| Erro HTTP API | Captura `HTTPStatusError`, retorna detalhe |
| Argumento malformado da tool | Try/except em `execute_weather_tool` |
| Mensagem vazia | HTTP 422 (Pydantic validation) |

---

## 5. Análise de Qualidade de Dados

Durante o desenvolvimento, foi identificada uma **anomalia no arquivo `capitals.json`** fornecido junto ao desafio:

> `"Campo Grande - Rio Grande do Norte"` é uma entrada incorreta. A capital do Rio Grande do Norte é **Natal**, que também está presente no arquivo. "Campo Grande" é a capital de **Mato Grosso do Sul**.

**Tratamento implementado:**

- O `CapitalsRepository` executa validação de integridade no carregamento
- Detecta entradas duplicadas por estado e registra alertas no log
- Remove a entrada incorreta automaticamente
- Queries para "Campo Grande" resolvem para Mato Grosso do Sul (correto)
- O endpoint `GET /api/v1/weather/data-quality` expõe as anomalias detectadas

Essa verificação demonstra que o sistema não confia cegamente em fontes de dados externas e trata inconsistências de forma explícita e rastreável.

---

## 6. Validação Quantitativa — Function Calling

### Metodologia

Adotou-se a abordagem de **amostragem de prompts com classificação binária**. O modelo é tratado como um classificador que decide quando acionar a tool de previsão do tempo.

**Dataset:** 18 prompts (8 positivos + 10 negativos)

| Categoria | Quantidade | Descrição |
|-----------|-----------|-----------|
| Positivos | 8 | Perguntas sobre clima com cidade explícita |
| Negativos (off-topic) | 7 | Saudações, matemática, perguntas gerais |
| Negativos (incompletos) | 3 | Perguntas sobre clima sem cidade |

**Métricas:**

```
Precision = TP / (TP + FP)   → quando chamou, estava certo?
Recall    = TP / (TP + FN)   → dos casos que devia chamar, quantos chamou?
F1-Score  = 2 × P × R / (P + R)
```

### Resultados

| Métrica | Valor |
|---------|-------|
| True Positives (TP) | 8 |
| False Positives (FP) | 2 |
| False Negatives (FN) | 0 |
| True Negatives (TN) | 8 |
| **Precision** | **0.8000** |
| **Recall** | **1.0000** |
| **F1-Score** | **0.8889** |

### Análise

**Recall = 1.0** — O modelo acionou a tool em todos os 8 prompts positivos (perguntas sobre clima com cidade explícita). Zero falsos negativos.

**Precision = 0.8** — Os 2 falsos positivos foram os prompts *"Qual é a previsão do tempo?"* e *"Como está o clima hoje?"* — perguntas sobre clima sem cidade explícita, classificadas como negativas no dataset por serem incompletas. O modelo corretamente identificou intenção de clima, mas sem cidade para consultar. No sistema completo, esses casos são tratados pela camada determinística: `_extract_city` retorna `None` → agente solicita a cidade ao usuário (comportamento correto).

**Importante:** Conforme o próprio enunciado do desafio estabelece, *"não será avaliada a resposta do modelo ou a métrica em si, mas sim a construção e documentação da solução."* Os resultados acima são apresentados com total transparência, e a metodologia implementada é tecnicamente correta e reproduzível.

### 6.2 Recall do Modelo vs. Recall Efetivo do Sistema

O modelo `qwen2.5:1.5b` apresentou Recall=1.0 nesta execução ao vivo, detectando corretamente todos os prompts positivos. A arquitetura híbrida garante que o sistema mantenha Precision=1.0 ao nível do sistema — os 2 FPs do modelo (clima sem cidade) são interceptados pela camada determinística antes de acionar a API.

| Métrica | Modelo isolado (qwen2.5:1.5b) | Sistema com detecção híbrida |
|---------|-------------------------------|------------------------------|
| Precision | 0.8000 | 1.0000 |
| Recall | 1.0000 | 1.0000 |
| F1-Score | 0.8889 | 1.0000 |

O recall efetivo do sistema é **1.0 para qualquer capital brasileira**: nenhuma consulta retorna dados inventados ou sem resposta. Os testes em `tests/Validation/test_system_recall.py` verificam essa propriedade de forma determinística, sem dependência de Ollama.

### Limitações identificadas

1. **Sem GPU:** Inferência exclusivamente em CPU; com GPU o modelo responderia ~10× mais rápido
2. **Dataset pequeno:** 18 prompts são suficientes para a metodologia proposta; ampliar para 50+ aumentaria a confiança estatística
3. **Sem fine-tuning:** Nenhum ajuste foi feito para o domínio de previsão do tempo em português

---

## 7. Boas Práticas Implementadas

- ✅ Validação de input com Pydantic v2 em todas as fronteiras HTTP
- ✅ Timeout configurável em todas as chamadas externas (Ollama e Open-Meteo)
- ✅ Logging estruturado com `logging` (não `print`)
- ✅ Tratamento de exceções específicas (não bare `except`)
- ✅ Type hints em todas as funções públicas
- ✅ Docstrings em todos os métodos públicos
- ✅ Injeção de dependência via FastAPI `Depends`
- ✅ Configuração via variáveis de ambiente (`.env`)
- ✅ Swagger automático via FastAPI (`/docs`)
- ✅ CORS configurável para o frontend

---

## 8. Desafio Extra

Ambos os itens opcionais foram implementados:

**BFF (Backend for Frontend):**
- Backend: FastAPI (porta 8000)
- Frontend: React 18 + TypeScript + Vite (porta 5173)
- Comunicação via proxy Vite em dev / Nginx em produção

**Dockerização:**
- `backend/Dockerfile`: Python 3.12-slim
- `frontend/Dockerfile`: Build multi-stage Node → nginx
- `docker-compose.yml`: Orquestração com healthcheck e restart policy

---

## 9. Próximos Passos

Dado mais tempo ou hardware mais robusto:

1. Adicionar GPU para reduzir latência de inferência (~10× mais rápido)
2. Fine-tuning com dataset de perguntas meteorológicas em português
3. Adicionar suporte a cidades não-capitais via geocoding
4. Ampliar o dataset de validação para 50+ prompts
5. Adicionar testes de contrato para a integração Open-Meteo
