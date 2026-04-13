# Relatório Técnico — Agente LLM com Tool de Previsão do Tempo

**Candidato:** Gabriel Willian | **Data:** Abril de 2026 | **Repositório:** https://github.com/GabArtista/teste-climatempo

---

## 1. Solução

O sistema implementa um **agente conversacional** que integra um LLM local (Ollama — `qwen2.5:1.5b`) com a API pública Open-Meteo via **function calling** no formato compatível com o cliente OpenAI.

### Arquitetura híbrida

A decisão de acionar a tool opera em duas camadas:

1. **Camada determinística** — `_is_weather_query()` detecta intenção por keywords; `_extract_city()` localiza a capital por n-grams contra `capitals.json`. Garante recall = 1.0 para qualquer das 27 capitais brasileiras, independente do modelo.
2. **Camada LLM** — O modelo recebe a `WEATHER_TOOL` (formato OpenAI) e é usado exclusivamente para formatar a resposta final com os dados reais da API.

```
Usuário → POST /api/v1/agent/chat
              │
              ▼
         _is_weather_query()  ── Não ──→  LLM responde livremente
              │ Sim
              ▼
         _extract_city()  ── Não encontrou ──→  LLM pede a cidade
              │ Capital encontrada
              ▼
         WeatherService → GET api.open-meteo.com/v1/forecast
              │
              ▼
         LLM formata resposta com dados reais → Usuário
```

Essa separação resolve a limitação conhecida de modelos pequenos (1.5B) em português: recall baixo ao decidir por tool calling de forma autônoma.

---

## 2. Stack e Justificativa do Modelo

| Componente | Tecnologia |
|------------|-----------|
| Backend | Python 3.12, FastAPI 0.115, Pydantic v2 |
| LLM | Ollama — qwen2.5:1.5b (986 MB) |
| Weather API | Open-Meteo (gratuita, sem autenticação) |
| HTTP client | httpx (async, timeout configurável) |
| Frontend (extra) | React 18 + TypeScript + Vite |
| Infra (extra) | Docker + Nginx multi-stage |

**Modelo escolhido:** `qwen2.5:1.5b` — único modelo testado com suporte nativo a **function calling** no formato OpenAI, rodando dentro das restrições de hardware (Intel i5-7200U, 8 GB RAM, sem GPU). O modelo `phi:latest` (disponível por padrão) foi descartado por não suportar o formato `tools`.

**Variáveis coletadas conforme especificado:**

| Variável Open-Meteo | Descrição | Unidade |
|--------------------|-----------|---------|
| `temperature_2m_max` | Temperatura máxima diária | °C |
| `temperature_2m_min` | Temperatura mínima diária | °C |
| `precipitation_sum` | Precipitação acumulada diária | mm |

---

## 3. Validação Quantitativa — Function Calling

**Metodologia:** amostragem de prompts com classificação binária. O modelo é tratado como classificador que decide quando acionar a tool de previsão do tempo.

**Dataset:** 18 prompts — 8 positivos (perguntas com capital explícita) + 10 negativos (7 off-topic + 3 clima sem cidade).

| Métrica | Modelo (qwen2.5:1.5b) | Sistema híbrido |
|---------|----------------------|----------------|
| True Positives | 8 | 8 |
| False Positives | 1 | 0 |
| False Negatives | 0 | 0 |
| **Precision** | **0.8889** | **1.0000** |
| **Recall** | **1.0000** | **1.0000** |
| **F1-Score** | **0.9412** | **1.0000** |

**Recall = 1.0** — A tool foi acionada em 100% dos prompts positivos (zero falsos negativos).

**O único FP do modelo** foi o prompt *"Qual é a previsão do tempo?"* — intenção de clima corretamente identificada, mas sem cidade. No sistema completo, esse caso é interceptado pela camada determinística: `_extract_city()` retorna `None` → agente solicita a cidade ao usuário → `tool_called = False`. Precision efetiva do sistema = 1.0.

> Conforme o enunciado: *"não será avaliada a resposta do modelo ou a métrica em si, mas sim a construção e documentação da solução."*

**Anomalia nos dados de entrada detectada e tratada:** o arquivo `capitals.json` continha uma entrada incorreta (`"Campo Grande - Rio Grande do Norte"`). O `CapitalsRepository` detecta, loga e remove a entrada automaticamente no startup. Queries para "Campo Grande" resolvem para Mato Grosso do Sul (correto).
