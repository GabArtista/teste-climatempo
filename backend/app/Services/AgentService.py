"""
Serviço de Agente Conversacional — Detecção de intenção em cascata + chamada de ferramentas.

[Bloco 1 — Humanizado]
Este módulo é o coração do sistema. Ele decide se a mensagem do usuário é uma
consulta de previsão do tempo ou uma conversa geral, e roteia cada caso para o
tratamento adequado. O objetivo é nunca alucinar dados meteorológicos: ou buscamos
a previsão real na API, ou conversamos normalmente sem inventar nada.

[Bloco 2 — Técnico]
Decisão arquitetural: o modelo qwen2.5:1.5b (1,5B parâmetros, CPU-only) tem baixo
recall para tool calling em português. Para garantir a recuperação confiável dos dados,
adotamos uma abordagem híbrida:

  1. Detectar intenção via pipeline de 5 estágios em cascata (determinístico primeiro,
     LLM apenas como último recurso)
  2. Extrair cidade via correspondência fuzzy contra o banco de capitais
  3. Intenção de tempo + cidade conhecida → chama Open-Meteo diretamente; LLM só formata
  4. Intenção de tempo + cidade desconhecida → informa cidades suportadas
  5. Intenção de tempo + sem cidade → pede ao LLM que solicite a cidade ao usuário
  6. Sem intenção de tempo → conversa LLM normal sem ferramentas

Pipeline de intenção em cascata (do mais barato ao mais caro):
  Estágio 1 — Keywords fortes (~0ms): "previsão", "temperatura", etc.
  Estágio 2 — Frases fixas (~0ms): "vai chover", "como está o tempo", etc.
  Estágio 3 — Padrões de exclusão (~1ms): comentários pessoais como "tô com frio aqui"
  Estágio 4 — Scoring multi-sinal (~1-5ms): sinais ponderados com thresholds
  Estágio 5 — Classificador LLM binário (~30-60s): último recurso para ambiguidade real

Isso elimina dados meteorológicos alucinados enquanto preserva conversação natural
para tópicos não-clima. Estima-se que o LLM seja necessário em < 10% dos casos
que passam pelo filtro de vocabulário do Estágio 1.
"""
import json
import logging
import re
from typing import Literal

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from app.Models.ChatMessage import ChatMessage, ChatResponse, MessageRole
from app.Models.WeatherForecast import WeatherResponse
from app.Repositories.CapitalsRepository import CapitalsRepository
from app.Services.WeatherService import CityNotFoundError, WeatherService
from app.Tools.WeatherTool import WEATHER_TOOL
from config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 — Strong keywords: almost always a weather query
# ---------------------------------------------------------------------------
# Palavras fortes: seu único sentido prático em PT/EN é meteorológico.
# "previsão" e "temperatura" raramente aparecem fora de contexto climático.
# Incluir as variantes sem acento evita depender de normalização prévia do input.
# "weather" e "forecast" cobrem mensagens mistas PT/EN de usuários técnicos.
_STRONG_KEYWORDS = {
    'previsão', 'previsao', 'temperatura', 'precipitação', 'precipitacao',
    'forecast', 'weather', 'temperature',
    'clima',   # "clima" in PT almost exclusively means meteorological climate
}

# ---------------------------------------------------------------------------
# Stage 2 — Fixed phrases that unambiguously express weather intent
# ---------------------------------------------------------------------------
# Expressões compostas: cada uma, isoladamente, só faz sentido em contexto
# meteorológico — não há ambiguidade semântica como há em palavras soltas.
# Incluir variantes com e sem acento (ex: "está"/"esta") evita normalização
# Unicode prévia, que tem custo não-nulo no hot path de cada requisição.
_WEATHER_PHRASES = [
    'como está o tempo', 'como esta o tempo',
    'como vai estar', 'vai chover', 'está chovendo', 'esta chovendo',
    'como está o clima', 'como esta o clima',
    'previsão do tempo', 'previsao do tempo',
    'vai fazer frio', 'vai fazer calor',
    'como vai o tempo', 'como vai o clima',
]

# ---------------------------------------------------------------------------
# Stage 3 — Exclusion patterns: personal/non-weather uses of weather words
# ---------------------------------------------------------------------------
# Todos usam re.I: variações de caixa são comuns em linguagem informal
# ("Tô com Frio", "ESTOU COM CALOR"). O custo de re.I é insignificante aqui
# pois os padrões são compilados uma única vez no import do módulo.
_EXCLUSION_PATTERNS = [
    # "tô/estou com frio/calor/febre" — sensação física em 1ª pessoa;
    # .{0,20} e .{0,10} deixam espaço para qualificadores ("estou muito com frio")
    re.compile(r'\b(t[oô]|estou|fiquei|sinto|me sinto|fico)\b.{0,20}\bcom\b.{0,10}\b(frio|calor|febre|quente|fria)\b', re.I),
    # aparelhos domésticos indicam ambiente interno — sem relação com clima externo
    re.compile(r'\b(ar[\s-]condicionado|ventilador|aquecedor|cobertor|blusa|casaco)\b', re.I),
    # "ela estava quente/fria" — temperatura corporal de terceiro, não atmosférica
    re.compile(r'\b(ela|ele|você|vc|a\s+\w+|o\s+\w+)\b.{0,20}\b(estava|está|ficou|ficava)\b.{0,15}\b(quente|fria|frio|gelad)\b', re.I),
    # locativos de espaço pessoal/indoor: âncora a conversa num ambiente fechado,
    # não numa localização geográfica externa onde o clima seria relevante
    re.compile(r'\b(aqui\s+(no|em|na|dentro)\b|em\s+casa\b|no\s+(escritório|escritorio|trabalho|apartamento|quarto)\b)', re.I),
    # co-ocorrência de doença + temperatura: sintoma clínico, não condição climática
    re.compile(r'\b(febre|resfriado|gripe|doente).{0,20}\b(frio|calor|quente)\b', re.I),
    # ordem inversa do mesmo padrão de doença ("frio e febre" vs "febre e frio")
    re.compile(r'\b(frio|calor|quente).{0,20}\b(febre|resfriado|gripe|doente)\b', re.I),
]

# ---------------------------------------------------------------------------
# Stage 4 — Multi-signal scoring thresholds
# ---------------------------------------------------------------------------
# 0.55: exige pelo menos dois sinais positivos moderados para confirmar clima
#   (ex: "?" +0.40 + temporal +0.20 = 0.70, ou interrogativo +0.35 + cidade +0.30 = 0.75)
# 0.20: zona morta — score abaixo indica ausência de sinais positivos claros,
#   mesmo sem sinais negativos fortes; cai em not_weather sem gastar o LLM
# A janela 0.20–0.55 é a zona ambígua real encaminhada ao Estágio 5 (LLM)
_THRESHOLD_WEATHER = 0.55       # score above → weather
_THRESHOLD_NOT_WEATHER = 0.20   # score below → not weather
# Baseline positivo: falso positivo (chamar API desnecessariamente) é preferível
# a falso negativo (ignorar consulta legítima de previsão); 0.10 dá uma vantagem
# inicial para clima sem comprometer a especificidade do pipeline
_SCORE_BASELINE = 0.10          # slight positive bias (FP better than FN)

# Interrogativos que no início da frase indicam forte intenção de pergunta.
# "que" foi excluído intencionalmente: funciona tanto como interrogativo
# ("Que tempo vai fazer?") quanto como exclamação ("Que sol bonito!") —
# a ambiguidade tornaria o sinal não confiável como indicador de consulta.
# "vai"/"irá"/"será" incluídos porque perguntas meteorológicas no futuro
# frequentemente começam com verbo modal ("Vai chover em SP?").
_INTERROGATIVES = {
    'como', 'qual', 'quais', 'quando', 'onde', 'quanto', 'quanta',
    'quantos', 'quantas', 'vai', 'irá', 'será',
}

# Marcadores temporais: sua presença sugere orientação futura, típica de consultas
# de previsão ("como vai estar no sábado?", "amanhã vai chover?").
# 'próxim'/'proxim' são prefixos que cobrem "próxima", "próximo", "proxima"
# sem precisar listar todas as formas flexionadas.
_TEMPORAL_MARKERS = {
    'amanhã', 'amanha', 'semana', 'hoje', 'sábado', 'sabado',
    'domingo', 'segunda', 'terça', 'terca', 'quarta', 'quinta',
    'sexta', 'próxim', 'proxim', 'fim de semana',
}

# Sujeitos pessoais como primeira palavra da frase: indicam que o falante está
# descrevendo a si mesmo, não fazendo uma consulta externa. Ex: "Eu tô com frio"
# vs "Como vai estar o tempo?". Aplicado apenas à posição inicial para minimizar
# falsos positivos ("Meu Deus, vai chover muito!" ainda detectaria clima pelo "?").
_PERSONAL_SUBJECTS = {
    'eu', 'tô', 'to', 'estou', 'me', 'meu', 'minha',
    'a gente', 'nós', 'nos',
}

# Preposições geográficas seguidas de letra maiúscula.
# NÃO usa re.I propositalmente: a maiúscula é exatamente o sinal que queremos
# detectar — nomes de cidades e estados são sempre capitalizados no texto.
# Se usássemos re.I, "em casa", "na foto", "no trabalho" gerariam falsos positivos
# pois "casa", "foto", "trabalho" também têm letras (ainda que minúsculas, que re.I
# tornaria equivalentes a maiúsculas). A ausência de re.I é a restrição intencional.
_GEO_PREPOSITIONS = re.compile(r'\b(em|no|na|para|pra)\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ]')

# Verbos meteorológicos no futuro: a combinação modal+verbo é quase exclusiva
# de consultas de previsão. "vai chover", "irá ventar", "será que vai fazer frio"
# raramente aparecem fora de contexto climático, diferente de "vai fazer calor"
# isoladamente que pode ser comentário pessoal.
_FUTURE_WEATHER_VERBS = re.compile(
    r'\b(vai|irá|sera|será)\s+(chover|nevar|ventar|fazer|ter)\b', re.I
)

# Vocabulário fraco: palavras relacionadas a clima mas com usos alternativos
# frequentes em PT ("frio" = sensação, "sol" = aprovação informal, "quente" = sexy).
# Sua presença sozinha não basta para classificar como consulta meteorológica;
# são usadas no Stage 1 apenas para decidir se há vocabulário climático suficiente
# para continuar o pipeline — se nenhuma palavra fraca ou forte aparecer, o Stage 1
# decide "not_weather" imediatamente sem precisar avaliar frases ou scoring.
_WEAK_KEYWORDS = {
    'clima', 'chuva', 'chover', 'choverá', 'chovera',
    'calor', 'frio', 'umidade', 'vento', 'nublado', 'sol',
    'ensolarado', 'nuvem', 'nuvens', 'quente', 'fria', 'quentes',
    'rain', 'graus',
}

_SYSTEM_PROMPT = (
    "Você é um assistente de previsão do tempo para capitais brasileiras. "
    "Responda sempre em português, de forma clara e amigável.\n\n"
    "IMPORTANTE:\n"
    "- Quando receber dados de previsão do tempo, formate-os de forma legível "
    "com datas, temperaturas em °C e precipitação em mm.\n"
    "- Quando o usuário perguntar sobre uma cidade que não é capital estadual, "
    "explique gentilmente que só temos dados das 26 capitais brasileiras.\n"
    "- Quando precisar pedir a cidade, seja direto e amigável."
)

_FORMAT_PROMPT = (
    "O usuário perguntou: {question}\n\n"
    "Aqui estão os dados reais de previsão do tempo obtidos da API:\n"
    "{weather_data}\n\n"
    "Por favor, formate esses dados de forma clara e amigável em português, "
    "incluindo as datas, temperaturas máxima e mínima em °C, e precipitação em mm. "
    "Use emojis para deixar mais visual. NÃO invente dados — use apenas os fornecidos acima."
)


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not running or not reachable."""


class AgentService:
    """
    Agente híbrido de previsão do tempo: detecção de intenção em cascata + LLM para formatação.

    [Bloco 1 — Humanizado]
    Esta classe é o cérebro do assistente. Ela recebe qualquer mensagem do usuário,
    decide se é uma pergunta sobre o tempo ou não, busca os dados reais quando necessário
    e usa o LLM apenas para escrever a resposta em linguagem natural. Nunca inventa dados.

    [Bloco 2 — Técnico]
    Orquestra um pipeline de 5 estágios para classificação de intenção, seguido de
    extração de entidades (cidade, número de dias) e chamada direta à WeatherService.
    O cliente OpenAI (AsyncOpenAI) aponta para o Ollama local via base_url configurável.
    O LLM é invocado em dois pontos distintos: como classificador binário no Estágio 5
    e como formatador de linguagem natural após a obtenção dos dados reais.
    """

    def __init__(
        self,
        settings: Settings,
        weather_service: WeatherService,
        repo: CapitalsRepository,
    ) -> None:
        """
        Inicializa o agente com suas dependências injetadas.

        [Bloco 1 — Humanizado]
        Recebe as três dependências necessárias para funcionar: as configurações da
        aplicação, o serviço de clima e o repositório de capitais. Cria o cliente
        de comunicação com o modelo de linguagem.

        [Bloco 2 — Técnico]
        Instancia um AsyncOpenAI apontando para o Ollama local (base_url e api_key
        extraídos de Settings). Mantém referências ao WeatherService e ao
        CapitalsRepository para uso nos métodos de extração e consulta.

        Args:
            settings: Configurações da aplicação (URLs, modelo, timeouts).
            weather_service: Serviço para buscar previsão do tempo via Open-Meteo.
            repo: Repositório das 27 capitais brasileiras com coordenadas.
        """
        self._settings = settings
        self._weather_service = weather_service
        self._repo = repo
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(self, message: str, history: list[ChatMessage]) -> ChatResponse:
        """
        Processa uma mensagem do usuário pelo pipeline completo do agente.

        [Bloco 1 — Humanizado]
        É o ponto de entrada principal do agente. Recebe o que o usuário digitou,
        descobre se é uma pergunta de clima, busca os dados se necessário e devolve
        uma resposta completa. É aqui que o fluxo agêntico acontece de fato.

        [Bloco 2 — Técnico]
        Fluxo: classify_intent → se weather: extract_city + extract_days → get_forecast
        → llm_format_weather; se not_weather: llm_general_chat. Exceções de conexão
        e timeout do Ollama são capturadas e relançadas como OllamaUnavailableError
        para tratamento uniforme na camada de roteamento HTTP. O histórico é passado
        para manter contexto conversacional (busca de cidade em turnos anteriores
        e coerência de respostas gerais).

        Args:
            message: Mensagem atual do usuário.
            history: Turnos anteriores da conversa (ChatMessage com role e content).

        Returns:
            ChatResponse com o texto da resposta, flag de tool_called, cidade consultada
            e motivo do fluxo tomado (success, no_city, non_weather, non_capital).

        Raises:
            OllamaUnavailableError: Se o Ollama não estiver rodando ou não responder.
        """
        logger.info("Chat message: %r", message[:60])

        try:
            intent = await self._classify_intent(message)
            if intent == "weather":
                return await self._handle_weather_query(message, history)
            else:
                return await self._handle_general_chat(message, history)

        except APIConnectionError as exc:
            logger.error("Ollama connection failed: %s", exc)
            raise OllamaUnavailableError(
                "Ollama não está disponível. Certifique-se que está rodando: ollama serve"
            ) from exc
        except APITimeoutError as exc:
            logger.error("Ollama timeout: %s", exc)
            raise OllamaUnavailableError("Ollama demorou demais para responder.") from exc

    async def check_health(self) -> dict:
        """
        Verifica se o backend Ollama está disponível e com o modelo carregado.

        [Bloco 1 — Humanizado]
        Serve como diagnóstico rápido: confirma que o Ollama está rodando e que
        o modelo configurado está presente. Útil para endpoints de healthcheck
        e para dar feedback claro ao operador quando algo não está certo.

        [Bloco 2 — Técnico]
        Chama client.models.list() via API OpenAI-compatible do Ollama e verifica
        se o settings.ollama_model aparece em algum m.id retornado. Retorna um dict
        com chaves 'status' ("ok" | "model_not_found" | "unavailable"), 'model' e
        'available'. Em caso de exceção (conexão recusada, timeout), captura e
        devolve status "unavailable" com a mensagem de erro — nunca propaga.

        Returns:
            Dict com 'status', 'model', 'available' e, em caso de erro, 'error'.
        """
        try:
            models = await self._client.models.list()
            available = any(self._settings.ollama_model in m.id for m in models.data)
            return {"status": "ok" if available else "model_not_found",
                    "model": self._settings.ollama_model, "available": available}
        except Exception as exc:
            return {"status": "unavailable", "model": self._settings.ollama_model,
                    "available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Cascading intent classification
    # ------------------------------------------------------------------

    async def _classify_intent(self, message: str) -> Literal["weather", "not_weather"]:
        """
        Pipeline de classificação de intenção em 5 estágios, do mais barato ao mais caro.

        [Bloco 1 — Humanizado]
        Decide se a mensagem é uma consulta de previsão do tempo ou não. Começa pelos
        métodos mais rápidos (simples verificação de vocabulário) e só avança para
        estágios mais lentos quando os anteriores não conseguem decidir com confiança.
        O LLM — o estágio mais lento — só é chamado em casos realmente ambíguos.

        [Bloco 2 — Técnico]
        Cada estágio retorna uma decisão final ou passa para o próximo:

        Estágio 1 — Keywords fortes (~0ms): interseção do conjunto de palavras da
          mensagem com _STRONG_KEYWORDS. Se há vocab de clima, avança; se não há
          nenhum vocab de clima (nem fraco), já decide "not_weather" aqui.

        Estágio 2 — Frases fixas (~0ms): verifica se o lower da mensagem contém
          alguma das _WEATHER_PHRASES (expressões inequívocas de consulta climática).

        Estágio 3 — Padrões de exclusão (~1ms): aplica _is_non_weather_context(),
          que testa regex de comentários pessoais, eletrodomésticos, sensações físicas.

        Estágio 4 — Scoring multi-sinal (~1-5ms): _score_weather_intent() acumula
          pesos positivos e negativos; threshold >= 0.55 → weather, <= 0.20 → not_weather,
          zona intermediária → avança para Estágio 5.

        Estágio 5 — LLM binário (~30-60s): prompt mínimo pedindo "sim" ou "não".
          Parseamento: "sim" em qualquer posição da resposta → weather.

        Distribuição esperada: ~60% encerram no Estágio 1, ~25% no Estágio 2,
        ~10% nos Estágios 3-4, ~5% chegam ao Estágio 5.

        Args:
            message: Mensagem do usuário (pode conter pontuação e maiúsculas).

        Returns:
            Literal "weather" ou "not_weather".
        """
        # Normalize whitespace (newlines, tabs, multiple spaces → single space)
        message = " ".join(message.split())
        lower = message.lower()
        words = set(re.sub(r'[?!.,;]', ' ', lower).split())

        # Stage 1 — Strong keywords: near-certain weather intent
        if words & _STRONG_KEYWORDS:
            logger.debug("Intent: weather (stage 1 — strong keyword)")
            return "weather"  # Stage 1: keyword forte — evita Stage 2 (frases), 3 (regex), 4 (scoring), 5 (LLM ~50ms)

        # Check for any weather vocabulary at all — if none, skip to not_weather
        has_weather_vocab = bool(words & _WEAK_KEYWORDS) or any(
            phrase in lower for phrase in _WEATHER_PHRASES
        )
        if not has_weather_vocab:
            logger.debug("Intent: not_weather (stage 1 — no weather vocab)")
            return "not_weather"  # Stage 1: sem vocabulário climático algum — evita todos os estágios restantes

        # Stage 2 — Fixed phrases: unambiguous weather expressions
        if any(phrase in lower for phrase in _WEATHER_PHRASES):
            logger.debug("Intent: weather (stage 2 — fixed phrase)")
            return "weather"  # Stage 2: frase fixa inequívoca — evita Stage 3 (regex), 4 (scoring), 5 (LLM ~50ms)

        # Stage 3 — Exclusion patterns: personal/non-weather uses
        if self._is_non_weather_context(message):
            logger.debug("Intent: not_weather (stage 3 — exclusion pattern)")
            return "not_weather"  # Stage 3: padrão de exclusão — evita Stage 4 (scoring) e 5 (LLM ~50ms)

        # Stage 4 — Multi-signal scoring
        score = self._score_weather_intent(message)
        logger.debug("Intent: stage 4 score=%.2f", score)
        if score >= _THRESHOLD_WEATHER:
            return "weather"
        if score <= _THRESHOLD_NOT_WEATHER:
            return "not_weather"

        # Stage 5 — LLM binary classifier (last resort)
        logger.info("Intent: ambiguous (score=%.2f), escalating to LLM", score)
        return await self._llm_classify_intent(message)

    def _is_non_weather_context(self, message: str) -> bool:
        """
        Estágio 3: verifica se a mensagem corresponde a padrões conhecidos de não-clima.

        [Bloco 1 — Humanizado]
        Algumas mensagens usam palavras de clima (frio, calor, chuva) sem ser uma
        consulta meteorológica — por exemplo, "tô com frio aqui em casa" ou "ela
        estava com febre e frio". Este método filtra esses casos antes de gastar
        tempo com scoring ou LLM.

        [Bloco 2 — Técnico]
        Aplica sequencialmente os 5 padrões regex de _EXCLUSION_PATTERNS:

        Padrão 1 — Sensação física pessoal: "tô/estou/fiquei com frio/calor/febre"
          (sujeito em 1ª pessoa + "com" + sensação corporal).

        Padrão 2 — Aparelhos domésticos: "ar-condicionado", "ventilador", "aquecedor",
          "cobertor", "blusa", "casaco" — contexto indoor, não consulta climática.

        Padrão 3 — Descrição física de terceiro: "ela/ele estava quente/fria/frio"
          (sujeito 3ª pessoa + verbo estado + adjetivo de temperatura).

        Padrão 4 — Locativos pessoais: "aqui no/em", "em casa", "no escritório/trabalho/
          apartamento/quarto" — ancora a conversa num espaço interno, não externo.

        Padrão 5 — Contexto de doença: "febre" ou "resfriado"/"gripe" perto de
          "frio"/"calor" — indica sintoma, não temperatura ambiente.

        Esses padrões existem porque palavras como "frio" e "calor" são altamente
        polissêmicas em português e disparariam falsos positivos no Estágio 4.

        Args:
            message: Mensagem original do usuário (com capitalização preservada).

        Returns:
            True se algum padrão de exclusão for detectado; False caso contrário.
        """
        for pattern in _EXCLUSION_PATTERNS:
            if pattern.search(message):
                return True
        return False

    def _score_weather_intent(self, message: str) -> float:
        """
        Estágio 4: pontua a mensagem com base em múltiplos sinais linguísticos.

        [Bloco 1 — Humanizado]
        Funciona como uma balança: sinais que indicam "é uma pergunta de clima" somam
        pontos positivos, sinais que indicam "é um comentário pessoal" subtraem. O
        resultado é um número entre 0 e 1 que representa a confiança de que a mensagem
        é uma consulta meteorológica.

        [Bloco 2 — Técnico]
        Filosofia de scoring: parte de um baseline positivo (0.10) porque falso positivo
        (tratar comentário pessoal como consulta de clima) é menos grave do que falso
        negativo (ignorar uma consulta legítima de previsão).

        Sinais positivos e seus pesos:
          +0.40 — Ponto de interrogação ("?")
          +0.35 — Interrogativo no início da frase (como, qual, vai, irá, será…)
          +0.15 — Preposição geográfica seguida de maiúscula (em São Paulo, no Recife)
          +0.20 — Marcador temporal (amanhã, semana, sábado, próxim…)
          +0.30 — Verbo meteorológico no futuro (vai chover, irá ventar…)
          +0.30 — Cidade capital identificada pelo CapitalsRepository

        Sinais negativos e seus pesos:
          -0.45 — Sujeito pessoal como primeira palavra (eu, tô, estou, me, meu…)
          -0.50 — Padrão de sensação corporal (estou com frio/calor/febre)
          -0.30 — Locativo pessoal (aqui, em casa, no trabalho…)
          -0.35 — Descrição física de terceiro (ela/ele estava quente/fria)

        Thresholds de decisão:
          score >= 0.55 → weather
          score <= 0.20 → not_weather
          0.20 < score < 0.55 → inconclusivo → avança para Estágio 5 (LLM)

        O score final é clampado em [0.0, 1.0].

        Args:
            message: Mensagem do usuário (capitalização original preservada para
                     detectar nomes próprios via _GEO_PREPOSITIONS).

        Returns:
            Float em [0.0, 1.0] representando confiança de intenção climática.
        """
        score = _SCORE_BASELINE
        lower = message.lower()
        words_raw = lower.split()
        words = set(re.sub(r'[?!.,;]', ' ', lower).split())

        # --- Positive signals ---

        # "?" é o sinal isolado mais forte: quase toda consulta de previsão é uma
        # pergunta. Sozinho já eleva 0.10 + 0.40 = 0.50, próximo do threshold 0.55.
        # Combinado com um interrogativo (+0.35), basta "Como vai o tempo?" para
        # ultrapassar 0.55 sem precisar de cidade ou temporal.
        if '?' in message:
            score += 0.40

        # Interrogativo no início: reforça que é uma pergunta real, não afirmação.
        # Peso 0.35 menor que "?" porque frases afirmativas com interrogativo
        # inicial são raras mas existem ("Qual sorte...!"). Combinado com "?" → +0.75.
        if words_raw and words_raw[0].strip('?!.,') in _INTERROGATIVES:
            score += 0.35

        # Preposição geográfica + maiúscula: "em Salvador", "no Recife", "para Manaus".
        # Peso baixo (0.15) porque "em" + maiúscula pode aparecer no início de frase
        # ("Em termos gerais..."). Funciona como sinal de desempate, não decisivo.
        # Usa a mensagem original (não lowercased) pois a maiúscula é o sinal detectado.
        if _GEO_PREPOSITIONS.search(message):
            score += 0.15

        # Marcador temporal: orienta a frase para o futuro, típico de consulta de previsão.
        # Peso 0.20 porque temporais também aparecem em relatos pessoais ("amanhã tenho
        # prova e tô com febre"). Serve como reforço, não como sinal único decisivo.
        if any(marker in lower for marker in _TEMPORAL_MARKERS):
            score += 0.20

        # Verbo meteorológico no futuro: combinação modal+verbo quase exclusiva de
        # previsão. Peso 0.30 alto porque "vai chover" / "irá ventar" raramente
        # aparece fora de contexto climático, diferente de palavras isoladas.
        if _FUTURE_WEATHER_VERBS.search(lower):
            score += 0.30

        # Capital identificada: confirma que há uma localização geográfica real na
        # mensagem. Peso 0.30 porque cidade sozinha sem outros sinais pode ser apenas
        # menção ("Gosto muito de Curitiba"). Combinado com "?" já passa o threshold.
        if self._extract_city(message) is not None:
            score += 0.30

        # --- Negative signals ---

        # Sujeito pessoal no início: forte indicador de comentário pessoal.
        # Peso alto (-0.45) porque "Eu tô com frio" e "Estou com calor" são os
        # falsos positivos mais comuns — usuários que reclamam do clima pessoal.
        # Aplicado apenas à primeira palavra para não penalizar "Eu quero saber
        # a previsão de amanhã" (que é consulta legítima com sujeito no início).
        first_word = words_raw[0].strip('?!.,') if words_raw else ''
        if first_word in _PERSONAL_SUBJECTS:
            score -= 0.45

        # Padrão de sensação corporal: o mais forte sinal negativo (-0.50) porque
        # "estou com frio" / "tô com febre" é virtualmente impossível ser uma consulta
        # meteorológica. Derruba qualquer combinação positiva exceto se houver cidade
        # explícita + "?" juntos (0.10 + 0.40 + 0.30 - 0.50 = 0.30, ainda abaixo de 0.55).
        if re.search(r'\b(estou|t[oô]|fiquei)\b.{0,15}\bcom\b.{0,10}\b(frio|calor|febre)\b', lower):
            score -= 0.50

        # Locativo pessoal: âncora a conversa num espaço indoor ("aqui no quarto",
        # "em casa"). Peso moderado (-0.30) porque pode coexistir com consulta legítima
        # ("aqui em SP, vai chover amanhã?") — nesse caso outros sinais positivos compensam.
        if re.search(r'\b(aqui|em\s+casa|no\s+(trabalho|escritório|escritorio|quarto|apartamento))\b', lower):
            score -= 0.30

        # Descrição física de terceiro: "ela estava quente", "ele ficou com frio".
        # Peso -0.35 porque indica temperatura corporal de uma pessoa, não condição
        # atmosférica. Menos agressivo que sensação em 1ª pessoa porque é mais raro.
        if re.search(r'\b(ela|ele)\b.{0,20}\b(estava|está|ficou)\b.{0,15}\b(quente|fria|frio)\b', lower):
            score -= 0.35

        return max(0.0, min(1.0, score))

    async def _llm_classify_intent(self, message: str) -> Literal["weather", "not_weather"]:
        """
        Estágio 5 (último recurso): pede ao LLM uma decisão binária de classificação.

        [Bloco 1 — Humanizado]
        Quando os estágios determinísticos (1 a 4) não chegam a uma conclusão
        confiante, delegamos a decisão ao modelo de linguagem. É o estágio mais
        lento e custoso, usado apenas para casos genuinamente ambíguos — como
        mensagens curtas sem contexto claro.

        [Bloco 2 — Técnico]
        Construção do prompt: mensagem mínima em português pedindo apenas "sim" ou
        "não" como resposta, sem nenhum contexto adicional. O objetivo é minimizar
        latência e tokens consumidos, já que o modelo qwen2.5:1.5b é lento em CPU.

        Chamada: client.chat.completions.create() com uma única mensagem de role
        "user". Sem system prompt para evitar tokens extras.

        Parsing da resposta: busca pela string "sim" (case-insensitive, após .lower()
        e .strip()) em qualquer posição da resposta — o modelo pode retornar "Sim.",
        "sim, parece", etc. Qualquer resposta sem "sim" é tratada como "not_weather".

        Custo estimado: ~30-60s em CPU i5-7200U com qwen2.5:1.5b. Por isso este
        estágio é reservado para < 5% dos casos.

        Args:
            message: Mensagem do usuário que não pôde ser classificada pelos estágios 1-4.

        Returns:
            Literal "weather" se o LLM responder "sim"; "not_weather" caso contrário.
        """
        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=[{
                "role": "user",
                "content": (
                    "Responda apenas 'sim' ou 'não', sem mais nada.\n"
                    f"A mensagem a seguir é uma consulta de previsão do tempo? "
                    f'"{message}"'
                ),
            }],
        )
        answer = (resp.choices[0].message.content or "").lower().strip()
        result: Literal["weather", "not_weather"] = "weather" if "sim" in answer else "not_weather"
        logger.info("LLM intent classification: %r → %s", answer[:20], result)
        return result

    # ------------------------------------------------------------------
    # City and days extraction
    # ------------------------------------------------------------------

    def _extract_city(self, message: str) -> dict | None:
        """
        Tenta encontrar uma capital brasileira mencionada na mensagem via n-gramas.

        [Bloco 1 — Humanizado]
        Varre a mensagem em busca de qualquer capital brasileira conhecida. Começa
        pelos grupos de 3 palavras (para pegar nomes compostos como "Rio de Janeiro"),
        depois 2 palavras ("São Paulo", "Porto Alegre") e por último palavras isoladas
        ("Recife", "Natal"). Retorna a primeira correspondência encontrada.

        [Bloco 2 — Técnico]
        Estratégia n-gram decrescente (3 → 2 → 1): garante que nomes compostos
        sejam identificados antes de suas partes. Exemplo: "Rio de Janeiro" deve
        ser encontrado antes que "Rio" pudesse gerar correspondência incorreta.

        Pré-processamento: remove pontuação (?!.,;) substituindo por espaço antes
        de dividir em palavras — evita que "Brasília?" não seja reconhecida.

        A normalização de acentos e caixa é delegada ao CapitalsRepository.find_city(),
        que aplica NFD → ASCII antes da comparação. Por isso, "SAO PAULO", "são paulo"
        e "Sao Paulo" são todos reconhecidos corretamente.

        Não há desambiguação interna: a primeira correspondência encontrada é retornada.
        Em mensagens com duas cidades, a que aparecer primeiro na ordem n-gram/posição
        vence.

        Args:
            message: Texto da mensagem do usuário (capitalização original).

        Returns:
            Dict com 'name', 'latitude', 'longitude' da capital, ou None se não encontrada.
        """
        # Substitui pontuação por espaço antes de dividir: evita que "Brasília?"
        # ou "Recife," não sejam reconhecidas porque o token incluiria o sinal.
        text = re.sub(r'[?!.,;]', ' ', message)
        words = text.split()

        # Descende de n=3 para n=1: garante que nomes compostos sejam testados
        # antes de suas partes — "Rio de Janeiro" (n=3) tem precedência sobre "Rio"
        # (n=1), evitando matches parciais incorretos em cidades de nome composto.
        # list_cities() é chamado internamente pelo find_city() a cada chamada, mas
        # o CapitalsRepository já mantém o índice em memória (O(1) por lookup),
        # tornando a iteração sobre até ~27*3 combinações negligenciável.
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i: i + n])
                result = self._repo.find_city(phrase)
                if result:
                    return result
        # n=1 foi o último tier — nenhum n-grama casou com capital conhecida;
        # retorna None para sinalizar ao chamador que deve pedir a cidade ao usuário.
        return None

    def _extract_days(self, message: str) -> int:
        """
        Extrai o número de dias de previsão solicitado na mensagem.

        [Bloco 1 — Humanizado]
        Detecta expressões temporais na pergunta do usuário para saber quantos dias
        de previsão buscar. Se o usuário não especificou, assume 3 dias — um padrão
        que cobre o curto prazo sem sobrecarregar a resposta.

        [Bloco 2 — Técnico]
        Padrões reconhecidos (em ordem de prioridade):

        1. Número explícito + "dia": regex r'(\\d+)\\s*dia' extrai qualquer número de
           1 a 7 (clamped). Ex.: "próximos 5 dias" → 5.

        2. "semana": qualquer ocorrência de "semana" → 7 dias.

        3. "amanhã" / "amanha": → 1 dia (ortografia com e sem acento).

        4. Default: 3 dias — cobre curto prazo sem exigir que o usuário especifique.

        O clamp max(1, min(7, n)) alinha com o limite da Open-Meteo API (máximo 7 dias
        no plano gratuito). Valores fora do range são silenciosamente ajustados.

        Args:
            message: Mensagem do usuário em texto livre.

        Returns:
            Inteiro entre 1 e 7 representando o número de dias de previsão.
        """
        match = re.search(r'(\d+)\s*dia', message.lower())
        if match:
            return max(1, min(7, int(match.group(1))))
        if 'semana' in message.lower():
            return 7
        if 'amanhã' in message.lower() or 'amanha' in message.lower():
            return 1
        return 3

    # ------------------------------------------------------------------
    # Weather flow
    # ------------------------------------------------------------------

    async def _handle_weather_query(
        self,
        message: str,
        history: list[ChatMessage],
    ) -> ChatResponse:
        """Handle a detected weather query."""
        city = self._extract_city(message)

        if city is None:
            # Check history for a city mentioned previously
            for msg in reversed(history):
                city = self._extract_city(msg.content)
                if city:
                    break

        if city is None:
            # Weather intent but no city found — ask for it
            logger.info("Weather intent detected, no city found — asking user")
            response = await self._llm_ask_for_city(message, history)
            return ChatResponse(response=response, tool_called=False, city_queried=None, reason="no_city")

        days = self._extract_days(message)
        logger.info("Weather query: city=%s days=%d", city['name'], days)

        try:
            forecast = await self._weather_service.get_forecast(
                city=city['name'], forecast_days=days
            )
        except CityNotFoundError as exc:
            return ChatResponse(
                response=str(exc),
                tool_called=False,
                city_queried=city['name'],
                reason="non_capital",
            )

        # Use LLM only to format the response — data is real, from the API
        formatted = await self._llm_format_weather(message, forecast)
        return ChatResponse(
            response=formatted,
            tool_called=True,
            city_queried=city['name'],
            reason="success",
        )

    async def _llm_format_weather(
        self, question: str, forecast: WeatherResponse
    ) -> str:
        """Ask the LLM to format real weather data into natural language."""
        weather_text = forecast.to_text()

        prompt = _FORMAT_PROMPT.format(
            question=question,
            weather_data=weather_text,
        )

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        result = resp.choices[0].message.content or weather_text
        logger.info("LLM formatted weather response for %s", forecast.city)
        return result

    async def _llm_ask_for_city(
        self, message: str, history: list[ChatMessage]
    ) -> str:
        """Ask the LLM to request the city from the user."""
        cities_sample = ", ".join(self._repo.list_cities()[:6])
        messages = [
            {"role": "system", "content": (
                f"{_SYSTEM_PROMPT}\n\n"
                f"Cidades disponíveis (26 capitais estaduais): {cities_sample} e outras.\n"
                "Peça ao usuário que informe uma capital estadual brasileira."
            )},
        ]
        for msg in history[-4:]:
            messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": message})

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=messages,
        )
        return resp.choices[0].message.content or "Por favor, informe o nome de uma capital estadual brasileira."

    # ------------------------------------------------------------------
    # General chat (non-weather)
    # ------------------------------------------------------------------

    async def _handle_general_chat(
        self, message: str, history: list[ChatMessage]
    ) -> ChatResponse:
        """Handle non-weather messages with plain LLM conversation."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for msg in history[-6:]:
            messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": message})

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=messages,
        )
        content = resp.choices[0].message.content or "Como posso ajudar?"
        return ChatResponse(response=content, tool_called=False, city_queried=None, reason="non_weather")
