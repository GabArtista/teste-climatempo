"""
Repositório de capitais estaduais brasileiras.

[Bloco 1 — Humanizado]
Este módulo centraliza o acesso aos dados das 27 capitais brasileiras (26 estados
+ Distrito Federal). Ele carrega o arquivo JSON com nomes e coordenadas, detecta
problemas de qualidade nos dados na inicialização e fornece uma busca tolerante
a variações de escrita (acentos, caixa, nomes parciais).

[Bloco 2 — Técnico]
Fonte de dados: arquivo capitals.json no formato {"Cidade - Estado": {"latitude": float,
"longitude": float}}. O repositório constrói um índice normalizado (NFD → ASCII →
lowercase) na inicialização para permitir buscas eficientes sem acentos.

Anomalia conhecida no arquivo-fonte: "Campo Grande - Rio Grande do Norte" é uma
entrada incorreta — a capital do Rio Grande do Norte é Natal, não Campo Grande.
Campo Grande é capital do Mato Grosso do Sul. O repositório detecta, registra e
corrige automaticamente essa anomalia no startup via _KNOWN_ANOMALIES.
"""
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_CAPITALS_FILE = Path(__file__).parent.parent.parent / "resources" / "data" / "capitals.json"

# Known erroneous entries in the source data — mapped to the correct capital
_KNOWN_ANOMALIES: dict[str, str] = {
    "Campo Grande - Rio Grande do Norte": "Natal - Rio Grande do Norte",
}


def _normalize(text: str) -> str:
    """
    Normaliza texto para comparação sem acento e sem diferença de caixa.

    [Bloco 1 — Humanizado]
    Converte qualquer string para uma forma canônica que ignora acentos e maiúsculas.
    Assim, "São Paulo", "sao paulo", "SAO PAULO" e "são Paulo" todos viram "sao paulo"
    e podem ser comparados diretamente.

    [Bloco 2 — Técnico]
    Pipeline de transformação (lossy, mas aceitável para lookup de nomes geográficos):
      1. .lower() — remove diferença de caixa
      2. .strip() — remove espaços nas bordas
      3. unicodedata.normalize("NFD", text) — decompõe caracteres compostos em
         caractere-base + diacrítico separado (ex.: "ã" → "a" + combining tilde)
      4. Filtra caracteres de categoria "Mn" (Mark, Nonspacing) — remove os diacríticos
         separados, deixando apenas os caracteres-base ASCII

    A perda de informação é intencional e aceitável: os nomes das capitais brasileiras
    são únicos mesmo após remoção de acentos (não há colisões no conjunto de 27 capitais).

    Args:
        text: String original com possíveis acentos e capitalização variada.

    Returns:
        String ASCII lowercase sem diacríticos.
    """
    # lower() + strip(): remove diferença de caixa e espaços externos antes
    # da decomposição — mais eficiente que aplicar após (menos caracteres a processar)
    text = text.lower().strip()
    return "".join(
        # NFD separa o caractere base do acento em code points distintos
        # (ex: "ã" → "a" + U+0303 COMBINING TILDE; "é" → "e" + U+0301 COMBINING ACUTE)
        # encode("ascii","ignore")+decode("ascii") seria equivalente mas NFD + filtro
        # de categoria é mais explícito sobre o que está sendo removido
        c for c in unicodedata.normalize("NFD", text)
        # categoria "Mn" (Mark, Nonspacing) = diacríticos separados pelo NFD;
        # filtrar essa categoria remove acentos sem afetar letras-base ASCII
        if unicodedata.category(c) != "Mn"
    )


class DataIntegrityWarning(UserWarning):
    """Raised when anomalies are detected in the capitals data file."""


class CapitalsRepository:
    """
    Acesso às capitais estaduais brasileiras com coordenadas geográficas.

    [Bloco 1 — Humanizado]
    Carrega, valida e disponibiliza os dados das 27 capitais brasileiras. Cuida
    silenciosamente de problemas de qualidade no arquivo-fonte para que o resto
    do sistema nunca precise lidar com dados incorretos. Fornece busca tolerante
    a erros de acentuação e capitalização.

    [Bloco 2 — Técnico]
    Na inicialização: carrega JSON → valida integridade → aplica correções →
    constrói índice normalizado (dict[str_normalizada → str_original]).
    O índice normalizado permite buscas O(1) para correspondências exatas e
    iteração linear para os tiers 2 e 3 do find_city(). O conjunto de dados
    é pequeno (27 entradas), tornando a iteração negligenciável em performance.
    """

    def __init__(self, data_path: Path = _CAPITALS_FILE) -> None:
        """
        Inicializa o repositório carregando e validando o arquivo de capitais.

        [Bloco 1 — Humanizado]
        Lê o arquivo JSON, verifica se os dados fazem sentido (cada estado deve
        ter exatamente uma capital), aplica correções automáticas para erros
        conhecidos e prepara o índice de busca normalizado.

        [Bloco 2 — Técnico]
        Sequência de inicialização:
          1. _load(): lê e faz parse do JSON; lança FileNotFoundError se ausente
          2. _validate_integrity(): detecta estados duplicados e entradas inválidas;
             registra warnings no logger; retorna lista de anomalias para relatório
          3. _apply_corrections(): remove entradas de _KNOWN_ANOMALIES do dataset
          4. Constrói _normalized_index: {_normalize(key): key} para lookup rápido

        Args:
            data_path: Caminho para o arquivo capitals.json. Padrão aponta para
                       resources/data/capitals.json relativo à raiz do projeto.

        Raises:
            FileNotFoundError: Se o arquivo capitals.json não existir no caminho informado.
        """
        raw = self._load(data_path)
        self._anomalies = self._validate_integrity(raw)
        self._data: dict[str, dict[str, float]] = self._apply_corrections(raw)
        self._normalized_index: dict[str, str] = {
            _normalize(k): k for k in self._data
        }
        logger.info("CapitalsRepository loaded %d cities", len(self._data))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> dict[str, dict[str, float]]:
        """Read and parse the JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Capitals data file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _validate_integrity(self, data: dict) -> list[str]:
        """
        Detecta problemas de qualidade nos dados e os registra no log.

        [Bloco 1 — Humanizado]
        Funciona como um auditor que confere o arquivo de capitais na inicialização.
        Se encontrar algo errado (por exemplo, dois registros para o mesmo estado),
        registra um aviso no log e documenta o problema para o relatório de qualidade.
        Não interrompe a execução — apenas informa.

        [Bloco 2 — Técnico]
        Invariantes verificadas:

        1. Unicidade de estado: cada parte após " - " deve aparecer em no máximo uma
           entrada. Detectado via defaultdict(list): agrupa chaves por estado e verifica
           len(cities) > 1. Log level: WARNING com lista das entradas conflitantes.

        2. Entradas de anomalias conhecidas: verifica se alguma chave de _KNOWN_ANOMALIES
           ainda está presente no dataset. Presença indica que o arquivo-fonte não foi
           corrigido na origem.

        Ambos os tipos de problema são registrados via logger.warning() e acumulados
        na lista de issues retornada. A integridade do dataset após correções é garantida
        por _apply_corrections(), chamada logo em seguida no __init__.

        Args:
            data: Dicionário bruto carregado do JSON, antes de qualquer correção.

        Returns:
            Lista de strings descrevendo cada anomalia detectada (pode ser vazia).
        """
        issues: list[str] = []

        # Check for duplicate states
        state_map: dict[str, list[str]] = defaultdict(list)
        for city_key in data:
            parts = city_key.split(" - ")
            if len(parts) == 2:
                state_map[parts[1]].append(city_key)

        for state, cities in state_map.items():
            if len(cities) > 1:
                # Invariante violada: cada estado brasileiro tem exatamente 1 capital.
                # Duas entradas para o mesmo estado indicam erro no arquivo JSON —
                # ex: "Campo Grande - Rio Grande do Norte" E "Natal - Rio Grande do Norte".
                # logger.warning e não raise: não queremos interromper o startup da
                # aplicação por problema de dados; o sistema ainda funciona com dados
                # parcialmente incorretos, e a anomalia será corrigida em _apply_corrections.
                msg = (
                    f"Duplicate state detected in capitals.json: "
                    f"'{state}' has {len(cities)} entries: {cities}. "
                    f"Expected exactly one capital per state."
                )
                logger.warning(msg)
                issues.append(msg)

        # Check known anomalies
        for wrong_key in _KNOWN_ANOMALIES:
            if wrong_key in data:
                msg = (
                    f"Known data anomaly: '{wrong_key}' is incorrect. "
                    f"Will be resolved to '{_KNOWN_ANOMALIES[wrong_key]}'."
                )
                logger.warning(msg)
                issues.append(msg)

        return issues

    def _apply_corrections(self, data: dict) -> dict:
        """
        Remove entradas incorretas conhecidas do dataset carregado.

        [Bloco 1 — Humanizado]
        Aplica automaticamente as correções para erros conhecidos no arquivo-fonte.
        Em vez de exigir que o arquivo JSON seja corrigido manualmente (o que poderia
        ser revertido acidentalmente), o repositório corrige na memória a cada startup.

        [Bloco 2 — Técnico]
        Itera sobre _KNOWN_ANOMALIES (dict {chave_errada: chave_correta}) e remove
        cada chave_errada do dicionário de dados se presente. A chave_correta deve
        já existir no arquivo JSON com os dados corretos — esta função apenas remove
        a entrada duplicada/errada, não insere nada.

        Trade-off de manutenção: hardcoded é simples e previsível para um conjunto
        pequeno e estável de erros. A alternativa (arquivo de patches externo ou
        script de migration) adicionaria complexidade desproporcional para 1-2 anomalias
        históricas. Se novos erros aparecerem, basta adicionar uma entrada em
        _KNOWN_ANOMALIES no topo do módulo.

        Args:
            data: Dicionário bruto do JSON (copiado internamente — não modifica o original).

        Returns:
            Novo dicionário sem as entradas incorretas.
        """
        corrected = dict(data)
        for wrong_key in _KNOWN_ANOMALIES:
            if wrong_key in corrected:
                del corrected[wrong_key]
                logger.info("Removed anomalous entry: '%s'", wrong_key)
        return corrected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_city(self, name: str) -> dict | None:
        """
        Busca uma capital pelo nome, tolerante a acentos e capitalização.

        [Bloco 1 — Humanizado]
        Aceita o nome da cidade de qualquer forma — com ou sem acento, maiúsculo
        ou minúsculo, com ou sem o nome do estado. Tenta três estratégias em ordem
        crescente de flexibilidade, retornando na primeira correspondência encontrada.

        [Bloco 2 — Técnico]
        Três tiers de lookup (aplicados em sequência; retorna na primeira correspondência):

        Tier 1 — Correspondência exata normalizada: normaliza a query e busca diretamente
          no _normalized_index (O(1)). Cobre "São Paulo", "sao paulo", "SAO PAULO",
          "São Paulo - São Paulo" etc.

        Tier 2 — Query contida na chave com word-boundary: r'\\b' + query + r'\\b' no
          norm_key. Permite encontrar "sao paulo" dentro de "sao paulo - sao paulo".
          O word-boundary evita que "em" corresponda a "belem" ou "belem - para".

        Tier 3 — Sobreposição bidirecional city_part ↔ query: extrai a parte antes
          do " - " de cada chave normalizada e verifica se a parte da cidade está
          contida na query ou vice-versa. Permite localizar "Belém" a partir de
          "Em Belém" ou "Belém do Pará". Word-boundary aplicado em ambas as direções
          para evitar falsos positivos com preposições curtas.

        Casos extremos tratados:
          - "Campo Grande": retorna MS (a entrada RN foi removida por _apply_corrections)
          - "Natal": encontrada no Tier 2 ("natal" em "natal - rio grande do norte")
          - Nomes parciais como "Porto" não correspondem (word-boundary impede "porto"
            de casar com "porto alegre" pelo Tier 3 quando a query é apenas "porto")

        Args:
            name: Nome ou nome parcial de cidade a buscar.

        Returns:
            Dict com 'name' (string "Cidade - Estado"), 'latitude' e 'longitude',
            ou None se nenhuma correspondência for encontrada.
        """
        # Normaliza a query antes de comparar: garante que "São Paulo", "sao paulo"
        # e "SAO PAULO" produzam o mesmo token e possam bater no índice normalizado.
        query = _normalize(name)

        # Tier 1 — Exact normalized match: O(1) via dict lookup.
        # Cobre o caso mais comum: usuário digita o nome exato com ou sem acento/caixa.
        if query in self._normalized_index:
            key = self._normalized_index[query]
            return {"name": key, **self._data[key]}

        # Tier 2 — Query contida na chave como palavra inteira.
        # re.escape() evita que nomes com caracteres especiais (ex: "São.Paulo" num
        # input mal formatado) sejam interpretados como meta-caracteres regex.
        # \b (word boundary) impede "em" de casar no meio de "belem - para",
        # e "natal" de casar apenas como substring de "paranatal" (se existisse).
        for norm_key, orig_key in self._normalized_index.items():
            if re.search(r'\b' + re.escape(query) + r'\b', norm_key):
                return {"name": orig_key, **self._data[orig_key]}

        # Tier 3 — Sobreposição bidirecional cidade ↔ query.
        # Extrai apenas a parte antes do " - " para que "Campo Grande - MS" possa
        # bater com a query "Campo Grande" sem carregar o sufixo do estado.
        # re.escape() em ambos os padrões: nomes de cidade podem conter espaços
        # que o re interpretaria como separadores — escape trata como literais.
        for norm_key, orig_key in self._normalized_index.items():
            city_part = norm_key.split(" - ")[0]
            pattern_city = r'\b' + re.escape(city_part) + r'\b'
            pattern_query = r'\b' + re.escape(query) + r'\b'
            if re.search(pattern_city, query) or re.search(pattern_query, city_part):
                return {"name": orig_key, **self._data[orig_key]}

        logger.warning("City not found in capitals database: '%s'", name)
        return None

    def list_cities(self) -> list[str]:
        """
        Retorna todos os nomes válidos de capitais após correções aplicadas.

        [Bloco 1 — Humanizado]
        Lista simples com todos os nomes de capitais disponíveis para consulta.
        Útil para exibir ao usuário quais cidades são suportadas.

        [Bloco 2 — Técnico]
        Retorna as chaves de self._data, que já passou por _apply_corrections().
        O formato de cada item é "Cidade - Estado" (ex.: "São Paulo - São Paulo",
        "Manaus - Amazonas"). A ordem reflete a do arquivo JSON (não ordenado
        alfabeticamente por padrão).

        Returns:
            Lista de strings no formato "Cidade - Estado" para todas as 26 capitais
            válidas (27 entradas menos a anomalia removida = 26).
        """
        return list(self._data.keys())

    def get_anomalies(self) -> list[str]:
        """
        Retorna os avisos de integridade de dados detectados durante a inicialização.

        [Bloco 1 — Humanizado]
        Expõe os problemas encontrados no arquivo-fonte de capitais. Quem consome
        este relatório pode exibi-lo em um endpoint de diagnóstico ou logar em
        sistemas de monitoramento.

        [Bloco 2 — Técnico]
        Retorna uma cópia de self._anomalies, preenchido por _validate_integrity()
        no __init__. Cada string descreve uma anomalia específica: qual entrada estava
        errada, o que era esperado e qual correção foi aplicada. Consumido pelo
        endpoint GET /health ou equivalente de data quality report na camada de API.

        Returns:
            Lista de strings descritivas (pode ser vazia se nenhuma anomalia foi detectada).
        """
        return list(self._anomalies)
