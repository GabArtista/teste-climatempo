"""
Repository for Brazilian state capitals data.

Data quality note: The source capitals.json contains one known anomaly:
  - "Campo Grande - Rio Grande do Norte" is an incorrect entry.
    The capital of Rio Grande do Norte is Natal, not Campo Grande.
    Campo Grande is the capital of Mato Grosso do Sul.
  - This repository detects and logs such inconsistencies on load,
    and resolves "Campo Grande" queries to the correct city (Mato Grosso do Sul).
"""
import json
import logging
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
    """Lowercase, strip whitespace, and remove diacritical marks."""
    text = text.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


class DataIntegrityWarning(UserWarning):
    """Raised when anomalies are detected in the capitals data file."""


class CapitalsRepository:
    """
    Provides access to Brazilian state capitals with their geographic coordinates.

    Performs integrity validation on load and gracefully handles known
    data quality issues in the source file.
    """

    def __init__(self, data_path: Path = _CAPITALS_FILE) -> None:
        """Load and validate capitals from JSON file."""
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
        Detect data quality issues and log them.

        Current checks:
        - Duplicate state entries (two cities claiming the same state).
        - Entries matching known incorrect records.

        Returns a list of anomaly descriptions (for reporting).
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
        """Remove known erroneous entries from the dataset."""
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
        Find a capital by name — case-insensitive and accent-insensitive.

        Resolution order:
        1. Exact normalized match
        2. Query string contained within a key
        3. City part of key contained in / overlapping with query

        Ambiguity note: querying "Campo Grande" returns the correct
        capital of Mato Grosso do Sul (the anomalous RN entry is removed).

        Args:
            name: City name or partial name to search.

        Returns:
            Dict with 'name', 'latitude', 'longitude', or None if not found.
        """
        query = _normalize(name)

        # 1. Exact normalized match
        if query in self._normalized_index:
            key = self._normalized_index[query]
            return {"name": key, **self._data[key]}

        # 2. Query contained in key (e.g. "sao paulo" in "sao paulo - sao paulo")
        for norm_key, orig_key in self._normalized_index.items():
            if query in norm_key:
                return {"name": orig_key, **self._data[orig_key]}

        # 3. City part overlaps with query
        for norm_key, orig_key in self._normalized_index.items():
            city_part = norm_key.split(" - ")[0]
            if city_part in query or query in city_part:
                return {"name": orig_key, **self._data[orig_key]}

        logger.warning("City not found in capitals database: '%s'", name)
        return None

    def list_cities(self) -> list[str]:
        """Return all valid capital city names."""
        return list(self._data.keys())

    def get_anomalies(self) -> list[str]:
        """Return data integrity warnings detected during load."""
        return list(self._anomalies)
