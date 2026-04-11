"""Unit tests for CapitalsRepository."""
import pytest

from app.Repositories.CapitalsRepository import CapitalsRepository


@pytest.fixture(scope="module")
def repo() -> CapitalsRepository:
    return CapitalsRepository()


def test_exact_match(repo):
    result = repo.find_city("Sao Paulo - Sao Paulo")
    assert result is not None
    assert result["latitude"] == pytest.approx(-23.548, abs=0.01)


def test_case_insensitive(repo):
    result = repo.find_city("curitiba")
    assert result is not None
    assert "latitude" in result


def test_accent_insensitive(repo):
    result = repo.find_city("São Paulo")
    assert result is not None


def test_partial_match(repo):
    result = repo.find_city("Manaus")
    assert result is not None
    assert result["latitude"] == pytest.approx(-3.102, abs=0.01)


def test_city_not_found(repo):
    result = repo.find_city("Mordor")
    assert result is None


def test_list_cities(repo):
    cities = repo.list_cities()
    # 26 valid capitals: the anomalous "Campo Grande - Rio Grande do Norte"
    # was removed (RN capital is Natal, already present in the dataset)
    assert len(cities) == 26
    assert any("Sao Paulo" in c for c in cities)


def test_anomaly_detected(repo):
    """The known data quality issue must be detected and reported."""
    anomalies = repo.get_anomalies()
    assert len(anomalies) > 0
    assert any("Campo Grande" in a and "Rio Grande do Norte" in a for a in anomalies)


def test_campo_grande_resolves_to_ms(repo):
    """'Campo Grande' must resolve to Mato Grosso do Sul, not the erroneous RN entry."""
    result = repo.find_city("Campo Grande")
    assert result is not None
    assert "Mato Grosso do Sul" in result["name"]
    # Latitude for MS Campo Grande is ~-20.4, not ~-5.5 (RN)
    assert result["latitude"] < -15
