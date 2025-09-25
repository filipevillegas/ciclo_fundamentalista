import importlib.util
import sys
import types

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _ensure_stub_modules():
    """Cria stubs mínimos para dependências externas não disponíveis durante os testes."""

    if importlib.util.find_spec("gspread") is None:
        gspread_stub = types.ModuleType("gspread")

        class _DummyClient:
            def open_by_key(self, *_args, **_kwargs):
                raise gspread_stub.exceptions.SpreadsheetNotFound("not implemented in tests")

        def _authorize(_creds=None, *_args, **_kwargs):
            return _DummyClient()

        gspread_stub.authorize = _authorize
        gspread_stub.service_account = _authorize

        exceptions_mod = types.ModuleType("gspread.exceptions")

        class SpreadsheetNotFound(Exception):
            pass

        class WorksheetNotFound(Exception):
            pass

        exceptions_mod.SpreadsheetNotFound = SpreadsheetNotFound
        exceptions_mod.WorksheetNotFound = WorksheetNotFound
        gspread_stub.exceptions = exceptions_mod

        sys.modules["gspread"] = gspread_stub
        sys.modules["gspread.exceptions"] = exceptions_mod

    if importlib.util.find_spec("google.auth") is None:
        google_mod = types.ModuleType("google")
        auth_mod = types.ModuleType("google.auth")
        exceptions_mod = types.ModuleType("google.auth.exceptions")

        class DefaultCredentialsError(Exception):
            pass

        def default():
            return types.SimpleNamespace(), None

        auth_mod.default = default
        exceptions_mod.DefaultCredentialsError = DefaultCredentialsError
        google_mod.auth = auth_mod

        sys.modules["google"] = google_mod
        sys.modules["google.auth"] = auth_mod
        sys.modules["google.auth.exceptions"] = exceptions_mod


_ensure_stub_modules()

from ciclo_fundamentalista import (  # noqa: E402
    Config,
    DataValidator,
    FinancialLogger,
    OptimizedSCCSCalculator,
)


def _make_logger(config: Config) -> FinancialLogger:
    config.LOG_LEVEL = "CRITICAL"
    return FinancialLogger(config)


def test_calculate_sccs_vectorized_returns_zero_when_no_indicators():
    config = Config()
    logger = _make_logger(config)
    calculator = OptimizedSCCSCalculator(config, logger)

    df = pd.DataFrame(
        {
            "EMPRESA": ["A", "B", "B"],
        }
    )

    scores = calculator.calculate_sccs_vectorized(df)

    assert np.all(scores == 0)
    assert len(scores) == len(df)


def test_calculate_sccs_vectorized_handles_zero_weight_sum():
    config = Config()
    config.SCCS_WEIGHTS = {key: 0.0 for key in config.SCCS_WEIGHTS}

    logger = _make_logger(config)
    calculator = OptimizedSCCSCalculator(config, logger)

    df = pd.DataFrame(
        {
            "EMPRESA": ["A", "A", "B"],
            "SALES_GROWTH": [10.0, 12.0, 8.0],
        }
    )

    scores = calculator.calculate_sccs_vectorized(df)

    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0)


def test_detect_outliers_skips_constant_series():
    config = Config()
    logger = _make_logger(config)
    validator = DataValidator(config, logger)

    df = pd.DataFrame(
        {
            "EMPRESA": ["A", "A", "B", "B"],
            "SALES_GROWTH": [5.0, 5.0, 5.0, 5.0],
        }
    )

    mask = validator.detect_outliers(df, ["SALES_GROWTH"])

    assert mask["SALES_GROWTH"].sum() == 0
