# Contract tests

Marked `@pytest.mark.contract`. Each test is parametrized over every implementation of one
interface: the real ones in `src/kta/` and the doubles in `tests/fakes/`. A double without a
contract test fails the gate. Run with `uv run pytest -m contract`.
