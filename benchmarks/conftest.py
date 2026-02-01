"""Benchmark configuration: thread pinning and CLI options."""
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--threads", type=int, default=0,
        help="Intra-op thread count (0=auto)",
    )


@pytest.fixture(scope="session", autouse=True)
def _pin_threads(request):
    threads = request.config.getoption("--threads")
    if threads > 0:
        old_omp = os.environ.get("OMP_NUM_THREADS")
        old_mamba = os.environ.get("MAMBA_THREADS")
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MAMBA_THREADS"] = str(threads)
        yield
        # Restore original values
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp
        if old_mamba is None:
            os.environ.pop("MAMBA_THREADS", None)
        else:
            os.environ["MAMBA_THREADS"] = old_mamba
    else:
        yield


@pytest.fixture(scope="session")
def thread_count(request):
    """Thread count from --threads CLI option, for passing to _create_session."""
    return request.config.getoption("--threads")
