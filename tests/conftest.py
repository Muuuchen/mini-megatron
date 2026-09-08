import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--exercises", action="store_true", help="run incomplete student tests too")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--exercises"):
        skip = pytest.mark.skip(reason="student exercise; run explicitly with --exercises")
        for item in items:
            if "exercise" in item.keywords: item.add_marker(skip)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)
