"""
Test package for ClaimCheck.
"""
import asyncio
import pytest

pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: async test")
    config.addinivalue_line("markers", "integration: integration test")
    config.addinivalue_line("markers", "slow: slow test")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
