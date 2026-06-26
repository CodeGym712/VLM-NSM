"""Skip all deprecated Stage 20 tests.

Stage 20 (state-based BC with LSTM/MLP) was abandoned in favor of
Diffusion Policy (vision-based VLA). These tests are kept for reference
but are no longer part of the active test suite.
"""
def pytest_ignore_collect(collection_path, config):
    return True  # ignore everything in this directory
