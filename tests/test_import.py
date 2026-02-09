def test_import():
    import gputools
    assert hasattr(gputools, "__version__")
    assert isinstance(gputools.__version__, str)
