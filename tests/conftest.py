import pytest


def _has_opencl():
    try:
        import pyopencl
        platforms = pyopencl.get_platforms()
        return any(p.get_devices() for p in platforms)
    except Exception:
        return False


HAS_OPENCL = _has_opencl()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires an OpenCL device")


def pytest_collection_modifyitems(config, items):
    if HAS_OPENCL:
        return
    skip_gpu = pytest.mark.skip(reason="no OpenCL device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
