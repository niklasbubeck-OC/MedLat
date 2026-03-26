"""Fast pytest checks (import, registry, one tiny model). Run on every CI push."""

import pytest


def test_medlat_import_and_registry_nonempty():
    import medlat
    from medlat import available_models

    names = list(available_models())
    assert len(names) > 0, "registry should list at least one model"


def test_get_model_continuous_small():
    from medlat import get_model

    m = get_model("continuous.aekl.f4_d3", img_size=32, dims=2)
    assert m is not None


def test_get_model_info():
    from medlat import get_model_info

    info = get_model_info("continuous.aekl.f4_d3")
    assert info.name == "continuous.aekl.f4_d3"
