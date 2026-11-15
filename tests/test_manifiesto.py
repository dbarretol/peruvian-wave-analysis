import hashlib

import pytest

from peruvian_wave_analysis import manifiesto


def test_registrar_y_verificar(tmp_path):
    f = tmp_path / "a" / "x.nc"
    f.parent.mkdir()
    f.write_bytes(b"datos")
    e = manifiesto.registrar(tmp_path, f, {"fuente": "prueba"})
    assert e["sha256"] == hashlib.sha256(b"datos").hexdigest()
    assert manifiesto.verificar(tmp_path, f)["fuente"] == "prueba"
    linea = (tmp_path / manifiesto.NOMBRE_SHA).read_text(encoding="utf-8").strip()
    assert linea == f"{e['sha256']}  a/x.nc"


def test_hash_distinto_es_error(tmp_path):
    f = tmp_path / "x.nc"
    f.write_bytes(b"uno")
    manifiesto.registrar(tmp_path, f, {})
    f.write_bytes(b"dos")
    with pytest.raises(manifiesto.ErrorManifiesto, match="no coincide"):
        manifiesto.verificar(tmp_path, f)
    with pytest.raises(manifiesto.ErrorManifiesto, match="otro hash"):
        manifiesto.registrar(tmp_path, f, {})


def test_insumo_sin_registrar_es_error(tmp_path):
    f = tmp_path / "x.nc"
    f.write_bytes(b"uno")
    with pytest.raises(manifiesto.ErrorManifiesto, match="no está en el manifiesto"):
        manifiesto.verificar(tmp_path, f)


def test_reregistrar_conserva_fecha(tmp_path):
    f = tmp_path / "x.nc"
    f.write_bytes(b"uno")
    e1 = manifiesto.registrar(tmp_path, f, {})
    e2 = manifiesto.registrar(tmp_path, f, {})
    assert e1["fecha_acceso_utc"] == e2["fecha_acceso_utc"]
