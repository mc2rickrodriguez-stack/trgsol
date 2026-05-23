"""
post_process.py — Post-procesamiento de markers para SOLVER TARGET
================================================================
Módulo independiente. No modifica solver_core.py ni recalcula CSM.
Solo genera propuestas y TXT Accumark.

Funciones públicas:
    generar_propuestas(df_usos, df_lotes_det, df_incompletos, capas_min, max_et,
                       df_comp=None, max_len_marker=None, modo_bundles="max")
    generar_txt_accumark(df_propuestas_sel, df_lotes_det, params_txt=None)
    preparar_reinyeccion(df_propuestas_sel)
    comparar_escenarios(antes, despues)
    estilos_para_nomenclatura(df_propuestas_sel, df_lotes_det)
"""

from __future__ import annotations

import math
from functools import reduce
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: Any) -> str:
    return str(s).strip().upper() if pd.notna(s) else ""


def _fmt_num(x: Any) -> str:
    try:
        f = float(x)
        return str(int(f)) if f.is_integer() else str(f).rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _gcd_lista(vals: list[int]) -> int:
    vals = [abs(int(v)) for v in vals if pd.notna(v) and int(v) > 0]
    if not vals:
        return 1
    return reduce(math.gcd, vals)


def _divisores(n: int) -> list[int]:
    """Devuelve divisores de n > 1 en orden ascendente."""
    n = int(n or 0)
    return [k for k in range(2, n + 1) if n % k == 0]


def _nombre_corto(modelo: str, max_len: int = 6) -> str:
    """Default de nomenclatura corta para el campo o:."""
    modelo = _norm(modelo)
    # Caso frecuente: TARGET -> ST35, LTARGET -> LST3, YTARGETLS -> YST3L
    compacto = modelo.replace("350", "3")
    return compacto[:max_len]


def _get_nom(modelo: str, nomenclaturas: dict[str, str] | None = None) -> str:
    modelo = _norm(modelo)
    if nomenclaturas:
        nom = nomenclaturas.get(modelo) or nomenclaturas.get(str(modelo))
        if nom and str(nom).strip():
            return _norm(nom)
    return _nombre_corto(modelo)


def _ancho_col(df: pd.DataFrame) -> str:
    if "ANCHO_CORTABLE" in df.columns:
        return "ANCHO_CORTABLE"
    if "WIDTH_CORTABLE" in df.columns:
        return "WIDTH_CORTABLE"
    return "ancho"


def _ordenar_items_marker(items: pd.DataFrame) -> pd.DataFrame:
    if items is None or items.empty:
        return pd.DataFrame(columns=["ESTILO", "TALLA", "PLACED_BUNDLES"])
    d = items.copy()
    for c in ["ESTILO", "TALLA"]:
        if c in d.columns:
            d[c] = d[c].map(_norm)
    d["PLACED_BUNDLES"] = pd.to_numeric(d["PLACED_BUNDLES"], errors="coerce").fillna(0).astype(int)
    d = d[d["PLACED_BUNDLES"] > 0].copy()
    return d.sort_values(["ESTILO", "TALLA"]).reset_index(drop=True)


def _nombre_marker_desde_items(
    items: pd.DataFrame,
    ancho: float,
    nomenclaturas: dict[str, str] | None = None,
) -> str:
    """
    Construye campo o: agrupando tallas por estilo.

    Ejemplo:
      LTARGET L=2, M=1, TARGET XL=2, M=2, YTARGET S=1
    → LST3 L_2 M_1 ST35 M_2 XL_2 YST3 S_1-65
    """
    req = {"ESTILO", "TALLA", "PLACED_BUNDLES"}
    if items is None or items.empty or not req.issubset(items.columns):
        return f"MARKER_NUEVO-{_fmt_num(ancho)}"

    d = _ordenar_items_marker(items)
    partes: list[str] = []
    estilo_anterior = None

    for _, r in d.iterrows():
        estilo = _norm(r["ESTILO"])
        talla = _norm(r["TALLA"])
        q = int(r["PLACED_BUNDLES"])
        if estilo != estilo_anterior:
            partes.append(_get_nom(estilo, nomenclaturas))
            estilo_anterior = estilo
        partes.append(f"{talla}_{q}")

    return f"{' '.join(partes)}-{_fmt_num(ancho)}"


def _items_marker_from_det(
    df_lotes_det: pd.DataFrame,
    marker: str,
    color: str | None = None,
    k: int | None = None,
) -> pd.DataFrame:
    """Reconstruye ESTILO/TALLA/PLACED_BUNDLES de un marker desde df_lotes_det."""
    cols = ["ESTILO", "TALLA", "PLACED_BUNDLES"]
    if df_lotes_det is None or df_lotes_det.empty:
        return pd.DataFrame(columns=cols)

    d = df_lotes_det[df_lotes_det["MARKER_NAME"].astype(str) == str(marker)].copy()
    if color is not None and "COLOR" in d.columns:
        d = d[d["COLOR"].astype(str) == str(color)]
    if d.empty or not set(cols).issubset(d.columns):
        return pd.DataFrame(columns=cols)

    items = (
        d[cols]
        .drop_duplicates(subset=["ESTILO", "TALLA"])
        .copy()
    )
    items = _ordenar_items_marker(items)

    if k and int(k) > 1 and not items.empty:
        items["PLACED_BUNDLES"] = (items["PLACED_BUNDLES"] // int(k)).astype(int)
        items = items[items["PLACED_BUNDLES"] > 0].copy()

    return _ordenar_items_marker(items)


def _items_text(items: pd.DataFrame) -> str:
    if items is None or items.empty:
        return ""
    return "; ".join(
        f"{r.ESTILO}-{r.TALLA}:{int(r.PLACED_BUNDLES)}"
        for r in _ordenar_items_marker(items).itertuples(index=False)
    )


def calcular_promedio_largo_unitario(df_comp: pd.DataFrame | None) -> pd.DataFrame:
    """
    Calcula longitud promedio por pieza:
        LONG_UNIT_YDS = LENGTH / PLACED_BUNDLES

    Agrupa por ANCHO_CORTABLE + ESTILO + TALLA.
    Requiere que df_comp tenga LENGTH desde solver_core.py.
    """
    cols = ["ANCHO_CORTABLE", "ESTILO", "TALLA", "LONG_UNIT_YDS"]
    if df_comp is None or df_comp.empty:
        return pd.DataFrame(columns=cols)

    df = df_comp.copy()
    if "ANCHO_CORTABLE" not in df.columns and "WIDTH_CORTABLE" in df.columns:
        df = df.rename(columns={"WIDTH_CORTABLE": "ANCHO_CORTABLE"})

    req = {"ANCHO_CORTABLE", "ESTILO", "TALLA", "LENGTH", "PLACED_BUNDLES"}
    if not req.issubset(df.columns):
        return pd.DataFrame(columns=cols)

    d = df[list(req)].copy()
    d["ESTILO"] = d["ESTILO"].map(_norm)
    d["TALLA"] = d["TALLA"].map(_norm)
    d["ANCHO_CORTABLE"] = pd.to_numeric(d["ANCHO_CORTABLE"], errors="coerce")
    d["LENGTH"] = pd.to_numeric(d["LENGTH"], errors="coerce")
    d["PLACED_BUNDLES"] = pd.to_numeric(d["PLACED_BUNDLES"], errors="coerce")
    d = d[(d["LENGTH"] > 0) & (d["PLACED_BUNDLES"] > 0) & d["ANCHO_CORTABLE"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["LONG_UNIT_YDS"] = d["LENGTH"] / d["PLACED_BUNDLES"]
    return (
        d.groupby(["ANCHO_CORTABLE", "ESTILO", "TALLA"], as_index=False)["LONG_UNIT_YDS"]
        .mean()
        .round({"LONG_UNIT_YDS": 4})
    )


def _proyectar_longitud_marker(
    items: pd.DataFrame,
    ancho: float,
    df_long_unit: pd.DataFrame | None,
) -> tuple[float | None, list[str]]:
    """Proyecta longitud = Σ(bundle_propuesto × LONG_UNIT_YDS)."""
    if df_long_unit is None or df_long_unit.empty or items is None or items.empty:
        return None, []

    total = 0.0
    faltantes: list[str] = []
    try:
        ancho_f = float(ancho)
    except Exception:
        ancho_f = ancho

    long_df = df_long_unit.copy()
    long_df["ESTILO"] = long_df["ESTILO"].map(_norm)
    long_df["TALLA"] = long_df["TALLA"].map(_norm)
    long_df["ANCHO_CORTABLE"] = pd.to_numeric(long_df["ANCHO_CORTABLE"], errors="coerce")

    for _, r in _ordenar_items_marker(items).iterrows():
        estilo = _norm(r["ESTILO"])
        talla = _norm(r["TALLA"])
        bundle = float(r["PLACED_BUNDLES"])
        match = long_df[
            (long_df["ANCHO_CORTABLE"] == ancho_f) &
            (long_df["ESTILO"] == estilo) &
            (long_df["TALLA"] == talla)
        ]
        if match.empty:
            faltantes.append(f"{estilo}-{talla}")
            continue
        total += bundle * float(match["LONG_UNIT_YDS"].iloc[0])

    return round(total, 4), faltantes


def estilos_para_nomenclatura(df_propuestas_sel: pd.DataFrame, df_lotes_det: pd.DataFrame) -> list[str]:
    """Lista de estilos reales involucrados en propuestas seleccionadas."""
    estilos: set[str] = set()
    if df_propuestas_sel is None or df_propuestas_sel.empty:
        return []

    for _, row in df_propuestas_sel.iterrows():
        tipo = str(row.get("tipo", ""))
        if tipo == "A_ESCALAMIENTO":
            marker = str(row.get("markers_base", ""))
            color = str(row.get("color", ""))
            k = int(row.get("k", 1) or 1)
            items = _items_marker_from_det(df_lotes_det, marker, color=color, k=k)
        elif tipo == "B_CONSOLIDACION":
            raw = str(row.get("markers_base", ""))
            markers = [m.strip() for m in raw.split(" + ") if m.strip()]
            sub = df_lotes_det[df_lotes_det["MARKER_NAME"].astype(str).isin(markers)].copy()
            items = sub[["ESTILO", "TALLA", "PLACED_BUNDLES"]].drop_duplicates() if not sub.empty else pd.DataFrame()
        else:
            items = pd.DataFrame()

        if not items.empty and "ESTILO" in items.columns:
            estilos.update(items["ESTILO"].map(_norm).tolist())

    return sorted(e for e in estilos if e)


# ─────────────────────────────────────────────────────────────────────────────
# TIPO A — ESCALAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def detectar_escalamiento(
    df_usos: pd.DataFrame,
    df_lotes_det: pd.DataFrame,
    capas_min: int,
) -> pd.DataFrame:
    """
    Detecta markers donde las capas asignadas < capas_min y el VECTOR completo
    de bundles del marker puede dividirse por un mismo factor k.
    """
    cols = [
        "id", "tipo", "markers_base", "nuevo_marker", "ancho", "color",
        "estilo", "talla", "bundles_orig", "bundles_nuevo",
        "capas_orig", "capas_nuevo", "k", "motivo",
    ]
    if df_lotes_det is None or df_lotes_det.empty:
        return pd.DataFrame(columns=cols)

    det = df_lotes_det.copy()
    if "CAPAS_ASIGNADAS" not in det.columns:
        return pd.DataFrame(columns=cols)

    ancho_col = _ancho_col(det)
    group_cols = ["MARKER_NAME", "COLOR", ancho_col]
    base = (
        det[["LOTE_ID", "MARKER_NAME", "COLOR", ancho_col, "CAPAS_ASIGNADAS"]]
        .drop_duplicates(subset=["LOTE_ID", "MARKER_NAME", "COLOR", ancho_col])
        .groupby(group_cols, as_index=False)["CAPAS_ASIGNADAS"]
        .sum()
        .rename(columns={"CAPAS_ASIGNADAS": "CAPAS_TOTALES", ancho_col: "ANCHO_CORTABLE"})
    )
    candidatos = base[base["CAPAS_TOTALES"] < int(capas_min)].copy()
    if candidatos.empty:
        return pd.DataFrame(columns=cols)

    propuestas: list[dict[str, Any]] = []
    for _, row in candidatos.iterrows():
        marker = row["MARKER_NAME"]
        color = row["COLOR"]
        ancho = row["ANCHO_CORTABLE"]
        capas = int(row["CAPAS_TOTALES"])

        vector = _items_marker_from_det(det, marker, color=color, k=None)
        if vector.empty:
            continue

        bundles = vector["PLACED_BUNDLES"].astype(int).tolist()
        gcd_bundles = _gcd_lista(bundles)
        divisores = _divisores(gcd_bundles)
        if not divisores:
            continue

        for k in divisores:
            vector_new = vector.copy()
            vector_new["PLACED_BUNDLES"] = (vector_new["PLACED_BUNDLES"] // k).astype(int)
            if (vector_new["PLACED_BUNDLES"] < 1).any():
                continue

            nuevas_capas = capas * k
            propuestas.append({
                "id": f"A-{len(propuestas)+1:03d}",
                "tipo": "A_ESCALAMIENTO",
                "markers_base": marker,
                "nuevo_marker": _nombre_marker_desde_items(vector_new, ancho),
                "ancho": ancho,
                "color": color,
                "estilo": ", ".join(vector_new["ESTILO"].drop_duplicates().tolist()),
                "talla": ", ".join(vector_new["TALLA"].tolist()),
                "bundles_orig": _items_text(vector),
                "bundles_nuevo": _items_text(vector_new),
                "capas_orig": capas,
                "capas_nuevo": nuevas_capas,
                "k": int(k),
                "motivo": (
                    f"Capas actuales ({capas}) < mínimo ({capas_min}). "
                    f"GCD bundles={gcd_bundles}. Dividir vector completo por {k}: "
                    f"{_items_text(vector)} → {_items_text(vector_new)}; "
                    f"capas proyectadas {capas}→{nuevas_capas}."
                ),
            })

    return pd.DataFrame(propuestas, columns=cols) if propuestas else pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# TIPO B — CONSOLIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def detectar_consolidacion(
    df_usos: pd.DataFrame,
    df_lotes_det: pd.DataFrame,
    max_et: int,
    df_comp: pd.DataFrame | None = None,
    max_len_marker: float | None = None,
    modo_bundles: str = "max",
) -> pd.DataFrame:
    """
    Detecta markers del mismo COLOR + ANCHO con intersección de ESTILO+TALLA.
    Propone consolidarlos si:
      - la unión de ESTILO+TALLA no excede max_et
      - la longitud proyectada no excede max_len_marker, si se proporciona

    modo_bundles:
      - max: toma el mayor bundle por ESTILO+TALLA
      - sum: suma bundles por ESTILO+TALLA
    """
    cols = [
        "id", "tipo", "markers_base", "nuevo_marker", "ancho", "color",
        "estilo", "talla", "n_estilos_talla", "interseccion",
        "modo_bundles", "longitud_proyectada", "max_len_marker",
        "faltantes_longitud", "motivo",
    ]
    if df_lotes_det is None or df_lotes_det.empty:
        return pd.DataFrame(columns=cols)

    modo_bundles = str(modo_bundles or "max").lower().strip()
    if modo_bundles not in {"max", "sum"}:
        modo_bundles = "max"

    det = df_lotes_det.copy()
    ancho_col = _ancho_col(det)
    req = {"MARKER_NAME", "COLOR", ancho_col, "ESTILO", "TALLA", "PLACED_BUNDLES"}
    if not req.issubset(det.columns):
        return pd.DataFrame(columns=cols)

    df_long_unit = calcular_promedio_largo_unitario(df_comp)
    propuestas: list[dict[str, Any]] = []

    for (color, ancho), grp in det.groupby(["COLOR", ancho_col]):
        markers = sorted(grp["MARKER_NAME"].astype(str).unique().tolist())
        if len(markers) < 2:
            continue

        marker_et: dict[str, pd.DataFrame] = {}
        for mkr in markers:
            sub = grp[grp["MARKER_NAME"].astype(str) == str(mkr)][["ESTILO", "TALLA", "PLACED_BUNDLES"]].drop_duplicates().copy()
            sub = _ordenar_items_marker(sub)
            if sub.empty:
                continue
            sub["ET"] = sub["ESTILO"] + "|" + sub["TALLA"]
            marker_et[mkr] = sub

        for m1, m2 in combinations(marker_et.keys(), 2):
            et1 = set(marker_et[m1]["ET"])
            et2 = set(marker_et[m2]["ET"])
            inter = et1 & et2
            if not inter:
                continue

            union = et1 | et2
            if len(union) > int(max_et):
                continue

            combinado_raw = pd.concat([marker_et[m1], marker_et[m2]], ignore_index=True)
            agg_func = "max" if modo_bundles == "max" else "sum"
            combinado = (
                combinado_raw.groupby(["ESTILO", "TALLA"], as_index=False)["PLACED_BUNDLES"]
                .agg(agg_func)
            )
            combinado = _ordenar_items_marker(combinado)

            longitud, faltantes = _proyectar_longitud_marker(combinado, ancho, df_long_unit)
            if max_len_marker is not None and longitud is not None:
                try:
                    if float(longitud) > float(max_len_marker):
                        continue
                except Exception:
                    pass

            nuevo_nombre = _nombre_marker_desde_items(combinado, ancho)
            estilos_txt = ", ".join(combinado["ESTILO"].drop_duplicates().tolist())
            tallas_txt = ", ".join(combinado["TALLA"].tolist())

            motivo_len = ""
            if longitud is not None:
                motivo_len = f" Longitud proyectada={longitud} yds"
                if max_len_marker is not None:
                    motivo_len += f" (máx {max_len_marker})."
                else:
                    motivo_len += "."
            elif df_comp is None or df_comp.empty:
                motivo_len = " No se proyectó longitud porque df_comparacion no fue recibido."
            elif faltantes:
                motivo_len = f" Longitud parcial/no calculada; faltan promedios para {', '.join(faltantes)}."

            propuestas.append({
                "id": f"B-{len(propuestas)+1:03d}",
                "tipo": "B_CONSOLIDACION",
                "markers_base": f"{m1} + {m2}",
                "nuevo_marker": nuevo_nombre,
                "ancho": ancho,
                "color": color,
                "estilo": estilos_txt,
                "talla": tallas_txt,
                "n_estilos_talla": len(union),
                "interseccion": ", ".join(sorted(inter)),
                "modo_bundles": modo_bundles,
                "longitud_proyectada": longitud,
                "max_len_marker": max_len_marker,
                "faltantes_longitud": ", ".join(faltantes),
                "motivo": (
                    f"Markers {m1} y {m2} comparten {len(inter)} ET. "
                    f"Consolidar en 1 marker con {len(union)} ET (máx {max_et}); "
                    f"modo_bundles={modo_bundles}." + motivo_len
                ),
            })

    return pd.DataFrame(propuestas, columns=cols) if propuestas else pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# TIPO C — LOTES INCOMPLETOS
# ─────────────────────────────────────────────────────────────────────────────

def analizar_incompletos(
    df_incompletos: pd.DataFrame,
    df_usos: pd.DataFrame,
    capas_min: int = 20,
    max_et: int = 6,
) -> pd.DataFrame:
    if df_incompletos is None or df_incompletos.empty:
        return pd.DataFrame()

    df = df_incompletos.copy()

    def clasificar(row):
        lbs = row.get("LBS_LOTE", row.get("LBS_TOTAL", 0))
        lbs_min = row.get("LBS_MIN", 0)
        motivo = str(row.get("MOTIVO", "")).lower()
        try:
            lbs_f = float(lbs)
            lbs_min_f = float(lbs_min)
        except Exception:
            lbs_f, lbs_min_f = 0.0, 0.0

        if "residuo" in motivo or "final" in motivo:
            return "RESIDUO_FINAL", "Evaluar absorber en lote anterior o crear marker reducido"
        if "capas" in motivo or (lbs_min_f > 0 and lbs_f < lbs_min_f * 0.5):
            return "CAPAS_INSUFICIENTES", "Aplicar Tipo A (Escalamiento)"
        if "fragment" in motivo or (lbs_min_f > 0 and lbs_min_f <= lbs_f < lbs_min_f * 0.8):
            return "FRAGMENTACION", "Aplicar Tipo B (Consolidación)"
        if "estilo" in motivo or "et" in motivo:
            return "EXCESO_ET", "Reducir combinaciones Estilo+Talla o consolidar selectivamente"
        return "LBS_FUERA_RANGO", "Revisar configuración LBS_MIN/LBS_MAX"

    df[["CAUSA", "ACCION_RECOMENDADA"]] = df.apply(lambda r: pd.Series(clasificar(r)), axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR
# ─────────────────────────────────────────────────────────────────────────────

def generar_propuestas(
    df_usos: pd.DataFrame,
    df_lotes_det: pd.DataFrame,
    df_incompletos: pd.DataFrame,
    capas_min: int = 20,
    max_et: int = 6,
    df_comp: pd.DataFrame | None = None,
    max_len_marker: float | None = None,
    modo_bundles: str = "max",
) -> dict[str, pd.DataFrame]:
    df_a = detectar_escalamiento(df_usos, df_lotes_det, capas_min)
    df_b = detectar_consolidacion(
        df_usos,
        df_lotes_det,
        max_et=max_et,
        df_comp=df_comp,
        max_len_marker=max_len_marker,
        modo_bundles=modo_bundles,
    )
    df_c = analizar_incompletos(df_incompletos, df_usos, capas_min, max_et)

    cols_comunes = ["id", "tipo", "markers_base", "nuevo_marker", "ancho", "color", "motivo"]
    frames = []
    for df in [df_a, df_b]:
        if not df.empty:
            frames.append(df[[c for c in cols_comunes if c in df.columns]])
    df_todas = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols_comunes)
    return {"A": df_a, "B": df_b, "C": df_c, "todas": df_todas}


# ─────────────────────────────────────────────────────────────────────────────
# TXT ACCUMARK
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS_TXT = {
    "d": "0",
    "a": "Q",
    "linea": "T ABIERTA ELC",
    "util": "90",
    "notch": "p-notch",
    "tolerancia": "TOLERANCIA_CORTE",
    "tipo_tela": "A",
}


def _bloque_order(
    nombre_order: str,
    items: pd.DataFrame,
    ancho: float,
    params: dict[str, Any],
) -> str:
    """Genera bloque =:order agrupando tallas bajo cada m:modelo."""
    d = _ordenar_items_marker(items)
    lineas = [
        "=:order",
        f"o:{nombre_order}",
        f"d:{params.get('d', _DEFAULTS_TXT['d'])}",
        f"a:{params.get('a', _DEFAULTS_TXT['a'])}",
        f"l:{params.get('linea', _DEFAULTS_TXT['linea'])}",
        f"w:{_fmt_num(ancho)}",
        f"u:{params.get('util', _DEFAULTS_TXT['util'])}",
        f"g:{params.get('notch', _DEFAULTS_TXT['notch'])}",
        f"b:{params.get('tolerancia', _DEFAULTS_TXT['tolerancia'])}",
        "",
    ]

    tipo_tela = params.get("tipo_tela", _DEFAULTS_TXT["tipo_tela"])
    for modelo, grp in d.groupby("ESTILO", sort=True):
        lineas.append(f"m:{modelo}")
        lineas.append(f"0:{tipo_tela}")
        for _, r in grp.sort_values("TALLA").iterrows():
            lineas.append(f"s:{_norm(r['TALLA'])}")
            lineas.append(f"q:{int(r['PLACED_BUNDLES'])}")
        lineas.append("")

    return "\n".join(lineas).rstrip()


def _items_for_propuesta(row: pd.Series, df_lotes_det: pd.DataFrame) -> pd.DataFrame:
    tipo = str(row.get("tipo", ""))
    if tipo == "A_ESCALAMIENTO":
        marker = str(row.get("markers_base", ""))
        color = str(row.get("color", ""))
        k = int(row.get("k", 1) or 1)
        return _items_marker_from_det(df_lotes_det, marker, color=color, k=k)

    if tipo == "B_CONSOLIDACION":
        markers = [m.strip() for m in str(row.get("markers_base", "")).split(" + ") if m.strip()]
        if not markers or df_lotes_det is None or df_lotes_det.empty:
            return pd.DataFrame(columns=["ESTILO", "TALLA", "PLACED_BUNDLES"])
        sub = df_lotes_det[df_lotes_det["MARKER_NAME"].astype(str).isin(markers)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["ESTILO", "TALLA", "PLACED_BUNDLES"])
        modo = str(row.get("modo_bundles", "max") or "max").lower().strip()
        agg_func = "sum" if modo == "sum" else "max"
        items = (
            sub.groupby(["ESTILO", "TALLA"], as_index=False)["PLACED_BUNDLES"]
            .agg(agg_func)
        )
        return _ordenar_items_marker(items)

    return pd.DataFrame(columns=["ESTILO", "TALLA", "PLACED_BUNDLES"])


def generar_txt_accumark(
    df_propuestas_sel: pd.DataFrame,
    df_lotes_det: pd.DataFrame,
    params_txt: dict | None = None,
) -> str:
    """Genera el contenido TXT Accumark para propuestas seleccionadas."""
    if params_txt is None:
        params_txt = {}
    nomenclaturas = params_txt.get("nomenclaturas", {}) or {}

    bloques: list[str] = []
    if df_propuestas_sel is None or df_propuestas_sel.empty:
        return ""

    for _, row in df_propuestas_sel.iterrows():
        ancho = row.get("ancho", 0)
        items = _items_for_propuesta(row, df_lotes_det)
        if items.empty:
            continue
        nombre_order = _nombre_marker_desde_items(items, ancho, nomenclaturas=nomenclaturas)
        bloques.append(_bloque_order(nombre_order, items, ancho, params_txt))

    return "\n\n".join(bloques)


# ─────────────────────────────────────────────────────────────────────────────
# REINYECCIÓN Y COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def preparar_reinyeccion(df_propuestas_sel: pd.DataFrame) -> pd.DataFrame:
    """Tabla auxiliar para auditar/reinyectar propuestas aprobadas."""
    if df_propuestas_sel is None or df_propuestas_sel.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "id", "tipo", "markers_base", "nuevo_marker", "ancho", "color", "estilo", "talla",
        "bundles_nuevo", "capas_nuevo", "modo_bundles", "longitud_proyectada", "motivo"
    ] if c in df_propuestas_sel.columns]
    return df_propuestas_sel[cols].copy()


def comparar_escenarios(antes: dict[str, Any], despues: dict[str, Any]) -> pd.DataFrame:
    metricas = [
        ("Lotes completos", "lotes_completos", "↑"),
        ("Lotes incompletos", "lotes_incompletos", "↓"),
        ("Fill Rate (%)", "fill_rate", "↑"),
        ("Lbs desperdicio", "lbs_desperdicio", "↓"),
        ("Utilización DDGG (%)", "utilizacion_ddgg", "↑"),
        ("Markers usados", "markers_usados", "—"),
    ]
    filas = []
    for label, key, mejor in metricas:
        va = antes.get(key, None)
        vd = despues.get(key, None)
        delta = None
        if va is not None and vd is not None:
            try:
                delta = round(float(vd) - float(va), 2)
            except Exception:
                delta = None
        filas.append({"Métrica": label, "Antes": va, "Después": vd, "Δ": delta, "Mejor cuando": mejor})
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# DATOS SINTÉTICOS PARA TESTING
# ─────────────────────────────────────────────────────────────────────────────

def datos_sinteticos() -> dict[str, pd.DataFrame]:
    df_lotes_det = pd.DataFrame([
        {"LOTE_ID": 1, "COLOR": "BLACK", "ANCHO_CORTABLE": 65, "MARKER_NAME": "M1", "ESTILO": "LTARGET", "TALLA": "L", "PLACED_BUNDLES": 8, "CAPAS_ASIGNADAS": 10},
        {"LOTE_ID": 1, "COLOR": "BLACK", "ANCHO_CORTABLE": 65, "MARKER_NAME": "M1", "ESTILO": "LTARGET", "TALLA": "M", "PLACED_BUNDLES": 4, "CAPAS_ASIGNADAS": 10},
        {"LOTE_ID": 2, "COLOR": "BLACK", "ANCHO_CORTABLE": 65, "MARKER_NAME": "M2", "ESTILO": "LTARGET", "TALLA": "M", "PLACED_BUNDLES": 1, "CAPAS_ASIGNADAS": 30},
        {"LOTE_ID": 2, "COLOR": "BLACK", "ANCHO_CORTABLE": 65, "MARKER_NAME": "M2", "ESTILO": "TARGET", "TALLA": "XL", "PLACED_BUNDLES": 2, "CAPAS_ASIGNADAS": 30},
    ])
    df_usos = pd.DataFrame()
    df_inc = pd.DataFrame()
    df_comp = pd.DataFrame([
        {"ANCHO_CORTABLE": 65, "ESTILO": "LTARGET", "TALLA": "L", "LENGTH": 10, "PLACED_BUNDLES": 2},
        {"ANCHO_CORTABLE": 65, "ESTILO": "LTARGET", "TALLA": "M", "LENGTH": 5, "PLACED_BUNDLES": 1},
        {"ANCHO_CORTABLE": 65, "ESTILO": "TARGET", "TALLA": "XL", "LENGTH": 12, "PLACED_BUNDLES": 2},
    ])
    return {"df_usos": df_usos, "df_lotes_det": df_lotes_det, "df_incompletos": df_inc, "df_comp": df_comp}
