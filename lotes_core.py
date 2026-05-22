"""
lotes_core.py
Motor de creación de lotes de teñido — ST350

Lógica:
  Para cada COLOR → ANCHO:
    Mientras haya capas disponibles:
      Intentar cerrar un lote empezando por el tamaño más grande
      y bajando en cascada hasta el más pequeño.
      Si ningún tamaño cierra → reportar incompleto y parar.

Merma:
  CSM_MERMA = CSM_MARKER × (1 + pct_merma/100)
  RIB_MERMA = LBS_RIB × (1 + pct_merma_rib/100)
  Solo afecta el consumo de lbs en los lotes (crudo).
  Las unidades (QUANTITY) no cambian.
  Los lotes se crean con el total: LBS_MARKER + LBS_RIB (con mermas).
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
#  Preparar pool
# ─────────────────────────────────────────────────────────────

def preparar_pool(df_usos: pd.DataFrame, df_comp: pd.DataFrame,
                  pct_merma: float = 10.0,
                  pct_merma_rib: float = 8.0) -> tuple:
    """
    Construye la tabla de markers disponibles para loteo.

    pct_merma     : porcentaje de merma sobre el consumo marker (default 10%)
                    CSM_MERMA = CSM_MARKER × (1 + pct_merma/100)
    pct_merma_rib : porcentaje de merma RIB (default 8%)
                    LBS_RIB_MERMA = LBS_RIB × (1 + pct_merma_rib/100)
    """
    factor_merma     = 1 + pct_merma / 100.0
    factor_merma_rib = 1 + pct_merma_rib / 100.0

    # Capas disponibles por marker-color
    capas = (
        df_usos[["MARKER_NAME", "COLOR", "GRUPO", "WIDTH_CORTABLE", "CSM_MARKER", "USOS"]]
        .drop_duplicates(subset=["MARKER_NAME", "COLOR"])
        .rename(columns={"WIDTH_CORTABLE": "ANCHO_CORTABLE", "USOS": "CAPAS_DISP"})
        .copy()
    )

    # CSM con merma marker
    capas["CSM_MERMA"] = (capas["CSM_MARKER"] * factor_merma).round(6)

    # Estilos-talla únicos por marker (frozenset para comparación rápida)
    df_comp = df_comp.copy()
    df_comp["ESTILO_TALLA"] = df_comp["ESTILO"] + "_" + df_comp["TALLA"]
    est_talla = (
        df_comp.groupby("MARKER_NAME")["ESTILO_TALLA"]
        .apply(frozenset)
        .reset_index()
        .rename(columns={"ESTILO_TALLA": "ET_SET"})
    )
    est_talla["N_ET"] = est_talla["ET_SET"].apply(len)

    # Detalle tallas por marker-color (para la salida)
    # Incluir CSM_RIB_TALLA si existe
    cols_det = ["MARKER_NAME", "COLOR", "ESTILO", "TALLA",
                "PLACED_BUNDLES", "CSM_TALLA", "ESTILO_TALLA"]
    if "CSM_RIB_TALLA" in df_comp.columns:
        cols_det.append("CSM_RIB_TALLA")

    det_tallas = df_comp[cols_det].drop_duplicates(
        subset=["MARKER_NAME", "COLOR", "ESTILO", "TALLA"]
    ).copy()

    # Guardar factor merma RIB en pool (para uso en _registrar_lote)
    pool = capas.merge(est_talla, on="MARKER_NAME", how="left")
    pool["FACTOR_MERMA_RIB"] = factor_merma_rib
    return pool, det_tallas


# ─────────────────────────────────────────────────────────────
#  Algoritmo principal de loteo
# ─────────────────────────────────────────────────────────────

def crear_lotes(pool: pd.DataFrame, det_tallas: pd.DataFrame,
                config_lotes: pd.DataFrame) -> dict:
    """
    Crea lotes en cascada por color-ancho.
    Incluye LBS_RIB y LBS_RIB_MERMA calculadas desde CSM_RIB_TALLA.
    El cierre de lote se basa en: LBS_MARKER_MERMA + LBS_RIB_MERMA.
    """
    # ── Calcular CSM_RIB por marker (suma de CSM_RIB_TALLA × capas en det_tallas)
    # CSM_RIB_MARKER = suma de CSM_RIB_TALLA de todas las tallas del marker
    has_rib = "CSM_RIB_TALLA" in det_tallas.columns
    if has_rib:
        # Deduplicar por MARKER_NAME + ESTILO + TALLA antes de sumar:
        # CSM_RIB_TALLA es propiedad de la talla del marker, no del color.
        # Sin esto, si el marker aparece en N colores se infla N veces,
        # sobreestimando CSM_TOTAL_MERMA y asignando muy pocas capas por lote.
        rib_por_marker = (
            det_tallas.drop_duplicates(subset=["MARKER_NAME", "ESTILO", "TALLA"])
            .groupby("MARKER_NAME")["CSM_RIB_TALLA"]
            .sum()
            .reset_index()
            .rename(columns={"CSM_RIB_TALLA": "CSM_RIB_MARKER"})
        )
        pool = pool.merge(rib_por_marker, on="MARKER_NAME", how="left")
        pool["CSM_RIB_MARKER"] = pool["CSM_RIB_MARKER"].fillna(0.0)
    else:
        pool["CSM_RIB_MARKER"] = 0.0

    factor_merma_rib = pool["FACTOR_MERMA_RIB"].iloc[0] if "FACTOR_MERMA_RIB" in pool.columns else 1.08
    pool["CSM_RIB_MERMA"]   = (pool["CSM_RIB_MARKER"] * factor_merma_rib).round(6)
    pool["CSM_TOTAL_MERMA"] = pool["CSM_MERMA"] + pool["CSM_RIB_MERMA"]

    # Ordenar configuración de mayor a menor LBS_MAX
    config = (
        config_lotes.copy()
        .sort_values("LBS_MAX", ascending=False)
        .reset_index(drop=True)
    )

    config_por_cat = {}
    for cat, grp in config.groupby("CATEGORIA"):
        config_por_cat[cat] = (
            grp.sort_values("LBS_MAX", ascending=False)
            .to_dict("records")
        )

    # Pool mutable: capas restantes por (marker, color)
    capas_rem = {}
    for _, row in pool.iterrows():
        key = (row["MARKER_NAME"], row["COLOR"])
        capas_rem[key] = int(row["CAPAS_DISP"])

    lote_id    = 0
    out_det    = []
    out_res    = []
    out_incomp = []

    grupos = sorted(pool["GRUPO"].unique())

    for grupo in grupos:
        cfgs = config_por_cat.get(grupo, [])
        if not cfgs:
            continue

        anchos_grp = sorted(pool[pool["GRUPO"] == grupo]["ANCHO_CORTABLE"].unique())
        colores_grp = sorted(pool[pool["GRUPO"] == grupo]["COLOR"].unique())

        for color in colores_grp:
            for ancho in anchos_grp:

                mk_base = pool[
                    (pool["COLOR"] == color) &
                    (pool["ANCHO_CORTABLE"] == ancho)
                ].copy()

                if mk_base.empty:
                    continue

                max_lotes_total = sum(c["MAX_LOTES"] for c in cfgs)
                lotes_creados   = 0
                lotes_por_tam   = {c["TAMAÑO_LOTE"]: 0 for c in cfgs}

                while lotes_creados < max_lotes_total:
                    mk_disp = mk_base.reset_index(drop=True).copy()
                    capas_rem_vals = [
                        capas_rem.get((mk_disp.at[i,"MARKER_NAME"], mk_disp.at[i,"COLOR"]), 0)
                        for i in range(len(mk_disp))
                    ]
                    mk_disp = mk_disp.assign(CAPAS_REM=capas_rem_vals)
                    mk_disp = mk_disp[mk_disp["CAPAS_REM"] > 0].reset_index(drop=True).copy()

                    if mk_disp.empty:
                        break

                    lote_cerrado = False

                    for cfg in cfgs:
                        tam      = cfg["TAMAÑO_LOTE"]
                        lbs_min  = cfg["LBS_MIN"]
                        lbs_max  = cfg["LBS_MAX"]
                        max_et   = int(cfg["MAX_EST_TALLA"])
                        cap_min  = int(cfg["CAPAS_MIN_MARKER"])
                        max_lot  = int(cfg["MAX_LOTES"])

                        if lotes_por_tam[tam] >= max_lot:
                            continue

                        mk_valid = mk_disp[mk_disp["CAPAS_REM"] >= cap_min].copy()
                        if mk_valid.empty:
                            mk_valid = mk_disp[mk_disp["CAPAS_REM"] > 0].copy()
                            if mk_valid.empty:
                                continue

                        resultado = _construir_lote(
                            mk_valid, capas_rem, lbs_min, lbs_max,
                            max_et, cap_min, color, ancho
                        )

                        if resultado is None:
                            continue

                        asignaciones, lbs_lote, completo = resultado

                        if not completo:
                            if cfg != cfgs[-1]:
                                continue
                            lote_id += 1
                            _registrar_lote(
                                lote_id, asignaciones, lbs_lote, False,
                                grupo, color, ancho, tam, cfg,
                                mk_disp, det_tallas, capas_rem,
                                out_det, out_res, out_incomp,
                                lbs_min, lbs_max
                            )
                            lotes_creados += 1
                            lotes_por_tam[tam] += 1
                            lote_cerrado = True
                            mk_base = pd.DataFrame()
                            break

                        lote_id += 1
                        _registrar_lote(
                            lote_id, asignaciones, lbs_lote, True,
                            grupo, color, ancho, tam, cfg,
                            mk_disp, det_tallas, capas_rem,
                            out_det, out_res, out_incomp,
                            lbs_min, lbs_max
                        )
                        lotes_creados += 1
                        lotes_por_tam[tam] += 1
                        lote_cerrado = True
                        break

                    if not lote_cerrado:
                        break

    # ── Consolidar DataFrames ─────────────────────────────────
    cols_det = [
        "LOTE_ID","CATEGORIA","TAMAÑO_LOTE","COLOR","ANCHO_CORTABLE",
        "MARKER_NAME","ESTILO","TALLA","CAPAS_ASIGNADAS","CSM_X_CAPA",
        "CSM_X_CAPA_MERMA","PLACED_BUNDLES","QUANTITY",
        "LBS_TALLA","LBS_TALLA_MERMA",
        "LBS_RIB_TALLA","LBS_RIB_TALLA_MERMA",
        "LBS_TOTAL_LOTE","LBS_TOTAL_LOTE_MERMA",
        "LBS_RIB_LOTE","LBS_RIB_LOTE_MERMA",
        "LBS_TOTAL_COMBINADO","LBS_TOTAL_COMBINADO_MERMA",
        "COMPLETO"
    ]
    cols_res = [
        "LOTE_ID","CATEGORIA","TAMAÑO_LOTE","COLOR","ANCHO_CORTABLE",
        "N_MARKERS","CAPAS_TOTALES",
        "LBS_TOTAL","LBS_TOTAL_MERMA",
        "LBS_RIB","LBS_RIB_MERMA",
        "LBS_COMBINADO","LBS_COMBINADO_MERMA",
        "COMPLETO"
    ]
    cols_inc = [
        "LOTE_ID","CATEGORIA","TAMAÑO_LOTE","COLOR","ANCHO",
        "LBS_LOTE","LBS_LOTE_MERMA","LBS_RIB","LBS_RIB_MERMA",
        "LBS_COMBINADO","LBS_COMBINADO_MERMA",
        "LBS_MIN","LBS_MAX","MOTIVO"
    ]

    def to_df(lst, cols):
        if not lst:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(lst)
        for col in cols:
            if col not in df.columns:
                df[col] = None
        return df[cols]

    return {
        "df_lotes_det":   to_df(out_det,    cols_det),
        "df_lotes_res":   to_df(out_res,    cols_res),
        "df_incompletos": to_df(out_incomp, cols_inc),
    }


# ─────────────────────────────────────────────────────────────
#  Registrar un lote (detalle + resumen + incompleto si aplica)
# ─────────────────────────────────────────────────────────────

def _registrar_lote(lote_id, asignaciones, lbs_lote, completo,
                    grupo, color, ancho, tam, cfg,
                    mk_disp, det_tallas, capas_rem,
                    out_det, out_res, out_incomp,
                    lbs_min, lbs_max):

    lbs_lote_merma     = 0.0
    lbs_rib_lote       = 0.0
    lbs_rib_lote_merma = 0.0
    has_rib_col = "CSM_RIB_TALLA" in det_tallas.columns

    for mk_name, capas_asig in asignaciones.items():
        capas_rem[(mk_name, color)] = max(
            0, capas_rem.get((mk_name, color), 0) - capas_asig
        )

        mk_row   = mk_disp[mk_disp["MARKER_NAME"] == mk_name].iloc[0]
        csm      = mk_row["CSM_MARKER"]
        csm_merm = mk_row["CSM_MERMA"]
        # factor para proporcionar merma rib por talla
        fmr = mk_row.get("FACTOR_MERMA_RIB", 1.08) if "FACTOR_MERMA_RIB" in mk_row.index else 1.08

        # Tallas del marker para este color
        tallas_mk = det_tallas[
            (det_tallas["MARKER_NAME"] == mk_name) &
            (det_tallas["COLOR"] == color)
        ].copy()
        if tallas_mk.empty:
            tallas_mk = det_tallas[
                det_tallas["MARKER_NAME"] == mk_name
            ].drop_duplicates(subset=["ESTILO", "TALLA"]).copy()

        for _, t_row in tallas_mk.iterrows():
            lbs_talla      = float(t_row.get("CSM_TALLA", 0)) * capas_asig
            factor_merma   = csm_merm / csm if csm > 0 else 1.0
            lbs_talla_merm = lbs_talla * factor_merma
            lbs_lote_merma += lbs_talla_merm

            # RIB por talla
            csm_rib_talla       = float(t_row.get("CSM_RIB_TALLA", 0)) if has_rib_col else 0.0
            lbs_rib_talla       = csm_rib_talla * capas_asig
            lbs_rib_talla_merma = lbs_rib_talla * fmr
            lbs_rib_lote       += lbs_rib_talla
            lbs_rib_lote_merma += lbs_rib_talla_merma

            out_det.append({
                "LOTE_ID":                  lote_id,
                "CATEGORIA":                grupo,
                "TAMAÑO_LOTE":              tam,
                "COLOR":                    color,
                "ANCHO_CORTABLE":           ancho,
                "MARKER_NAME":              mk_name,
                "ESTILO":                   t_row["ESTILO"],
                "TALLA":                    t_row["TALLA"],
                "CAPAS_ASIGNADAS":          capas_asig,
                "CSM_X_CAPA":               round(csm, 6),
                "CSM_X_CAPA_MERMA":         round(csm_merm, 6),
                "PLACED_BUNDLES":           t_row["PLACED_BUNDLES"],
                "QUANTITY":                 int(capas_asig * t_row["PLACED_BUNDLES"]),
                "LBS_TALLA":                round(lbs_talla, 4),
                "LBS_TALLA_MERMA":          round(lbs_talla_merm, 4),
                "LBS_RIB_TALLA":            round(lbs_rib_talla, 4),
                "LBS_RIB_TALLA_MERMA":      round(lbs_rib_talla_merma, 4),
                "LBS_TOTAL_LOTE":           round(lbs_lote, 4),
                "LBS_TOTAL_LOTE_MERMA":     0,    # se actualiza abajo
                "LBS_RIB_LOTE":             0,    # se actualiza abajo
                "LBS_RIB_LOTE_MERMA":       0,    # se actualiza abajo
                "LBS_TOTAL_COMBINADO":      0,    # se actualiza abajo
                "LBS_TOTAL_COMBINADO_MERMA":0,    # se actualiza abajo
                "COMPLETO":                 "✅ Completo" if completo else "⚠️ Incompleto",
            })

    # Totales definitivos del lote
    lbs_lote_merma     = round(lbs_lote_merma, 4)
    lbs_rib_lote       = round(lbs_rib_lote, 4)
    lbs_rib_lote_merma = round(lbs_rib_lote_merma, 4)
    lbs_comb           = round(lbs_lote + lbs_rib_lote, 4)
    lbs_comb_merma     = round(lbs_lote_merma + lbs_rib_lote_merma, 4)

    # Re-evaluar completo con los totales reales combinados (marker + RIB con mermas)
    # Esto corrige casos donde CSM_TOTAL_MERMA del pool difiere del cálculo talla-a-talla
    completo = lbs_min <= lbs_comb_merma <= lbs_max

    for row in out_det:
        if row["LOTE_ID"] == lote_id:
            row["LBS_TOTAL_LOTE_MERMA"]       = lbs_lote_merma
            row["LBS_RIB_LOTE"]               = lbs_rib_lote
            row["LBS_RIB_LOTE_MERMA"]         = lbs_rib_lote_merma
            row["LBS_TOTAL_COMBINADO"]         = lbs_comb
            row["LBS_TOTAL_COMBINADO_MERMA"]   = lbs_comb_merma
            row["COMPLETO"]                    = "✅ Completo" if completo else "⚠️ Incompleto"

    out_res.append({
        "LOTE_ID":              lote_id,
        "CATEGORIA":            grupo,
        "TAMAÑO_LOTE":          tam,
        "COLOR":                color,
        "ANCHO_CORTABLE":       ancho,
        "N_MARKERS":            len(asignaciones),
        "CAPAS_TOTALES":        sum(asignaciones.values()),
        "LBS_TOTAL":            round(lbs_lote, 2),
        "LBS_TOTAL_MERMA":      lbs_lote_merma,
        "LBS_RIB":              lbs_rib_lote,
        "LBS_RIB_MERMA":        lbs_rib_lote_merma,
        "LBS_COMBINADO":        lbs_comb,
        "LBS_COMBINADO_MERMA":  lbs_comb_merma,
        "COMPLETO":             "✅ Completo" if completo else "⚠️ Incompleto",
    })

    if not completo:
        out_incomp.append({
            "LOTE_ID":             lote_id,
            "CATEGORIA":           grupo,
            "TAMAÑO_LOTE":         tam,
            "COLOR":               color,
            "ANCHO":               ancho,
            "LBS_LOTE":            round(lbs_lote, 2),
            "LBS_LOTE_MERMA":      lbs_lote_merma,
            "LBS_RIB":             lbs_rib_lote,
            "LBS_RIB_MERMA":       lbs_rib_lote_merma,
            "LBS_COMBINADO":       lbs_comb,
            "LBS_COMBINADO_MERMA": lbs_comb_merma,
            "LBS_MIN":             lbs_min,
            "LBS_MAX":             lbs_max,
            "MOTIVO":              f"Residuo — {lbs_comb_merma:.0f} lbs crudo combinado (rango: {lbs_min:.0f}–{lbs_max:.0f})",
        })


# ─────────────────────────────────────────────────────────────
#  Construcción greedy de un lote
# ─────────────────────────────────────────────────────────────

def _construir_lote(mk_disp, capas_rem, lbs_min, lbs_max,
                    max_et, capas_min, color, ancho):
    """
    Greedy: toma markers de mayor CSM_TOTAL_MERMA a menor.
    CSM_TOTAL_MERMA = CSM_MERMA (marker) + CSM_RIB_MERMA (rib)
    Retorna (asignaciones, lbs_total_base_marker, completo) o None.
    """
    mk_sorted = mk_disp.sort_values("CSM_TOTAL_MERMA", ascending=False).copy()

    asignaciones  = {}
    lbs_acum_merm = 0.0  # marker + rib con merma (para cerrar lote)
    lbs_acum_base = 0.0  # solo marker sin merma (para reporte base)
    et_acum       = frozenset()

    for _, mk_row in mk_sorted.iterrows():
        mk_name         = mk_row["MARKER_NAME"]
        csm_merm        = mk_row["CSM_MERMA"]
        csm_rib_merm    = mk_row.get("CSM_RIB_MERMA", 0.0)
        csm_total_merm  = mk_row["CSM_TOTAL_MERMA"]
        csm_base        = mk_row["CSM_MARKER"]
        et_mk           = mk_row["ET_SET"]
        cap_disp        = capas_rem.get((mk_name, color), 0)

        if cap_disp <= 0:
            continue

        # Verificar restricción de estilos-talla
        et_nueva = et_acum | et_mk
        if len(et_nueva) > max_et:
            continue

        # Lbs restantes hasta LBS_MAX (usando total con merma)
        lbs_restantes = lbs_max - lbs_acum_merm
        if lbs_restantes <= 0:
            break

        # Capas máximas que caben por lbs (con total merma)
        capas_por_lbs = int(lbs_restantes / csm_total_merm) if csm_total_merm > 0 else cap_disp

        # Capas a asignar
        capas_asig = min(cap_disp, capas_por_lbs)

        if capas_asig <= 0:
            continue

        # Si no cumple capas_min pero es el único marker disponible, aceptar
        if capas_asig < capas_min and len(mk_sorted) > 1:
            continue

        asignaciones[mk_name]  = capas_asig
        lbs_acum_merm         += capas_asig * csm_total_merm
        lbs_acum_base         += capas_asig * csm_base
        et_acum                = et_nueva

        if lbs_acum_merm >= lbs_min:
            break

    if not asignaciones:
        return None

    completo = lbs_min <= lbs_acum_merm <= lbs_max
    return asignaciones, lbs_acum_base, completo
