"""
asignacion_core.py
Motor de asignación de pedidos a lotes — ST350

Lógica:
  Para cada SKU (ESTILO + TALLA + COLOR):
    Lotes ordenados por LOTE_ID (secuencia de corte)
    Pedidos ordenados por PRIORIDAD asc → PO_DATE asc → PEDIDO_LINEA asc

    Para cada lote:
      DISPONIBLE = QUANTITY del lote para ese SKU
      Para cada pedido pendiente (más urgente primero):
        ASIGNADO = min(PENDIENTE_PEDIDO, DISPONIBLE)
        Registrar asignación lote → pedido
        Si DISPONIBLE agotado → parar
      Si queda DISPONIBLE → registrar como SIN_PEDIDO
"""

import math
import pandas as pd
import numpy as np


def norm(s):
    return str(s).strip().upper() if pd.notna(s) else ""


# ─────────────────────────────────────────────────────────────
#  Cargar y normalizar pedidos
# ─────────────────────────────────────────────────────────────

def cargar_pedidos(file_pedidos) -> pd.DataFrame:
    """
    Lee el archivo de balances de pedido y normaliza.
    DEMANDA = abs(BALANCE) × 12  (unidades enteras)
    """
    df = pd.read_excel(file_pedidos, sheet_name="Hoja1")

    required = ["ESTILO BASE", "COLOR", "PEDIDO|LINEA", "PRIORIDAD", "PO DATE", "TALLA STD", "BALANCE"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"El archivo de pedidos no tiene las columnas: {missing}")

    df["ESTILO"]       = df["ESTILO BASE"].map(norm)
    df["COLOR"]        = df["COLOR"].map(norm)
    df["TALLA"]        = df["TALLA STD"].map(norm)
    df["PEDIDO_LINEA"] = df["PEDIDO|LINEA"].astype(str).str.strip()
    df["PRIORIDAD"]    = df["PRIORIDAD"].map(norm)
    df["PO_DATE"]      = pd.to_datetime(df["PO DATE"], errors="coerce")
    df["DEMANDA"]      = df["BALANCE"].abs() * 12
    df["DEMANDA"]      = df["DEMANDA"].apply(math.ceil).astype(int)
    df                 = df[df["DEMANDA"] > 0].copy()

    return df[[
        "PEDIDO_LINEA", "ESTILO", "TALLA", "COLOR",
        "PRIORIDAD", "PO_DATE", "DEMANDA"
    ]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  Asignación de pedidos a lotes
# ─────────────────────────────────────────────────────────────

def asignar_pedidos(df_lotes_det: pd.DataFrame,
                    df_pedidos:   pd.DataFrame) -> dict:
    """
    Asigna pedidos a lotes secuencialmente.

    df_lotes_det : detalle de lotes (LOTE_ID, ESTILO, TALLA, COLOR, QUANTITY)
    df_pedidos   : pedidos normalizados de cargar_pedidos()

    Retorna dict con:
      df_asignacion   : detalle lote × pedido × asignado
      df_resumen_ped  : resumen por pedido (total asignado, pendiente, estado)
      df_sin_pedido   : producción sin pedido asignado
      df_res_prioridad: resumen por prioridad (demanda vs planeado)
    """
    # ── Preparar tabla de producción por lote-SKU ─────────────
    # Solo considerar lotes COMPLETOS — los incompletos no se procesarán
    df_lotes_completos = df_lotes_det[df_lotes_det["COMPLETO"] == "✅ Completo"].copy() \
        if "COMPLETO" in df_lotes_det.columns else df_lotes_det.copy()

    # Agrupar QUANTITY por LOTE_ID + ESTILO + TALLA + COLOR
    prod = (
        df_lotes_completos
        .groupby(["LOTE_ID", "CATEGORIA", "TAMAÑO_LOTE", "COLOR",
                  "ANCHO_CORTABLE", "ESTILO", "TALLA"], as_index=False)
        ["QUANTITY"]
        .sum()
        .sort_values("LOTE_ID")
        .reset_index(drop=True)
    )

    # ── Preparar pedidos pendientes (copia mutable) ───────────
    # Ordenar: PRIORIDAD asc → PO_DATE asc → PEDIDO_LINEA asc
    ped = df_pedidos.copy()
    ped["PENDIENTE"] = ped["DEMANDA"].copy()
    ped = ped.sort_values(
        ["PRIORIDAD", "PO_DATE", "PEDIDO_LINEA"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    out_asig     = []   # detalle asignación
    out_sin_ped  = []   # producción sin pedido

    # ── Iterar por SKU ────────────────────────────────────────
    skus = prod[["ESTILO", "TALLA", "COLOR"]].drop_duplicates()

    for _, sku_row in skus.iterrows():
        estilo = sku_row["ESTILO"]
        talla  = sku_row["TALLA"]
        color  = sku_row["COLOR"]

        # Lotes de este SKU en orden secuencial
        lotes_sku = prod[
            (prod["ESTILO"] == estilo) &
            (prod["TALLA"]  == talla)  &
            (prod["COLOR"]  == color)
        ].copy()

        # Pedidos de este SKU aún pendientes
        mask_sku = (
            (ped["ESTILO"] == estilo) &
            (ped["TALLA"]  == talla)  &
            (ped["COLOR"]  == color)
        )

        for _, lote_row in lotes_sku.iterrows():
            lote_id    = lote_row["LOTE_ID"]
            categoria  = lote_row["CATEGORIA"]
            tam_lote   = lote_row["TAMAÑO_LOTE"]
            ancho      = lote_row["ANCHO_CORTABLE"]
            disponible = int(lote_row["QUANTITY"])

            if disponible <= 0:
                continue

            # Pedidos pendientes de este SKU
            ped_sku = ped[mask_sku & (ped["PENDIENTE"] > 0)].copy()

            if ped_sku.empty:
                # Todo el lote va a SIN_PEDIDO
                out_sin_ped.append({
                    "LOTE_ID":        lote_id,
                    "CATEGORIA":      categoria,
                    "TAMAÑO_LOTE":    tam_lote,
                    "ANCHO_CORTABLE": ancho,
                    "ESTILO":         estilo,
                    "TALLA":          talla,
                    "COLOR":          color,
                    "CANTIDAD_LOTE":  disponible,
                    "SIN_PEDIDO":     disponible,
                    "NOTA":           "🔴 Sin pedido — producción no solicitada",
                })
                continue

            # Asignar FIFO
            for idx in ped_sku.index:
                if disponible <= 0:
                    break

                pendiente_ped = int(ped.at[idx, "PENDIENTE"])
                prioridad     = ped.at[idx, "PRIORIDAD"]
                po_date       = ped.at[idx, "PO_DATE"]
                pedido_linea  = ped.at[idx, "PEDIDO_LINEA"]
                demanda_orig  = int(ped.at[idx, "DEMANDA"])

                asignado = min(pendiente_ped, disponible)

                out_asig.append({
                    "LOTE_ID":        lote_id,
                    "CATEGORIA":      categoria,
                    "TAMAÑO_LOTE":    tam_lote,
                    "ANCHO_CORTABLE": ancho,
                    "ESTILO":         estilo,
                    "TALLA":          talla,
                    "COLOR":          color,
                    "CANTIDAD_LOTE":  int(lote_row["QUANTITY"]),
                    "PEDIDO_LINEA":   pedido_linea,
                    "PRIORIDAD":      prioridad,
                    "PO_DATE":        po_date,
                    "DEMANDA_PEDIDO": demanda_orig,
                    "ASIGNADO":       asignado,
                    "PENDIENTE_ANTES":pendiente_ped,
                    "PENDIENTE_DESP": pendiente_ped - asignado,
                    "ESTADO_LINEA":   "✅ Completo" if pendiente_ped - asignado == 0 else "⚠️ Parcial",
                })

                # Actualizar pendiente en ped
                ped.at[idx, "PENDIENTE"] = pendiente_ped - asignado
                disponible -= asignado

            # Si quedó disponible → SIN_PEDIDO
            if disponible > 0:
                out_sin_ped.append({
                    "LOTE_ID":        lote_id,
                    "CATEGORIA":      categoria,
                    "TAMAÑO_LOTE":    tam_lote,
                    "ANCHO_CORTABLE": ancho,
                    "ESTILO":         estilo,
                    "TALLA":          talla,
                    "COLOR":          color,
                    "CANTIDAD_LOTE":  int(lote_row["QUANTITY"]),
                    "SIN_PEDIDO":     disponible,
                    "NOTA":           "🔴 Sin pedido — exceso de producción",
                })

    # ── Resumen por pedido ────────────────────────────────────
    if out_asig:
        df_asig = pd.DataFrame(out_asig)
        res_ped = (
            df_asig.groupby(
                ["PEDIDO_LINEA","ESTILO","TALLA","COLOR","PRIORIDAD","PO_DATE"],
                as_index=False
            ).agg(
                DEMANDA_PEDIDO=("DEMANDA_PEDIDO", "first"),
                TOTAL_ASIGNADO=("ASIGNADO",       "sum"),
            )
        )
        res_ped["PENDIENTE_FINAL"] = res_ped["DEMANDA_PEDIDO"] - res_ped["TOTAL_ASIGNADO"]
        res_ped["FILL_RATE_%"]     = (
            res_ped["TOTAL_ASIGNADO"] / res_ped["DEMANDA_PEDIDO"] * 100
        ).round(2)
        res_ped["ESTADO_PEDIDO"] = res_ped["PENDIENTE_FINAL"].apply(
            lambda x: "✅ Completo" if x <= 0 else "⚠️ Parcial"
        )
    else:
        df_asig  = pd.DataFrame()
        res_ped  = pd.DataFrame()

    # Solo agregar pedidos con CERO producción (los parciales ya están en res_ped)
    if not ped.empty:
        pedidos_en_resumen = set(res_ped["PEDIDO_LINEA"].tolist()) if not res_ped.empty else set()
        sin_asig = ped[
            (ped["PENDIENTE"] > 0) &
            (~ped["PEDIDO_LINEA"].isin(pedidos_en_resumen))
        ][["PEDIDO_LINEA","ESTILO","TALLA","COLOR","PRIORIDAD","PO_DATE","DEMANDA"]].copy()

        if not sin_asig.empty:
            sin_asig["TOTAL_ASIGNADO"] = 0
            sin_asig["PENDIENTE_FINAL"] = sin_asig["DEMANDA"]
            sin_asig["FILL_RATE_%"]    = 0.0
            sin_asig["ESTADO_PEDIDO"]  = "❌ Sin producción"
            sin_asig = sin_asig.rename(columns={"DEMANDA":"DEMANDA_PEDIDO"})
            if not res_ped.empty:
                res_ped = pd.concat([res_ped, sin_asig], ignore_index=True)
            else:
                res_ped = sin_asig

    # ── Resumen por prioridad ─────────────────────────────────
    if not res_ped.empty:
        res_prior = (
            res_ped.groupby("PRIORIDAD", as_index=False)
            .agg(
                N_PEDIDOS      =("PEDIDO_LINEA",   "count"),
                DEMANDA_TOTAL  =("DEMANDA_PEDIDO",  "sum"),
                ASIGNADO_TOTAL =("TOTAL_ASIGNADO",  "sum"),
                PENDIENTE_TOTAL=("PENDIENTE_FINAL", "sum"),
            )
        )
        res_prior["FILL_RATE_%"] = (
            res_prior["ASIGNADO_TOTAL"] / res_prior["DEMANDA_TOTAL"] * 100
        ).round(2)
        res_prior = res_prior.sort_values("PRIORIDAD").reset_index(drop=True)
    else:
        res_prior = pd.DataFrame()

    df_sin_ped = pd.DataFrame(out_sin_ped) if out_sin_ped else pd.DataFrame(columns=[
        "LOTE_ID","CATEGORIA","TAMAÑO_LOTE","ANCHO_CORTABLE",
        "ESTILO","TALLA","COLOR","CANTIDAD_LOTE","SIN_PEDIDO","NOTA"
    ])

    return {
        "df_asignacion":    df_asig,
        "df_resumen_ped":   res_ped,
        "df_sin_pedido":    df_sin_ped,
        "df_res_prioridad": res_prior,
    }
