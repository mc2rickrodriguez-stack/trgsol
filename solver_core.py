"""
solver_core.py
Motor MILP para asignación de markers TARGET.

Fórmulas de consumo:
  CSM_MARKER     = (ANCHO_FINAL/36) × (LENGTH/36) × (PESO_OZ/16)
                   ANCHO_FINAL = Width_cortable + 2 pulgadas
  CSM_TALLA      = CSM_MARKER × (Marker_Length_% / 100)
  ESTANDAR_TALLA = (Lbs/Doc_STD / 12) × PLACED_BUNDLES
  DIFERENCIA_LBS = ESTANDAR_TALLA - CSM_TALLA   (+ = solver más eficiente)
  DIFERENCIA_%   = (DIFERENCIA_LBS / ESTANDAR_TALLA) × 100
"""

import math
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, hstack, diags, vstack


def norm(s):
    return str(s).strip().upper() if pd.notna(s) else ""


# ─────────────────────────────────────────────────────────────
#  Carga y normalización
# ─────────────────────────────────────────────────────────────

def cargar_datos(file_markers, file_balance, file_estandares, peso_oz=3.77):
    def check_sheets(buf, required, label):
        xl = pd.ExcelFile(buf)
        missing = [s for s in required if s not in xl.sheet_names]
        if missing:
            raise ValueError(f"{label}: faltan hojas {missing}. Disponibles: {xl.sheet_names}")

    check_sheets(file_markers, ["MARKER", "MODEL"],      "Archivo de Markers")
    check_sheets(file_balance, ["BALANCE", "CAPACIDAD"], "Archivo de Balance")

    # ── MARKER ───────────────────────────────────────────────
    df_marker_raw = pd.read_excel(file_markers, sheet_name="MARKER")
    df_model_raw  = pd.read_excel(file_markers, sheet_name="MODEL")

    df_marker = df_marker_raw[["Marker Name", "Width", "Length"]].copy()
    df_marker.columns = ["MARKER_NAME", "WIDTH_CORTABLE", "LENGTH"]
    df_marker["MARKER_NAME"]    = df_marker["MARKER_NAME"].map(norm)
    df_marker["WIDTH_CORTABLE"] = pd.to_numeric(df_marker["WIDTH_CORTABLE"], errors="coerce")
    df_marker["LENGTH"]         = pd.to_numeric(df_marker["LENGTH"],         errors="coerce")
    df_marker["ANCHO_FINAL"]    = df_marker["WIDTH_CORTABLE"] + 2
    df_marker["CSM_MARKER"]     = (
        (df_marker["ANCHO_FINAL"] / 36) *
        (df_marker["LENGTH"]      / 36) *
        (peso_oz / 16)
    ).round(6)

    # ── MODEL ────────────────────────────────────────────────
    df_model = df_model_raw[[
        "Marker Name", "Model Name", "Size", "Placed Bundles", "Marker Length %"
    ]].copy()
    df_model.columns = ["MARKER_NAME", "MODEL_NAME", "SIZE", "PLACED_BUNDLES", "MK_LEN_PCT"]
    for c in ["MARKER_NAME", "MODEL_NAME", "SIZE"]:
        df_model[c] = df_model[c].map(norm)
    df_model["PLACED_BUNDLES"] = pd.to_numeric(df_model["PLACED_BUNDLES"], errors="coerce").fillna(0)
    df_model["MK_LEN_PCT"]     = pd.to_numeric(df_model["MK_LEN_PCT"],     errors="coerce").fillna(0)

    # ── BALANCE ───────────────────────────────────────────────
    df_balance_raw = pd.read_excel(file_balance, sheet_name="BALANCE")
    df_cap_raw     = pd.read_excel(file_balance, sheet_name="CAPACIDAD")

    df_balance = df_balance_raw[["ESTILO BASE", "COLOR STD", "TALLA STD", "BAL. PREPARAR", "GRUPO"]].copy()
    df_balance.columns = ["ESTILO", "COLOR", "TALLA", "BAL_DZ", "GRUPO"]
    for c in ["ESTILO", "COLOR", "TALLA", "GRUPO"]:
        df_balance[c] = df_balance[c].map(norm)
    df_balance["BAL_DZ"]  = pd.to_numeric(df_balance["BAL_DZ"], errors="coerce").fillna(0)
    df_balance["DEMANDA"] = df_balance["BAL_DZ"].apply(lambda x: math.ceil(abs(x) * 12))
    df_balance = df_balance[df_balance["DEMANDA"] > 0].reset_index(drop=True)

    # ── CAPACIDAD ─────────────────────────────────────────────
    df_cap = df_cap_raw.dropna(subset=["ANCHO_CORTABLE"]).copy()
    df_cap["GRUPO"]          = df_cap["GRUPO"].map(norm)
    df_cap["ANCHO_CORTABLE"] = pd.to_numeric(df_cap["ANCHO_CORTABLE"], errors="coerce")
    df_cap["ANCHO_FINAL"]    = pd.to_numeric(df_cap["ANCHO_FINAL"],    errors="coerce")
    df_cap["DDGG"]           = pd.to_numeric(df_cap["DDGG"],           errors="coerce")
    df_cap["CAPACIDAD"]      = pd.to_numeric(df_cap["CAPACIDAD"],      errors="coerce").fillna(0)
    df_cap = df_cap[df_cap["CAPACIDAD"] > 0].reset_index(drop=True)

    # ── ESTÁNDARES ────────────────────────────────────────────
    df_est_raw = pd.read_excel(file_estandares, sheet_name="Hoja1")
    df_est = df_est_raw[["ESTILO", "Talla", "Lbs/Doc_STD", "Lbs/Doc_RIB"]].copy()
    df_est.columns = ["MODEL_NAME", "SIZE", "LBS_DOC_STD", "LBS_DOC_RIB"]
    df_est["MODEL_NAME"]  = df_est["MODEL_NAME"].map(norm)
    df_est["SIZE"]        = df_est["SIZE"].map(norm)
    df_est["LBS_DOC_STD"] = pd.to_numeric(df_est["LBS_DOC_STD"], errors="coerce").fillna(0)
    df_est["LBS_DOC_RIB"] = pd.to_numeric(df_est["LBS_DOC_RIB"], errors="coerce").fillna(0)

    # ── Tabla maestra marker × talla ─────────────────────────
    df_mk_valid = df_marker[df_marker["CSM_MARKER"] > 0].copy()

    df_mm = df_model.merge(
        df_mk_valid[["MARKER_NAME", "WIDTH_CORTABLE", "ANCHO_FINAL", "LENGTH", "CSM_MARKER"]],
        on="MARKER_NAME", how="inner"
    ).merge(
        df_est, on=["MODEL_NAME", "SIZE"], how="left"
    )

    df_mm["CSM_TALLA"]      = (df_mm["CSM_MARKER"] * (df_mm["MK_LEN_PCT"] / 100)).round(6)
    df_mm["ESTANDAR_TALLA"] = ((df_mm["LBS_DOC_STD"] / 12) * df_mm["PLACED_BUNDLES"]).round(6)
    # RIB: consumo por talla = (Lbs/Doc_RIB / 12) × PLACED_BUNDLES
    df_mm["LBS_DOC_RIB"]    = df_mm["LBS_DOC_RIB"].fillna(0)
    df_mm["CSM_RIB_TALLA"]  = ((df_mm["LBS_DOC_RIB"] / 12) * df_mm["PLACED_BUNDLES"]).round(6)
    df_mm["DIFER_LBS"]      = (df_mm["ESTANDAR_TALLA"] - df_mm["CSM_TALLA"]).round(6)
    df_mm["DIFER_PCT"]      = np.where(
        df_mm["ESTANDAR_TALLA"] > 0,
        (df_mm["DIFER_LBS"] / df_mm["ESTANDAR_TALLA"] * 100).round(2),
        0.0
    )

    df_mm["REQ_KEY"]      = df_mm["MODEL_NAME"] + "_" + df_mm["SIZE"]
    df_balance["REQ_KEY"] = df_balance["ESTILO"] + "_" + df_balance["TALLA"]

    return {
        "df_balance":     df_balance,
        "df_balance_raw": df_balance_raw,
        "df_cap":         df_cap,
        "df_cap_raw":     df_cap_raw,
        "df_mm":          df_mm,
        "df_est":         df_est,
        "df_marker":      df_mk_valid,
    }


# ─────────────────────────────────────────────────────────────
#  Validación previa
# ─────────────────────────────────────────────────────────────

def validar(datos):
    df_balance     = datos["df_balance"]
    df_cap         = datos["df_cap"]
    df_mm          = datos["df_mm"]
    df_est         = datos["df_est"]

    grupos_con_cap = set(df_cap["GRUPO"].unique())
    mk_req_keys    = set(df_mm["REQ_KEY"])
    est_keys       = set(df_est["MODEL_NAME"] + "_" + df_est["SIZE"])

    def get_motivo(row):
        m = []
        if row["REQ_KEY"] not in mk_req_keys: m.append("Sin markers")
        if row["REQ_KEY"] not in est_keys:    m.append("Sin estándares")
        if row["GRUPO"] not in grupos_con_cap: m.append("Grupo sin capacidad definida")
        return " | ".join(m)

    df_balance = df_balance.copy()
    df_balance["MOTIVO_EXCL"] = df_balance.apply(get_motivo, axis=1)

    df_excluidos  = df_balance[df_balance["MOTIVO_EXCL"] != ""].copy()
    df_bal_activo = df_balance[df_balance["MOTIVO_EXCL"] == ""].copy()

    resumen = {
        "total":     len(df_balance),
        "activos":   len(df_bal_activo),
        "excluidos": len(df_excluidos),
        "colores":   df_bal_activo["COLOR"].nunique(),
        "grupos":    sorted(df_bal_activo["GRUPO"].unique().tolist()),
    }
    return df_bal_activo, df_excluidos, resumen


# ─────────────────────────────────────────────────────────────
#  Solver MILP
# ─────────────────────────────────────────────────────────────

def ejecutar_solver(datos, df_bal_activo, params, progress_callback=None):
    df_cap = datos["df_cap"]
    df_mm  = datos["df_mm"]

    MAX_HOLG  = params["max_holgura_pct"]
    PEN       = params["factor_pen"]
    N_MIN_USO = int(params.get("n_min_usos", 1))   # mínimo de usos por marker
    OPTS = {
        "presolve":    True,
        "disp":        False,
        "time_limit":  params["time_limit"],
        "mip_rel_gap": params["mip_rel_gap"],
    }

    anchos_por_grupo = df_cap.groupby("GRUPO")["ANCHO_CORTABLE"].apply(set).to_dict()
    cap_ini   = df_cap.groupby(["GRUPO", "ANCHO_CORTABLE"], as_index=False)["CAPACIDAD"].sum()
    cap_rem   = cap_ini.set_index(["GRUPO", "ANCHO_CORTABLE"])["CAPACIDAD"].to_dict()
    cap_total = dict(cap_rem)

    out_usos    = []
    out_cumpl   = []
    out_comp    = []
    out_errores = []

    grupos = sorted(df_bal_activo["GRUPO"].unique())
    total_colores = df_bal_activo.groupby(["GRUPO", "COLOR"]).ngroups
    resueltos = 0

    for grupo in grupos:
        anchos_grp = anchos_por_grupo.get(grupo, set())
        if not anchos_grp:
            continue

        mm_grp  = df_mm[df_mm["WIDTH_CORTABLE"].isin(anchos_grp)].copy()
        bal_grp = df_bal_activo[df_bal_activo["GRUPO"] == grupo]
        colores = sorted(bal_grp["COLOR"].dropna().unique())

        for color in colores:
            req_c = (
                bal_grp[bal_grp["COLOR"] == color]
                .groupby("REQ_KEY", as_index=False)["DEMANDA"]
                .sum()
            )
            req_c["REQ_ID"] = req_c["REQ_KEY"] + "__" + color

            req_keys_c = set(req_c["REQ_KEY"])

            # ── FILTRO ESTRICTO: solo markers donde TODAS las tallas
            #    están en el balance de requerimientos de este color.
            #    Si un marker tiene aunque sea UNA talla que no se pide,
            #    se excluye completamente para evitar producción no solicitada.
            all_req_keys_por_marker = (
                df_mm.groupby("MARKER_NAME")["REQ_KEY"]
                .apply(set)
                .to_dict()
            )
            def marker_todas_tallas_pedidas(mk_name):
                tallas_marker = all_req_keys_por_marker.get(mk_name, set())
                return tallas_marker.issubset(req_keys_c)

            mk_c = mm_grp[mm_grp["REQ_KEY"].isin(req_keys_c)].copy()
            mk_c = mk_c[mk_c["MARKER_NAME"].apply(marker_todas_tallas_pedidas)]
            mk_c = mk_c[mk_c.apply(
                lambda r: cap_rem.get((grupo, r["WIDTH_CORTABLE"]), 0) > 0, axis=1
            )]

            if mk_c.empty:
                msg = "Sin markers válidos — todos tienen tallas fuera del balance de este color"
                out_errores.append({"GRUPO": grupo, "COLOR": color, "MOTIVO": msg})
                resueltos += 1
                if progress_callback:
                    progress_callback(grupo, color, total_colores, resueltos, f"⚠️ {msg}")
                continue

            map_reqid = dict(zip(req_c["REQ_KEY"], req_c["REQ_ID"]))
            mk_c = mk_c.copy()
            mk_c["REQ_ID"] = mk_c["REQ_KEY"].map(map_reqid)

            var_keys = (
                mk_c[["MARKER_NAME", "WIDTH_CORTABLE", "ANCHO_FINAL", "CSM_MARKER"]]
                .drop_duplicates(subset=["MARKER_NAME"])
                .reset_index(drop=True)
            )
            var_keys["VAR_IDX"] = np.arange(len(var_keys))

            mk_f = mk_c.merge(var_keys[["MARKER_NAME", "VAR_IDX"]], on="MARKER_NAME", how="left")
            mk_f = mk_f[mk_f["REQ_ID"].notna()]

            req_ids = req_c["REQ_ID"].to_numpy()
            req_map = {rid: i for i, rid in enumerate(req_ids)}
            dem_vec = req_c.set_index("REQ_ID").loc[req_ids, "DEMANDA"].to_numpy(dtype=float)

            n_req = len(req_ids)
            n_x   = len(var_keys)

            rows_d = mk_f["REQ_ID"].map(req_map).to_numpy()
            cols_d = mk_f["VAR_IDX"].to_numpy()
            data_d = mk_f["PLACED_BUNDLES"].to_numpy(dtype=float)
            A_dem  = csr_matrix((data_d, (rows_d, cols_d)), shape=(n_req, n_x))

            anchos_c    = sorted(var_keys["WIDTH_CORTABLE"].unique())
            ancho_map   = {a: i for i, a in enumerate(anchos_c)}
            cap_rem_vec = np.array([cap_rem.get((grupo, a), 0.0) for a in anchos_c])

            a_rows = var_keys["WIDTH_CORTABLE"].map(ancho_map).to_numpy(dtype=int)
            a_cols = var_keys["VAR_IDX"].to_numpy(dtype=int)
            a_data = var_keys["CSM_MARKER"].to_numpy(dtype=float)
            A_cap  = csr_matrix((a_data, (a_rows, a_cols)), shape=(len(anchos_c), n_x))

            # ── Variables binarias z_i para restricción de mínimo de usos ──
            # Si N_MIN_USO > 1: x_i >= N_min*z_i  y  x_i <= M*z_i
            # Variables: [x_0..x_{n_x-1}, u_0..u_{n_req-1}, z_0..z_{n_x-1}]
            use_min = N_MIN_USO > 1
            n_z     = n_x if use_min else 0
            n_total = n_x + n_req + n_z
            M_big   = 1e6   # cota superior para x

            c_obj = np.concatenate([
                var_keys["CSM_MARKER"].to_numpy(dtype=float),  # x
                np.full(n_req, PEN),                            # u (slack demanda)
                np.zeros(n_z),                                  # z (binarias, sin costo)
            ])
            integrality = np.concatenate([
                np.ones(n_x,   dtype=int),   # x enteras
                np.zeros(n_req, dtype=int),  # u continuas
                np.ones(n_z,   dtype=int),   # z binarias
            ])
            lb = np.zeros(n_total)
            ub = np.concatenate([
                np.full(n_x,   M_big),
                dem_vec,
                np.ones(n_z),   # z ∈ {0,1}
            ])

            # Restricciones base
            A1 = hstack([A_dem,
                         diags(np.ones(n_req), format="csr"),
                         csr_matrix((n_req, n_z))], format="csr")
            A2 = hstack([A_dem,
                         csr_matrix((n_req, n_req)),
                         csr_matrix((n_req, n_z))], format="csr")
            A3 = hstack([A_cap,
                         csr_matrix((len(anchos_c), n_req)),
                         csr_matrix((len(anchos_c), n_z))], format="csr")

            lb_tot = np.concatenate([dem_vec,              np.full(n_req, -np.inf), np.full(len(anchos_c), -np.inf)])
            ub_tot = np.concatenate([np.full(n_req, np.inf), (1+MAX_HOLG)*dem_vec, cap_rem_vec])

            if use_min:
                # R4: x_i >= N_min * z_i  →  x_i - N_min*z_i >= 0
                A4_x = diags(np.ones(n_x), format="csr")
                A4_z = diags(np.full(n_x, -float(N_MIN_USO)), format="csr")
                A4   = hstack([A4_x, csr_matrix((n_x, n_req)), A4_z], format="csr")
                lb4  = np.zeros(n_x)
                ub4  = np.full(n_x, np.inf)

                # R5: x_i <= M * z_i  →  x_i - M*z_i <= 0
                A5_x = diags(np.ones(n_x), format="csr")
                A5_z = diags(np.full(n_x, -M_big), format="csr")
                A5   = hstack([A5_x, csr_matrix((n_x, n_req)), A5_z], format="csr")
                lb5  = np.full(n_x, -np.inf)
                ub5  = np.zeros(n_x)

                A_tot  = vstack([A1, A2, A3, A4, A5], format="csr")
                lb_tot = np.concatenate([lb_tot, lb4, lb5])
                ub_tot = np.concatenate([ub_tot, ub4, ub5])
            else:
                A_tot = vstack([A1, A2, A3], format="csr")

            try:
                res = milp(
                    c=c_obj,
                    integrality=integrality,
                    bounds=Bounds(lb, ub),
                    constraints=LinearConstraint(A_tot, lb=lb_tot, ub=ub_tot),
                    options=OPTS,
                )

                if res.status != 0:
                    msg = f"HiGHS status {res.status}: {res.message}"
                    out_errores.append({"GRUPO": grupo, "COLOR": color, "MOTIVO": msg})
                    resueltos += 1
                    if progress_callback:
                        progress_callback(grupo, color, total_colores, resueltos, f"❌ {msg}")
                    continue

                x = np.rint(res.x[:n_x]).astype(int)
                u = res.x[n_x:n_x + n_req]

                cons_ancho = A_cap.dot(x)
                for ai, ancho in enumerate(anchos_c):
                    cap_rem[(grupo, ancho)] = max(
                        0.0, cap_rem.get((grupo, ancho), 0.0) - cons_ancho[ai]
                    )

                # ── USOS_MARKER ────────────────────────────────
                o1 = var_keys.copy()
                o1["GRUPO"]          = grupo
                o1["COLOR"]          = color
                o1["USOS"]           = x
                o1["LBS_CONSUMIDAS"] = (x * o1["CSM_MARKER"]).round(4)
                o1 = o1[o1["USOS"] > 0][[
                    "MARKER_NAME", "COLOR", "GRUPO",
                    "WIDTH_CORTABLE", "ANCHO_FINAL",
                    "USOS", "CSM_MARKER", "LBS_CONSUMIDAS"
                ]]
                out_usos.append(o1)

                # ── CUMPLIMIENTO ───────────────────────────────
                prod_vec = A_dem.dot(x)
                o2 = pd.DataFrame({
                    "REQ_ID":      req_ids,
                    "GRUPO":       grupo,
                    "COLOR":       color,
                    "DEMANDA":     dem_vec.astype(int),
                    "PRODUCIDO":   prod_vec.astype(int),
                    "FALTANTE":    np.ceil(u).astype(int),
                    "FILL_RATE_%": np.where(
                        dem_vec > 0, (100.0 * prod_vec / dem_vec).round(2), 0.0
                    ),
                })
                out_cumpl.append(o2)

                # ── COMPARACION_ESTANDAR ───────────────────────
                mk_usados = var_keys[x > 0]["MARKER_NAME"].tolist()
                usos_map  = dict(zip(var_keys["MARKER_NAME"], x))

                comp_rows = mk_c[mk_c["MARKER_NAME"].isin(mk_usados)].copy()
                comp_rows = comp_rows.drop_duplicates(subset=["MARKER_NAME", "MODEL_NAME", "SIZE"])
                comp_rows["USOS"]  = comp_rows["MARKER_NAME"].map(usos_map)
                comp_rows["COLOR"] = color
                comp_rows["GRUPO"] = grupo

                # Lbs reales multiplicadas por USOS
                comp_rows["LBS_PLANEADAS"] = (comp_rows["CSM_TALLA"]      * comp_rows["USOS"]).round(4)
                comp_rows["LBS_ESTANDAR"]  = (comp_rows["ESTANDAR_TALLA"] * comp_rows["USOS"]).round(4)
                comp_rows["DIFER_LBS"]     = (comp_rows["LBS_ESTANDAR"]   - comp_rows["LBS_PLANEADAS"]).round(4)
                comp_rows["DIFER_PCT"]     = np.where(
                    comp_rows["LBS_ESTANDAR"] > 0,
                    (comp_rows["DIFER_LBS"] / comp_rows["LBS_ESTANDAR"] * 100).round(2),
                    0.0
                )

                comp_rows = comp_rows[[
                    "MARKER_NAME", "COLOR", "GRUPO",
                    "WIDTH_CORTABLE", "ANCHO_FINAL", "LENGTH",
                    "MODEL_NAME", "SIZE", "PLACED_BUNDLES", "MK_LEN_PCT",
                    "USOS", "CSM_MARKER", "CSM_TALLA", "ESTANDAR_TALLA",
                    "LBS_PLANEADAS", "LBS_ESTANDAR", "DIFER_LBS", "DIFER_PCT",
                    "CSM_RIB_TALLA",
                ]].rename(columns={
                    "MODEL_NAME":    "ESTILO",
                    "SIZE":          "TALLA",
                    "WIDTH_CORTABLE":"ANCHO_CORTABLE",
                })
                out_comp.append(comp_rows)

                fill_avg = o2["FILL_RATE_%"].mean()
                resueltos += 1
                if progress_callback:
                    progress_callback(
                        grupo, color, total_colores, resueltos,
                        f"✅ {grupo}·{color} — {(x>0).sum()} markers | Fill {fill_avg:.1f}%"
                    )

            except Exception as e:
                msg = str(e)
                out_errores.append({"GRUPO": grupo, "COLOR": color, "MOTIVO": msg})
                resueltos += 1
                if progress_callback:
                    progress_callback(grupo, color, total_colores, resueltos, f"❌ {msg}")

    # ── Consolidar ────────────────────────────────────────────
    def safe_concat(lst, cols):
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=cols)

    df_usos  = safe_concat(out_usos, [
        "MARKER_NAME","COLOR","GRUPO","WIDTH_CORTABLE","ANCHO_FINAL",
        "USOS","CSM_MARKER","LBS_CONSUMIDAS"
    ])
    df_cumpl = safe_concat(out_cumpl, [
        "REQ_ID","GRUPO","COLOR","DEMANDA","PRODUCIDO","FALTANTE","FILL_RATE_%"
    ])
    df_comp  = safe_concat(out_comp, [
        "MARKER_NAME","COLOR","GRUPO","ANCHO_CORTABLE","ANCHO_FINAL","LENGTH",
        "ESTILO","TALLA","PLACED_BUNDLES","MK_LEN_PCT",
        "USOS","CSM_MARKER","CSM_TALLA","ESTANDAR_TALLA",
        "LBS_PLANEADAS","LBS_ESTANDAR","DIFER_LBS","DIFER_PCT",
        "CSM_RIB_TALLA",
    ])
    df_errores = (
        pd.DataFrame(out_errores) if out_errores
        else pd.DataFrame(columns=["GRUPO","COLOR","MOTIVO"])
    )

    # ── Balance DDGG ──────────────────────────────────────────
    bal_ddgg = []
    for (grp, ancho), cap_ini_v in cap_total.items():
        consumido  = cap_ini_v - cap_rem.get((grp, ancho), 0.0)
        diferencia = cap_ini_v - consumido
        util       = 100.0 * consumido / cap_ini_v if cap_ini_v > 0 else 0.0
        fila_cap   = df_cap[(df_cap["GRUPO"] == grp) & (df_cap["ANCHO_CORTABLE"] == ancho)]
        bal_ddgg.append({
            "GRUPO":           grp,
            "DDGG":            fila_cap["DDGG"].values[0]        if not fila_cap.empty else None,
            "ANCHO_FINAL":     fila_cap["ANCHO_FINAL"].values[0]  if not fila_cap.empty else None,
            "ANCHO_CORTABLE":  ancho,
            "CAPACIDAD_TOTAL": round(cap_ini_v, 2),
            "CONSUMIDO":       round(consumido, 2),
            "DIFERENCIA":      round(diferencia, 2),
            "UTILIZACION_%":   round(util, 2),
        })
    df_ddgg = pd.DataFrame(bal_ddgg)

    return {
        "df_usos":        df_usos,
        "df_cumpl":       df_cumpl,
        "df_ddgg":        df_ddgg,
        "df_comparacion": df_comp,
        "df_errores":     df_errores,
    }
