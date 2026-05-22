"""
seccion_post_process.py
========================
Fragmento listo para pegar en App.py (Sección Post Process Markers).

Uso:
    from seccion_post_process import render_post_process
    render_post_process()

Requiere session_state con:
    st.session_state._res_solver  → dict con df_usos y df_comparacion
    st.session_state._res_lotes   → dict con df_lotes_det, df_lotes_res, df_incompletos
"""

import io
import streamlit as st
import pandas as pd

from post_process import (
    generar_propuestas,
    generar_txt_accumark,
    preparar_reinyeccion,
    comparar_escenarios,
    estilos_para_nomenclatura,
    _nombre_corto,
)


def render_post_process():
    """Renderiza la sección completa de Post Process Markers."""

    with st.expander("🔧 Post Process Markers", expanded=False):
        solver_ok = "_res_solver" in st.session_state and st.session_state._res_solver
        lotes_ok = "_res_lotes" in st.session_state and st.session_state._res_lotes

        if not solver_ok or not lotes_ok:
            st.info("⚠️ Ejecuta el Solver y crea Lotes antes de usar este módulo.")
            return

        df_usos = st.session_state._res_solver.get("df_usos", pd.DataFrame()).copy()
        df_comp = st.session_state._res_solver.get("df_comparacion", pd.DataFrame()).copy()
        df_lotes_det = st.session_state._res_lotes.get("df_lotes_det", pd.DataFrame()).copy()

        raw_inc = st.session_state._res_lotes.get("df_incompletos", None)
        df_incompletos = pd.DataFrame() if raw_inc is None or getattr(raw_inc, "empty", False) else raw_inc.copy()

        # ── Parámetros de detección ───────────────────────────────────────────
        st.subheader("⚙️ Parámetros de detección")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            capas_min = st.number_input(
                "CAPAS_MIN_MARKER",
                min_value=1,
                value=20,
                step=1,
                help="Markers con menos capas se evalúan para escalamiento por GCD del vector completo de bundles.",
            )
        with col2:
            max_et = st.number_input(
                "MAX_EST_TALLA",
                min_value=1,
                value=6,
                step=1,
                help="Máximo de combinaciones Estilo+Talla permitidas en un marker consolidado.",
            )
        with col3:
            max_len_marker = st.number_input(
                "Longitud máxima marker",
                min_value=1.0,
                value=90.0,
                step=1.0,
                format="%.2f",
                help="Filtra consolidaciones cuya longitud proyectada exceda este valor. Proyección = Σ(bundle × promedio LENGTH/PLACED_BUNDLES).",
            )
        with col4:
            modo_bundles = st.selectbox(
                "Bundles al consolidar",
                options=["max", "sum"],
                index=0,
                help="max = mayor bundle por Estilo+Talla; sum = suma bundles al combinar markers.",
            )

        if df_comp.empty or "LENGTH" not in df_comp.columns:
            st.warning(
                "⚠️ df_comparacion no trae LENGTH. La consolidación funcionará, pero la longitud proyectada no podrá calcularse. "
                "Usa el solver_core.py actualizado incluido en esta entrega."
            )

        if st.button("🔍 Detectar Propuestas", type="primary"):
            with st.spinner("Analizando markers y lotes…"):
                resultado = generar_propuestas(
                    df_usos=df_usos,
                    df_lotes_det=df_lotes_det,
                    df_incompletos=df_incompletos,
                    capas_min=int(capas_min),
                    max_et=int(max_et),
                    df_comp=df_comp,
                    max_len_marker=float(max_len_marker),
                    modo_bundles=modo_bundles,
                )
                st.session_state._pp_resultado = resultado
                st.session_state._pp_params = {
                    "capas_min": int(capas_min),
                    "max_et": int(max_et),
                    "max_len_marker": float(max_len_marker),
                    "modo_bundles": modo_bundles,
                }

        if "_pp_resultado" not in st.session_state:
            return

        resultado = st.session_state._pp_resultado
        df_a = resultado.get("A", pd.DataFrame())
        df_b = resultado.get("B", pd.DataFrame())
        df_c = resultado.get("C", pd.DataFrame())

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tipo A — Escalamiento", len(df_a))
        c2.metric("Tipo B — Consolidación", len(df_b))
        c3.metric("Lotes incompletos analizados", len(df_c))

        tab_a, tab_b, tab_c, tab_txt, tab_reinj = st.tabs([
            "⬆️ A — Escalamiento",
            "🔗 B — Consolidación",
            "🧩 C — Lotes Incompletos",
            "📄 TXT Accumark",
            "📦 Reinyección",
        ])

        # ── A ────────────────────────────────────────────────────────────────
        with tab_a:
            if df_a.empty:
                st.success("✅ No se detectaron markers escalables con capas insuficientes.")
                st.session_state._pp_sel_a = pd.DataFrame()
            else:
                st.caption("Escalamiento calculado con el GCD del vector completo de bundles del marker.")
                df_a_disp = df_a.copy()
                df_a_disp.insert(0, "✅ Implementar", False)
                edited_a = st.data_editor(
                    df_a_disp,
                    column_config={
                        "✅ Implementar": st.column_config.CheckboxColumn(default=False),
                        "motivo": st.column_config.TextColumn(width="large"),
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="editor_a",
                )
                st.session_state._pp_sel_a = edited_a[edited_a["✅ Implementar"]].drop(columns=["✅ Implementar"])

        # ── B ────────────────────────────────────────────────────────────────
        with tab_b:
            if df_b.empty:
                st.success("✅ No se detectaron consolidaciones válidas bajo las reglas actuales.")
                st.session_state._pp_sel_b = pd.DataFrame()
            else:
                st.caption("Consolidaciones filtradas por MAX_EST_TALLA y longitud máxima proyectada.")
                df_b_disp = df_b.copy()
                df_b_disp.insert(0, "✅ Implementar", False)
                edited_b = st.data_editor(
                    df_b_disp,
                    column_config={
                        "✅ Implementar": st.column_config.CheckboxColumn(default=False),
                        "motivo": st.column_config.TextColumn(width="large"),
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="editor_b",
                )
                st.session_state._pp_sel_b = edited_b[edited_b["✅ Implementar"]].drop(columns=["✅ Implementar"])

        # ── C ────────────────────────────────────────────────────────────────
        with tab_c:
            if df_c.empty:
                st.success("✅ No hay lotes incompletos.")
            else:
                cols_mostrar = [
                    c for c in [
                        "LOTE_ID", "COLOR", "ANCHO", "ANCHO_CORTABLE", "LBS_LOTE", "LBS_MIN", "LBS_MAX",
                        "CAUSA", "ACCION_RECOMENDADA", "MOTIVO",
                    ] if c in df_c.columns
                ]
                st.dataframe(
                    df_c[cols_mostrar],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ACCION_RECOMENDADA": st.column_config.TextColumn(width="medium"),
                        "MOTIVO": st.column_config.TextColumn(width="large"),
                    },
                )
                if "CAUSA" in df_c.columns:
                    st.caption("Distribución por causa")
                    st.dataframe(
                        df_c["CAUSA"].value_counts().rename_axis("Causa").reset_index(name="Lotes"),
                        hide_index=True,
                    )

        # ── TXT ──────────────────────────────────────────────────────────────
        with tab_txt:
            st.subheader("📄 Generar TXT Accumark")
            sel_a = st.session_state.get("_pp_sel_a", pd.DataFrame())
            sel_b = st.session_state.get("_pp_sel_b", pd.DataFrame())
            sel_todas = pd.concat([sel_a, sel_b], ignore_index=True)

            if sel_todas.empty:
                st.info("Selecciona propuestas en A y/o B para generar el TXT.")
            else:
                st.caption(f"{len(sel_todas)} propuestas seleccionadas.")

                with st.expander("⚙️ Parámetros del TXT", expanded=True):
                    col_p1, col_p2, col_p3, col_p4, col_p5, col_p6, col_p7 = st.columns(7)
                    d_param = col_p1.text_input("d:", value="0")
                    a_param = col_p2.text_input("a:", value="Q")
                    linea = col_p3.text_input("l:", value="T ABIERTA ELC")
                    util = col_p4.text_input("u:", value="90")
                    notch = col_p5.text_input("g:", value="p-notch")
                    tolerancia = col_p6.text_input("b:", value="TOLERANCIA_CORTE")
                    tipo_tela = col_p7.text_input("0:", value="A")

                estilos = estilos_para_nomenclatura(sel_todas, df_lotes_det)
                nomenclaturas = {}
                if estilos:
                    st.caption("Nomenclatura para `o:`. El modelo real en `m:` no se modifica.")
                    df_nom_base = pd.DataFrame({
                        "ESTILO": estilos,
                        "NOMENCLATURA": [_nombre_corto(e) for e in estilos],
                    })
                    df_nom = st.data_editor(
                        df_nom_base,
                        use_container_width=True,
                        hide_index=True,
                        key="editor_nomenclatura",
                    )
                    nomenclaturas = {
                        str(r["ESTILO"]).strip().upper(): str(r["NOMENCLATURA"]).strip().upper()
                        for _, r in df_nom.iterrows()
                        if str(r.get("NOMENCLATURA", "")).strip()
                    }

                params_txt = {
                    "d": d_param,
                    "a": a_param,
                    "linea": linea,
                    "util": util,
                    "notch": notch,
                    "tolerancia": tolerancia,
                    "tipo_tela": tipo_tela,
                    "nomenclaturas": nomenclaturas,
                }

                txt = generar_txt_accumark(sel_todas, df_lotes_det, params_txt=params_txt)
                st.text_area("Vista previa TXT", txt, height=420)

                st.download_button(
                    "⬇️ Descargar TXT Accumark",
                    data=txt.encode("utf-8"),
                    file_name="markers_accumark_post_process.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        # ── Reinyección ──────────────────────────────────────────────────────
        with tab_reinj:
            sel_a = st.session_state.get("_pp_sel_a", pd.DataFrame())
            sel_b = st.session_state.get("_pp_sel_b", pd.DataFrame())
            sel_todas = pd.concat([sel_a, sel_b], ignore_index=True)
            df_reinj = preparar_reinyeccion(sel_todas)
            if df_reinj.empty:
                st.info("Selecciona propuestas para preparar la tabla de reinyección/auditoría.")
            else:
                st.dataframe(df_reinj, use_container_width=True, hide_index=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_reinj.to_excel(writer, index=False, sheet_name="PROPUESTAS")
                st.download_button(
                    "⬇️ Descargar propuestas aprobadas",
                    data=buffer.getvalue(),
                    file_name="propuestas_post_process.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
