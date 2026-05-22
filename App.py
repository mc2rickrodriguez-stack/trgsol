"""
app.py — SOLVER ST350 (Versión unificada)
Todo en una sola página con secciones expandibles.
"""

import io, math, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from solver_core    import cargar_datos, validar, ejecutar_solver
from lotes_core     import preparar_pool, crear_lotes
from asignacion_core import cargar_pedidos, asignar_pedidos

st.set_page_config(
    page_title="SOLVER ST350",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title{font-size:2rem;font-weight:700;color:#1F3864;margin-bottom:.2rem}
  .subtitle{color:#666;font-size:.95rem;margin-bottom:1rem}
  .stProgress>div>div{background-color:#1F3864}
  .section-header{font-size:1.2rem;font-weight:600;color:#1F3864;margin:0}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧵 SOLVER ST350</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Optimización · Lotes · Asignación de Pedidos · Recomendaciones</p>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — Todos los parámetros
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parámetros")

    st.subheader("🧵 Solver")
    peso_oz = st.number_input(
        "Peso tela (oz/yd²)", min_value=1.0, max_value=20.0,
        value=3.77, step=0.01, format="%.2f",
        help="CSM_MARKER = (ANCHO_FINAL/36) × (LENGTH/36) × (PESO/16)"
    )
    n_min_usos = st.number_input(
        "Mínimo usos por marker", min_value=1, max_value=500, value=1, step=1,
        help="El solver asigna ≥ N usos a un marker o no lo usa. Default 1 = sin restricción."
    )
    max_holgura = st.slider("Holgura máxima (%)", 0.0, 20.0, 2.0, 0.5)
    factor_pen  = st.select_slider("Factor penalización",
        options=[100,500,1_000,5_000,10_000,50_000,100_000], value=10_000)
    time_limit  = st.slider("Tiempo límite/color (seg)", 10, 300, 60, 10)
    mip_gap     = st.slider("Gap MIP (%)", 0.01, 5.0, 0.1, 0.01, format="%.2f")

    st.divider()
    st.subheader("🧺 Lotes")
    pct_merma = st.number_input(
        "% Merma Marker (crudo)", min_value=0.0, max_value=50.0,
        value=10.0, step=0.5, format="%.1f",
        help="CSM_MERMA = CSM_MARKER × (1 + merma/100)"
    )
    st.caption(f"Con {pct_merma:.1f}% merma: 4.00 lbs → {4.00*(1+pct_merma/100):.2f} lbs/capa")
    pct_merma_rib = st.number_input(
        "% Merma RIB (crudo)", min_value=0.0, max_value=50.0,
        value=8.0, step=0.5, format="%.1f",
        help="LBS_RIB_MERMA = LBS_RIB × (1 + merma_rib/100)"
    )
    st.caption(f"Con {pct_merma_rib:.1f}% merma RIB: 4.00 lbs → {4.00*(1+pct_merma_rib/100):.2f} lbs/capa")
    st.divider()
    st.caption("**Versión:** 3.0 · Motor: SciPy MILP / HiGHS")

params_solver = {
    "max_holgura_pct": max_holgura / 100.0,
    "factor_pen":      float(factor_pen),
    "time_limit":      float(time_limit),
    "mip_rel_gap":     mip_gap / 100.0,
    "n_min_usos":      int(n_min_usos),
}

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 1 — CARGA DE ARCHIVOS
# ═══════════════════════════════════════════════════════════════
with st.expander("📁 Sección 1 — Carga de Archivos", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        f_mk = st.file_uploader("MARKERS_ST350.xlsx", type=["xlsx"], key="mk",
                                 help="Hojas: MARKER y MODEL")
        if f_mk:
            st.session_state["_b_mk"] = f_mk.read()
            st.session_state["_n_mk"] = f_mk.name
        if "_b_mk" in st.session_state:
            st.success(f"✅ {st.session_state['_n_mk']}")

    with c2:
        f_bl = st.file_uploader("Balance + Capacidad", type=["xlsx"], key="bl",
                                 help="Hojas: BALANCE y CAPACIDAD")
        if f_bl:
            st.session_state["_b_bl"] = f_bl.read()
            st.session_state["_n_bl"] = f_bl.name
        if "_b_bl" in st.session_state:
            st.success(f"✅ {st.session_state['_n_bl']}")

    with c3:
        f_es = st.file_uploader("ESTANDARES_ST350.xlsx", type=["xlsx"], key="es",
                                 help="Hoja: Hoja1")
        if f_es:
            st.session_state["_b_es"] = f_es.read()
            st.session_state["_n_es"] = f_es.name
        if "_b_es" in st.session_state:
            st.success(f"✅ {st.session_state['_n_es']}")

    with c4:
        f_ped = st.file_uploader("BALANCES_PEDIDO.xlsx", type=["xlsx"], key="ped",
                                  help="Hoja: Hoja1 | Columnas: ESTILO BASE, COLOR, PEDIDO|LINEA, PRIORIDAD, PO DATE, TALLA STD, BALANCE")
        if f_ped:
            try:
                df_ped_raw = cargar_pedidos(io.BytesIO(f_ped.read()))
                st.session_state["_df_pedidos"] = df_ped_raw
                st.session_state["_n_ped"]      = f_ped.name
                st.success(f"✅ {f_ped.name} — {len(df_ped_raw):,} pedidos")
            except Exception as e:
                st.error(f"❌ {e}")
        if "_df_pedidos" in st.session_state and "_n_ped" in st.session_state and not f_ped:
            st.info(f"📦 {st.session_state['_n_ped']} en memoria")

_bm = st.session_state.get("_b_mk")
_bb = st.session_state.get("_b_bl")
_be = st.session_state.get("_b_es")
archivos_ok = bool(_bm and _bb and _be)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 2 — CAPACIDAD Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════
if archivos_ok:
    with st.expander("📋 Sección 2 — Capacidad y Validación", expanded=False):

        # Detectar si los archivos cambiaron (usando hash de bytes)
        _hash_files = hash((_bm[:100], _bb[:100], _be[:100], round(peso_oz,2)))
        _hash_prev  = st.session_state.get("_hash_files_prev")
        _datos_raw  = st.session_state.get("_datos_raw")

        if _hash_files != _hash_prev or _datos_raw is None:
            try:
                with st.spinner("Leyendo archivos..."):
                    _datos_raw = cargar_datos(
                        io.BytesIO(_bm), io.BytesIO(_bb), io.BytesIO(_be),
                        peso_oz=peso_oz
                    )
                st.session_state["_datos_raw"]        = _datos_raw
                st.session_state["_hash_files_prev"]  = _hash_files
                st.session_state["_cap_base"]         = _datos_raw["df_cap"].copy()
            except Exception as e:
                st.error(f"❌ {e}"); st.stop()

        _datos_raw = st.session_state["_datos_raw"]
        _cap_base  = st.session_state.get("_cap_base", _datos_raw["df_cap"].copy())

        st.info("✏️ Edita la tabla y presiona **Aplicar cambios** para confirmar.")
        df_cap_ed = st.data_editor(
            _cap_base, use_container_width=True, num_rows="dynamic",
            column_config={
                "GRUPO":          st.column_config.TextColumn("Grupo", width="small"),
                "ANCHO_FINAL":    st.column_config.NumberColumn("Ancho Final", format="%.0f"),
                "ANCHO_CORTABLE": st.column_config.NumberColumn("Ancho Cortable", format="%.0f"),
                "DDGG":           st.column_config.NumberColumn("DDGG", format="%.0f"),
                "CAPACIDAD":      st.column_config.NumberColumn("Capacidad (lbs)", format="%,.0f", min_value=0),
            },
            hide_index=True, key="cap_ed",
        )

        col_ap, col_inf = st.columns([1, 3])
        with col_ap:
            btn_aplicar_cap = st.button("✅ Aplicar cambios de capacidad",
                                        type="primary", use_container_width=True, key="btn_cap")
        with col_inf:
            if st.session_state.get("_datos"):
                st.caption("✅ Capacidad aplicada — puedes ejecutar el solver.")
            else:
                st.caption("⬆️ Edita la tabla y presiona Aplicar para habilitar el solver.")

        if btn_aplicar_cap:
            df_cap_ed["GRUPO"]          = df_cap_ed["GRUPO"].map(lambda s: str(s).strip().upper() if pd.notna(s) else "")
            df_cap_ed["ANCHO_CORTABLE"] = pd.to_numeric(df_cap_ed["ANCHO_CORTABLE"], errors="coerce")
            df_cap_ed["CAPACIDAD"]      = pd.to_numeric(df_cap_ed["CAPACIDAD"], errors="coerce").fillna(0)
            df_cap_ed = df_cap_ed[df_cap_ed["ANCHO_CORTABLE"].notna() & (df_cap_ed["CAPACIDAD"] > 0)].copy()

            datos = _datos_raw.copy()
            datos["df_cap"] = df_cap_ed
            df_bal_activo, df_excluidos, resumen = validar(datos)

            # Limpiar solver anterior si cambia la capacidad
            for k in ["_res_solver","_df_bal_raw","_df_excl","_res_lotes","_res_asig"]:
                st.session_state.pop(k, None)

            st.session_state["_datos"]        = datos
            st.session_state["_cap_base"]     = df_cap_ed.copy()
            st.session_state["_bal_activo"]   = df_bal_activo
            st.session_state["_df_excluidos"] = df_excluidos
            st.session_state["_resumen"]      = resumen
            st.success("✅ Capacidad aplicada.")
            st.rerun()

        # Mostrar validación si ya está aplicada
        if st.session_state.get("_datos"):
            resumen_show = st.session_state.get("_resumen", {})
            df_excl_show = st.session_state.get("_df_excluidos", pd.DataFrame())
            if resumen_show:
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Total",         resumen_show.get("total",0))
                m2.metric("Con cobertura", resumen_show.get("activos",0))
                m3.metric("Excluidos",     resumen_show.get("excluidos",0),
                          delta=f"-{resumen_show.get('excluidos',0)}" if resumen_show.get("excluidos") else None,
                          delta_color="inverse" if resumen_show.get("excluidos") else "off")
                m4.metric("Colores",       resumen_show.get("colores",0))
                m5.metric("Grupos",        ", ".join(resumen_show.get("grupos",[])) or "—")
                if not df_excl_show.empty:
                    with st.expander(f"⚠️ {len(df_excl_show)} excluidos"):
                        st.dataframe(df_excl_show[["ESTILO","COLOR","TALLA","GRUPO","DEMANDA","MOTIVO_EXCL"]],
                                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 3 — EJECUTAR SOLVER
# ═══════════════════════════════════════════════════════════════
datos      = st.session_state.get("_datos")
bal_activo = st.session_state.get("_bal_activo")
resumen    = st.session_state.get("_resumen", {})

if archivos_ok and datos is not None and bal_activo is not None:
    with st.expander("🚀 Sección 3 — Ejecutar Solver", expanded=False):
        col_btn, col_info = st.columns([1,3])
        with col_btn:
            run_solver = st.button("🚀 EJECUTAR SOLVER", type="primary", use_container_width=True)
        with col_info:
            st.caption(
                f"**{resumen.get('colores',0)} colores** · "
                f"Peso: {peso_oz} oz/yd² · Mín usos: {n_min_usos} · "
                f"Holgura: {max_holgura}% · Tiempo: {time_limit}s"
            )

        if run_solver:
            pb = st.progress(0); st_txt = st.empty()
            log_exp = st.expander("📋 Log", expanded=True)
            lines = []
            def cb(grupo, color, total, resueltos, msg):
                pb.progress(int(100*resueltos/total))
                st_txt.caption(f"**{grupo}·{color}** ({resueltos}/{total})")
                lines.append(f"`{grupo}` · **{color}** — {msg}")
                with log_exp:
                    for l in lines[-15:]: st.markdown(l)

            with st.spinner("Optimizando..."):
                res_solver = ejecutar_solver(datos, bal_activo, params_solver, progress_callback=cb)

            pb.progress(100); st_txt.success("✅ Solver finalizado")
            for k in ["_res_solver","_df_bal_raw","_df_excl"]:
                st.session_state.pop(k, None)
            st.session_state["_res_solver"]  = res_solver
            st.session_state["_df_bal_raw"]  = datos["df_balance_raw"]
            st.session_state["_df_excl"]     = st.session_state.get("_df_excluidos", pd.DataFrame())
            # Limpiar lotes y asignacion anteriores
            for k in ["_res_lotes","_res_asig"]: st.session_state.pop(k, None)
            st.rerun()

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 4 — RESULTADOS DEL SOLVER
# ═══════════════════════════════════════════════════════════════
res_solver = st.session_state.get("_res_solver")

if res_solver:
    df_usos  = res_solver["df_usos"]
    df_cumpl = res_solver["df_cumpl"]
    df_ddgg  = res_solver["df_ddgg"]
    df_comp  = res_solver["df_comparacion"]
    df_err   = res_solver["df_errores"]
    df_br    = st.session_state.get("_df_bal_raw", pd.DataFrame())
    df_excl  = st.session_state.get("_df_excl",    pd.DataFrame())

    with st.expander("📊 Sección 4 — Resultados del Solver", expanded=True):
        # KPIs
        fill_p  = df_cumpl["FILL_RATE_%"].mean() if not df_cumpl.empty else 0
        c100    = int((df_cumpl.groupby("COLOR")["FILL_RATE_%"].mean()>=99.9).sum()) if not df_cumpl.empty else 0
        lbs_tot = df_usos["LBS_CONSUMIDAS"].sum() if not df_usos.empty else 0
        if not df_comp.empty:
            lbs_aho = df_comp["DIFER_LBS"].sum()
            pct_m   = lbs_aho/df_comp["LBS_ESTANDAR"].sum()*100 if df_comp["LBS_ESTANDAR"].sum()>0 else 0
        else: lbs_aho = pct_m = 0

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Fill Rate",       f"{fill_p:.1f}%")
        k2.metric("Colores 100%",    f"{c100}/{df_cumpl['COLOR'].nunique() if not df_cumpl.empty else 0}")
        k3.metric("Lbs consumidas",  f"{lbs_tot:,.1f}")
        k4.metric("Usos marker",     f"{df_usos['USOS'].sum():,.0f}" if not df_usos.empty else "0")
        k5.metric("Lbs ahorradas",   f"{lbs_aho:,.1f}",
                  delta=f"{pct_m:+.1f}% vs std",
                  delta_color="normal" if lbs_aho>=0 else "inverse")
        k6.metric("Errores",         len(df_err))

        t1,t2,t3,t4,t5,t6 = st.tabs([
            "📌 Usos Marker","✅ Cumplimiento","⚖️ DDGG",
            "🔬 vs Estándar","📈 Gráficas","⚠️ Errores"
        ])

        with t1:
            if not df_usos.empty:
                c1,c2,c3 = st.columns(3)
                gf=c1.multiselect("Grupo",sorted(df_usos["GRUPO"].unique()),default=list(df_usos["GRUPO"].unique()),key="u_g")
                cf=c2.multiselect("Color",sorted(df_usos["COLOR"].unique()),default=[],key="u_c")
                af=c3.multiselect("Ancho",sorted(df_usos["WIDTH_CORTABLE"].unique()),default=[],key="u_a")
                d=df_usos[df_usos["GRUPO"].isin(gf)]
                if cf: d=d[d["COLOR"].isin(cf)]
                if af: d=d[d["WIDTH_CORTABLE"].isin(af)]
                st.dataframe(d.style.format({"CSM_MARKER":"{:.4f}","LBS_CONSUMIDAS":"{:,.2f}"}),
                             use_container_width=True,hide_index=True)
                st.caption(f"{len(d):,} filas | {d['USOS'].sum():,} usos | {d['LBS_CONSUMIDAS'].sum():,.1f} lbs")

        with t2:
            if not df_cumpl.empty:
                c1,c2,c3=st.columns(3)
                gf2=c1.multiselect("Grupo",sorted(df_cumpl["GRUPO"].unique()),default=list(df_cumpl["GRUPO"].unique()),key="c_g")
                cf2=c2.multiselect("Color",sorted(df_cumpl["COLOR"].unique()),default=[],key="c_c")
                fr_min=c3.slider("Fill mín %",0,100,0,key="c_fr")
                d=df_cumpl[df_cumpl["GRUPO"].isin(gf2)&(df_cumpl["FILL_RATE_%"]>=fr_min)]
                if cf2: d=d[d["COLOR"].isin(cf2)]
                def cfill(v):
                    if v>=99: return "background-color:#C6EFCE;color:#276221"
                    if v>=80: return "background-color:#FFEB9C;color:#9C6500"
                    return "background-color:#FFC7CE;color:#9C0006"
                st.dataframe(d.style.map(cfill,subset=["FILL_RATE_%"]).format({"FILL_RATE_%":"{:.2f}"}),
                             use_container_width=True,hide_index=True)

        with t3:
            if not df_ddgg.empty:
                def cu(v):
                    if v>95: return "background-color:#FFC7CE;color:#9C0006"
                    if v>75: return "background-color:#FFEB9C;color:#9C6500"
                    return "background-color:#C6EFCE;color:#276221"
                st.dataframe(df_ddgg.style.map(cu,subset=["UTILIZACION_%"]).format(
                    {"CAPACIDAD_TOTAL":"{:,.0f}","CONSUMIDO":"{:,.0f}",
                     "DIFERENCIA":"{:,.0f}","UTILIZACION_%":"{:.2f}"}),
                    use_container_width=True,hide_index=True)

        with t4:
            if not df_comp.empty:
                c1,c2,c3,c4=st.columns(4)
                gfc=c1.multiselect("Grupo",sorted(df_comp["GRUPO"].unique()),default=list(df_comp["GRUPO"].unique()),key="cp_g")
                cfc=c2.multiselect("Color",sorted(df_comp["COLOR"].unique()),default=[],key="cp_c")
                efc=c3.multiselect("Estilo",sorted(df_comp["ESTILO"].unique()),default=[],key="cp_e")
                tfc=c4.multiselect("Talla",sorted(df_comp["TALLA"].unique()),default=[],key="cp_t")
                d=df_comp[df_comp["GRUPO"].isin(gfc)]
                if cfc: d=d[d["COLOR"].isin(cfc)]
                if efc: d=d[d["ESTILO"].isin(efc)]
                if tfc: d=d[d["TALLA"].isin(tfc)]
                def cd(v):
                    if v>0: return "background-color:#C6EFCE;color:#276221"
                    if v<0: return "background-color:#FFC7CE;color:#9C0006"
                    return ""
                st.dataframe(d.style.map(cd,subset=["DIFER_LBS","DIFER_PCT"]).format({
                    "CSM_MARKER":"{:.4f}","CSM_TALLA":"{:.4f}","ESTANDAR_TALLA":"{:.4f}",
                    "LBS_PLANEADAS":"{:,.4f}","LBS_ESTANDAR":"{:,.4f}",
                    "DIFER_LBS":"{:+,.4f}","DIFER_PCT":"{:+.2f}%","MK_LEN_PCT":"{:.2f}"}),
                    use_container_width=True,hide_index=True)
                ts=d["LBS_ESTANDAR"].sum(); tp=d["LBS_PLANEADAS"].sum(); td=d["DIFER_LBS"].sum()
                st.caption(f"Std: {ts:,.2f} | Plan: {tp:,.2f} | Dif: {td:+,.2f} ({td/ts*100:+.2f}%)" if ts>0 else "")

        with t5:
            if not df_cumpl.empty:
                g1,g2=st.columns(2)
                with g1:
                    fr_c=df_cumpl.groupby(["COLOR","GRUPO"])["FILL_RATE_%"].mean().round(2).reset_index()
                    fr_c["L"]=fr_c["GRUPO"]+" · "+fr_c["COLOR"]
                    fr_c=fr_c.sort_values("FILL_RATE_%")
                    f1=px.bar(fr_c,x="FILL_RATE_%",y="L",orientation="h",color="FILL_RATE_%",
                              color_continuous_scale=["#e74c3c","#f39c12","#2ecc71"],range_color=[0,100],
                              title="Fill Rate por Color",labels={"FILL_RATE_%":"Fill Rate (%)","L":""})
                    f1.add_vline(x=100,line_dash="dash",line_color="gray",opacity=0.6)
                    f1.update_layout(height=max(350,len(fr_c)*22),coloraxis_showscale=False,margin=dict(l=0,r=10,t=40,b=20))
                    st.plotly_chart(f1,use_container_width=True)
                with g2:
                    if not df_ddgg.empty:
                        dd=df_ddgg.copy(); dd["L"]=dd["GRUPO"]+" · "+dd["ANCHO_CORTABLE"].astype(int).astype(str)+"\""
                        f2=px.bar(dd,x="UTILIZACION_%",y="L",orientation="h",color="UTILIZACION_%",
                                  color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"],range_color=[0,100],
                                  title="Utilización DDGG",text="UTILIZACION_%")
                        f2.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
                        f2.add_vline(x=100,line_dash="dash",line_color="gray",opacity=0.6)
                        f2.update_layout(height=350,coloraxis_showscale=False,margin=dict(l=0,r=40,t=40,b=20))
                        st.plotly_chart(f2,use_container_width=True)
                if not df_comp.empty:
                    g3,g4=st.columns(2)
                    with g3:
                        ac=df_comp.groupby("COLOR").agg(D=("DIFER_LBS","sum"),E=("LBS_ESTANDAR","sum")).reset_index()
                        ac["P"]=(ac["D"]/ac["E"]*100).round(2); ac=ac.sort_values("P")
                        f3=px.bar(ac,x="COLOR",y="P",color="P",
                                  color_continuous_scale=["#e74c3c","#f5f5f5","#2ecc71"],range_color=[-15,15],
                                  title="% Ahorro por Color",text="P")
                        f3.update_traces(texttemplate="%{text:+.1f}%",textposition="outside")
                        f3.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.6)
                        f3.update_layout(height=380,coloraxis_showscale=False,margin=dict(l=20,r=20,t=40,b=80))
                        st.plotly_chart(f3,use_container_width=True)
                    with g4:
                        ae=df_comp.groupby("ESTILO").agg(D=("DIFER_LBS","sum"),E=("LBS_ESTANDAR","sum")).reset_index()
                        ae["P"]=(ae["D"]/ae["E"]*100).round(2); ae=ae.sort_values("P")
                        f4=px.bar(ae,x="P",y="ESTILO",orientation="h",color="P",
                                  color_continuous_scale=["#e74c3c","#f5f5f5","#2ecc71"],range_color=[-15,15],
                                  title="% Ahorro por Estilo",text="P")
                        f4.update_traces(texttemplate="%{text:+.1f}%",textposition="outside")
                        f4.add_vline(x=0,line_dash="dash",line_color="gray",opacity=0.6)
                        f4.update_layout(height=380,coloraxis_showscale=False,margin=dict(l=20,r=60,t=40,b=20))
                        st.plotly_chart(f4,use_container_width=True)
                    # Tabla estilo
                    st.markdown("**Resumen por Estilo:**")
                    re=df_comp.groupby("ESTILO").agg(
                        LBS_ESTANDAR=("LBS_ESTANDAR","sum"),LBS_PLANEADAS=("LBS_PLANEADAS","sum"),
                        DIFER_LBS=("DIFER_LBS","sum")).reset_index()
                    re["DIFER_%"]=(re["DIFER_LBS"]/re["LBS_ESTANDAR"]*100).round(2)
                    re=re.sort_values("DIFER_%",ascending=False)
                    tot=pd.DataFrame([{"ESTILO":"TOTAL","LBS_ESTANDAR":re["LBS_ESTANDAR"].sum(),
                        "LBS_PLANEADAS":re["LBS_PLANEADAS"].sum(),"DIFER_LBS":re["DIFER_LBS"].sum(),
                        "DIFER_%":re["DIFER_LBS"].sum()/re["LBS_ESTANDAR"].sum()*100 if re["LBS_ESTANDAR"].sum()>0 else 0}])
                    re=pd.concat([re,tot],ignore_index=True)
                    def ced(v):
                        if isinstance(v,str): return "font-weight:bold"
                        if v>0: return "background-color:#C6EFCE;color:#276221"
                        if v<0: return "background-color:#FFC7CE;color:#9C0006"
                        return ""
                    st.dataframe(re.style.map(ced,subset=["DIFER_LBS","DIFER_%"]).format(
                        {"LBS_ESTANDAR":"{:,.2f}","LBS_PLANEADAS":"{:,.2f}",
                         "DIFER_LBS":"{:+,.2f}","DIFER_%":"{:+.2f}%"}),
                        use_container_width=True,hide_index=True)

        with t6:
            ce,cx=st.columns(2)
            with ce:
                st.markdown("**❌ Errores solver:**")
                if df_err.empty: st.success("Sin errores.")
                else: st.dataframe(df_err,use_container_width=True,hide_index=True)
            with cx:
                st.markdown("**⚠️ Excluidos:**")
                if df_excl.empty: st.success("Sin excluidos.")
                else: st.dataframe(df_excl[["ESTILO","COLOR","TALLA","GRUPO","DEMANDA","MOTIVO_EXCL"]],
                                   use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 5 — CREAR LOTES
# ═══════════════════════════════════════════════════════════════
if res_solver:
    with st.expander("🧺 Sección 5 — Crear Lotes", expanded=False):

        # Pool: solo recalcular si cambió el solver o la merma
        _pool_hash = hash((id(res_solver), round(pct_merma, 2), round(pct_merma_rib, 2)))
        if st.session_state.get("_pool_hash") != _pool_hash or st.session_state.get("_pool") is None:
            pool_pool, det_pool = preparar_pool(
                res_solver["df_usos"], res_solver["df_comparacion"],
                pct_merma=pct_merma, pct_merma_rib=pct_merma_rib
            )
            st.session_state["_pool"]      = pool_pool
            st.session_state["_det"]       = det_pool
            st.session_state["_pool_hash"] = _pool_hash
        else:
            pool_pool = st.session_state["_pool"]
            det_pool  = st.session_state["_det"]

        grupos_disp = sorted(pool_pool["GRUPO"].unique())
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Markers disponibles", pool_pool["MARKER_NAME"].nunique())
        p2.metric("Colores",  pool_pool["COLOR"].nunique())
        p3.metric("Anchos",   str(sorted(pool_pool["ANCHO_CORTABLE"].unique())))
        p4.metric("Grupos",   ", ".join(grupos_disp))

        # ── Alerta RIB ──────────────────────────────────────────
        if "CSM_RIB_TALLA" in det_pool.columns:
            _sin_rib = det_pool[det_pool["CSM_RIB_TALLA"].isna() | (det_pool["CSM_RIB_TALLA"] == 0)]
            _estilos_sin_rib = _sin_rib[["ESTILO","TALLA"]].drop_duplicates()
            if not _estilos_sin_rib.empty:
                with st.expander(f"⚠️ {len(_estilos_sin_rib)} combinaciones Estilo/Talla con RIB = 0 o sin dato", expanded=False):
                    st.dataframe(_estilos_sin_rib, use_container_width=True, hide_index=True)
                    st.caption("Si el estilo realmente no lleva RIB, el valor 0 es correcto y no afecta el cálculo.")
            else:
                st.success("✅ Todos los estilos/tallas tienen consumo RIB definido.")
        else:
            st.warning("⚠️ La columna Lbs/Doc_RIB no fue encontrada en ESTANDARES. Los cálculos RIB no estarán disponibles.")

        st.divider()
        st.info("✏️ Edita la configuración y presiona **Guardar configuración** antes de crear los lotes.")

        # Usar config guardada como base (o default si no hay)
        _cfg_base = st.session_state.get("_cfg_guardada",
            pd.DataFrame([{"CATEGORIA":g,"TAMAÑO_LOTE":2200,"LBS_MIN":2100,
                "LBS_MAX":2200,"MAX_EST_TALLA":5,"CAPAS_MIN_MARKER":10,"MAX_LOTES":60}
                for g in grupos_disp])
        )
        df_cfg_ed = st.data_editor(
            _cfg_base, use_container_width=True, num_rows="dynamic",
            column_config={
                "CATEGORIA":        st.column_config.TextColumn("Categoría"),
                "TAMAÑO_LOTE":      st.column_config.NumberColumn("Tamaño Lote (lbs)", min_value=1, format="%d"),
                "LBS_MIN":          st.column_config.NumberColumn("Lbs Mínimas",  min_value=0, format="%d"),
                "LBS_MAX":          st.column_config.NumberColumn("Lbs Máximas",  min_value=1, format="%d"),
                "MAX_EST_TALLA":    st.column_config.NumberColumn("Máx Estilos-Talla", min_value=1, format="%d"),
                "CAPAS_MIN_MARKER": st.column_config.NumberColumn("Capas Mín/Marker",  min_value=1, format="%d"),
                "MAX_LOTES":        st.column_config.NumberColumn("Máx Lotes",    min_value=1, format="%d"),
            },
            hide_index=True, key="cfg_lotes",
        )

        # Botón guardar config (separado del botón crear)
        col_sv, col_sv_inf = st.columns([1, 3])
        with col_sv:
            btn_guardar_cfg = st.button("💾 Guardar configuración",
                                        use_container_width=True, key="btn_save_cfg")
        with col_sv_inf:
            cfg_guardada = st.session_state.get("_cfg_guardada")
            if cfg_guardada is not None:
                tp_g = cfg_guardada["TAMAÑO_LOTE"].mul(cfg_guardada["MAX_LOTES"]).sum() if "LBS_PLAN" not in cfg_guardada.columns else cfg_guardada["LBS_PLAN"].sum()
                st.caption(f"✅ Config guardada — {len(cfg_guardada)} categorías · Plan: {tp_g:,.0f} lbs")
            else:
                st.caption("⬆️ Edita y guarda la configuración antes de crear lotes.")

        if btn_guardar_cfg:
            df_sv = df_cfg_ed.copy()
            for col in ["TAMAÑO_LOTE","LBS_MIN","LBS_MAX","MAX_EST_TALLA","CAPAS_MIN_MARKER","MAX_LOTES"]:
                df_sv[col] = pd.to_numeric(df_sv[col], errors="coerce").fillna(0)
            df_sv["CATEGORIA"] = df_sv["CATEGORIA"].astype(str).str.strip().str.upper()
            df_sv["LBS_PLAN"]  = df_sv["TAMAÑO_LOTE"] * df_sv["MAX_LOTES"]
            st.session_state["_cfg_guardada"] = df_sv
            st.success("✅ Configuración guardada.")
            st.rerun()

        # Vista previa con LBS_PLAN de la config guardada
        cfg_para_lotes = st.session_state.get("_cfg_guardada")
        if cfg_para_lotes is not None:
            st.markdown("**Config confirmada (LBS_PLAN calculado):**")
            st.dataframe(cfg_para_lotes.style.format({
                "TAMAÑO_LOTE":"{:,.0f}","LBS_MIN":"{:,.0f}",
                "LBS_MAX":"{:,.0f}","LBS_PLAN":"{:,.0f}"}),
                use_container_width=True, hide_index=True)

        st.divider()
        lbl,linf=st.columns([1,3])
        with lbl:
            run_lotes=st.button("🧺 CREAR LOTES", type="primary",
                                use_container_width=True,
                                disabled=cfg_para_lotes is None)
        with linf:
            if cfg_para_lotes is not None:
                pool_lbs_mk  = (pool_pool["CAPAS_DISP"]*pool_pool["CSM_MERMA"]).sum()
                pool_lbs_rib = (pool_pool["CAPAS_DISP"]*pool_pool.get("CSM_RIB_MERMA", 0)).sum() if "CSM_RIB_MERMA" in pool_pool.columns else 0
                st.caption(f"Pool marker: {pool_lbs_mk:,.0f} lbs · Pool RIB: {pool_lbs_rib:,.0f} lbs · Total: {pool_lbs_mk+pool_lbs_rib:,.0f} lbs · Merma marker: {pct_merma:.1f}% · Merma RIB: {pct_merma_rib:.1f}%")
            else:
                st.caption("⚠️ Guarda la configuración primero para habilitar este botón.")

        if run_lotes and cfg_para_lotes is not None:
            with st.spinner("Creando lotes..."):
                df_ped = st.session_state.get("_df_pedidos")
                rl = crear_lotes(pool_pool, det_pool, cfg_para_lotes)
                ra = asignar_pedidos(rl["df_lotes_det"], df_ped) if df_ped is not None else None
            for k in ["_res_lotes","_res_asig","_cfg_lotes"]:
                st.session_state.pop(k, None)
            st.session_state["_res_lotes"] = rl
            st.session_state["_res_asig"]  = ra
            st.session_state["_cfg_lotes"] = cfg_para_lotes
            st.success("✅ Lotes creados" + (" y pedidos asignados" if ra else ""))
            st.rerun()

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 6 — RESULTADOS DE LOTES
# ═══════════════════════════════════════════════════════════════
res_lotes = st.session_state.get("_res_lotes")
res_asig  = st.session_state.get("_res_asig")

if res_lotes:
    df_det   = res_lotes["df_lotes_det"]
    df_res   = res_lotes["df_lotes_res"]
    df_incomp= res_lotes["df_incompletos"]

    with st.expander("📦 Sección 6 — Resultados de Lotes", expanded=True):
        n_comp  = int((df_res["COMPLETO"]=="✅ Completo").sum())   if not df_res.empty else 0
        n_incomp= int((df_res["COMPLETO"]=="⚠️ Incompleto").sum()) if not df_res.empty else 0
        lbs_b   = df_res["LBS_TOTAL"].sum()           if not df_res.empty else 0
        lbs_m   = df_res["LBS_TOTAL_MERMA"].sum()     if not df_res.empty else 0
        lbs_rib = df_res["LBS_RIB"].sum()             if not df_res.empty and "LBS_RIB" in df_res.columns else 0
        lbs_rib_m= df_res["LBS_RIB_MERMA"].sum()     if not df_res.empty and "LBS_RIB_MERMA" in df_res.columns else 0
        lbs_comb= df_res["LBS_COMBINADO"].sum()       if not df_res.empty and "LBS_COMBINADO" in df_res.columns else lbs_b
        lbs_comb_m= df_res["LBS_COMBINADO_MERMA"].sum() if not df_res.empty and "LBS_COMBINADO_MERMA" in df_res.columns else lbs_m
        pool_t  = (st.session_state.get("_pool",pd.DataFrame()))
        pool_lbs_disp = (pool_t["CAPAS_DISP"]*pool_t["CSM_MERMA"]).sum() if not pool_t.empty else 0

        k1,k2,k3,k4,k5,k6,k7,k8=st.columns(8)
        k1.metric("Total lotes",       len(df_res))
        k2.metric("Completos",         n_comp)
        k3.metric("Incompletos",       n_incomp, delta=f"-{n_incomp}" if n_incomp else None, delta_color="inverse" if n_incomp else "off")
        k4.metric("Lbs Marker",        f"{lbs_b:,.0f}")
        k5.metric("Lbs Marker crudo",  f"{lbs_m:,.0f}")
        k6.metric("Lbs RIB",           f"{lbs_rib:,.0f}")
        k7.metric("Lbs RIB crudo",     f"{lbs_rib_m:,.0f}")
        k8.metric("Total combinado crudo", f"{lbs_comb_m:,.0f}")

        tl1,tl2,tl3,tl4=st.tabs(["📋 Detalle","📊 Resumen","📈 Gráficas","⚠️ Incompletos"])

        with tl1:
            if not df_det.empty:
                c1,c2,c3,c4=st.columns(4)
                gfd=c1.multiselect("Grupo", sorted(df_det["CATEGORIA"].unique()),default=list(df_det["CATEGORIA"].unique()),key="dl_g")
                cfd=c2.multiselect("Color", sorted(df_det["COLOR"].unique()),default=[],key="dl_c")
                afd=c3.multiselect("Ancho", sorted(df_det["ANCHO_CORTABLE"].unique()),default=[],key="dl_a")
                lfd=c4.multiselect("Lote",  sorted(df_det["LOTE_ID"].unique()),default=[],key="dl_l")
                d=df_det[df_det["CATEGORIA"].isin(gfd)]
                if cfd: d=d[d["COLOR"].isin(cfd)]
                if afd: d=d[d["ANCHO_CORTABLE"].isin(afd)]
                if lfd: d=d[d["LOTE_ID"].isin(lfd)]
                def ccomp(v):
                    if v=="✅ Completo": return "background-color:#C6EFCE;color:#276221"
                    if v=="⚠️ Incompleto": return "background-color:#FFEB9C;color:#9C6500"
                    return ""
                st.dataframe(d.style.map(ccomp,subset=["COMPLETO"]).format(
                    {"CSM_X_CAPA":"{:.4f}","CSM_X_CAPA_MERMA":"{:.4f}",
                     "LBS_TALLA":"{:,.4f}","LBS_TALLA_MERMA":"{:,.4f}",
                     "LBS_RIB_TALLA":"{:,.4f}","LBS_RIB_TALLA_MERMA":"{:,.4f}",
                     "LBS_TOTAL_LOTE":"{:,.2f}","LBS_TOTAL_LOTE_MERMA":"{:,.2f}",
                     "LBS_RIB_LOTE":"{:,.2f}","LBS_RIB_LOTE_MERMA":"{:,.2f}",
                     "LBS_TOTAL_COMBINADO":"{:,.2f}","LBS_TOTAL_COMBINADO_MERMA":"{:,.2f}"}),
                    use_container_width=True, hide_index=True)
                lb=d.drop_duplicates("LOTE_ID")["LBS_TOTAL_LOTE"].sum()
                lm=d.drop_duplicates("LOTE_ID")["LBS_TOTAL_LOTE_MERMA"].sum()
                lr=d.drop_duplicates("LOTE_ID")["LBS_RIB_LOTE"].sum() if "LBS_RIB_LOTE" in d.columns else 0
                lrm=d.drop_duplicates("LOTE_ID")["LBS_RIB_LOTE_MERMA"].sum() if "LBS_RIB_LOTE_MERMA" in d.columns else 0
                lc=d.drop_duplicates("LOTE_ID")["LBS_TOTAL_COMBINADO_MERMA"].sum() if "LBS_TOTAL_COMBINADO_MERMA" in d.columns else lm
                st.caption(f"{len(d):,} filas | {d['LOTE_ID'].nunique()} lotes | {d['QUANTITY'].sum():,} piezas | Marker: {lb:,.0f} lbs ({lm:,.0f} crudo) | RIB: {lr:,.0f} lbs ({lrm:,.0f} crudo) | Combinado crudo: {lc:,.0f} lbs")

        with tl2:
            if not df_res.empty:
                c1,c2=st.columns(2)
                gfr=c1.multiselect("Grupo",sorted(df_res["CATEGORIA"].unique()),default=list(df_res["CATEGORIA"].unique()),key="rl_g")
                cfr=c2.multiselect("Color",sorted(df_res["COLOR"].unique()),default=[],key="rl_c")
                dr=df_res[df_res["CATEGORIA"].isin(gfr)]
                if cfr: dr=dr[dr["COLOR"].isin(cfr)]
                def ccr(v):
                    if v=="✅ Completo": return "background-color:#C6EFCE;color:#276221"
                    if v=="⚠️ Incompleto": return "background-color:#FFEB9C;color:#9C6500"
                    return ""
                st.dataframe(dr.style.map(ccr,subset=["COMPLETO"]).format(
                    {"LBS_TOTAL":"{:,.2f}","LBS_TOTAL_MERMA":"{:,.2f}",
                     "LBS_RIB":"{:,.2f}","LBS_RIB_MERMA":"{:,.2f}",
                     "LBS_COMBINADO":"{:,.2f}","LBS_COMBINADO_MERMA":"{:,.2f}"}),
                    use_container_width=True,hide_index=True)

        with tl3:
            if not df_res.empty:
                gl1,gl2=st.columns(2)
                with gl1:
                    lc=df_res.groupby(["CATEGORIA","COLOR"])["LBS_TOTAL"].sum().reset_index()
                    lc["L"]=lc["CATEGORIA"]+" · "+lc["COLOR"]
                    lc=lc.sort_values("LBS_TOTAL")
                    fig=px.bar(lc,x="LBS_TOTAL",y="L",orientation="h",color="CATEGORIA",
                               title="Lbs asignadas por Color",labels={"LBS_TOTAL":"Lbs","L":""})
                    fig.update_layout(height=max(350,len(lc)*24),margin=dict(l=0,r=20,t=40,b=20))
                    st.plotly_chart(fig,use_container_width=True)
                with gl2:
                    sc=df_res.groupby(["CATEGORIA","COMPLETO"]).size().reset_index(name="N")
                    fig2=px.bar(sc,x="CATEGORIA",y="N",color="COMPLETO",
                                color_discrete_map={"✅ Completo":"#2ecc71","⚠️ Incompleto":"#f39c12"},
                                barmode="group",title="Completos vs Incompletos")
                    fig2.update_layout(height=350,margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig2,use_container_width=True)
                fig3=px.histogram(df_res,x="LBS_TOTAL",color="CATEGORIA",nbins=30,
                                  title="Distribución de Lbs por Lote",labels={"LBS_TOTAL":"Lbs"})
                fig3.update_layout(height=300,margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig3,use_container_width=True)

        with tl4:
            if df_incomp.empty: st.success("🎉 Todos los lotes completos.")
            else:
                st.warning(f"**{len(df_incomp)} lotes incompletos** — revisión manual requerida.")
                st.dataframe(df_incomp,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 7 — ASIGNACIÓN DE PEDIDOS
# ═══════════════════════════════════════════════════════════════
if res_asig:
    df_asig    = res_asig["df_asignacion"]
    df_res_ped = res_asig["df_resumen_ped"]
    df_sin_ped = res_asig["df_sin_pedido"]
    df_prior   = res_asig["df_res_prioridad"]

    with st.expander("📦 Sección 7 — Asignación de Pedidos", expanded=True):
        nc_ped = int((df_res_ped["ESTADO_PEDIDO"]=="✅ Completo").sum())   if not df_res_ped.empty else 0
        np_ped = int((df_res_ped["ESTADO_PEDIDO"]=="⚠️ Parcial").sum())    if not df_res_ped.empty else 0
        ns_ped = int((df_res_ped["ESTADO_PEDIDO"]=="❌ Sin producción").sum()) if not df_res_ped.empty else 0
        fr_g   = df_res_ped["FILL_RATE_%"].mean() if not df_res_ped.empty else 0

        a1,a2,a3,a4,a5=st.columns(5)
        a1.metric("Completos",      nc_ped)
        a2.metric("Parciales",      np_ped)
        a3.metric("Sin producción", ns_ped)
        a4.metric("Fill Rate prom", f"{fr_g:.1f}%")
        a5.metric("Sin pedido",     f"{df_sin_ped['SIN_PEDIDO'].sum():,}" if not df_sin_ped.empty else "0")

        ta1,ta2,ta3,ta4=st.tabs(["📋 Detalle Asignación","📊 Resumen Pedidos","📊 Prioridad","🔴 Sin Pedido"])

        with ta1:
            if not df_asig.empty:
                c1,c2,c3=st.columns(3)
                gfa=c1.multiselect("Grupo",sorted(df_asig["CATEGORIA"].unique()),default=list(df_asig["CATEGORIA"].unique()),key="as_g")
                cfa=c2.multiselect("Color",sorted(df_asig["COLOR"].unique()),default=[],key="as_c")
                pfa=c3.multiselect("Prioridad",sorted(df_asig["PRIORIDAD"].unique()),default=[],key="as_p")
                da=df_asig[df_asig["CATEGORIA"].isin(gfa)]
                if cfa: da=da[da["COLOR"].isin(cfa)]
                if pfa: da=da[da["PRIORIDAD"].isin(pfa)]
                def ces(v):
                    if v=="✅ Completo":      return "background-color:#C6EFCE;color:#276221"
                    if v=="⚠️ Parcial":       return "background-color:#FFEB9C;color:#9C6500"
                    if v=="❌ Sin producción": return "background-color:#FFC7CE;color:#9C0006"
                    return ""
                st.dataframe(da.style.map(ces,subset=["ESTADO_LINEA"]),use_container_width=True,hide_index=True)
                st.caption(f"{len(da):,} filas | {da['ASIGNADO'].sum():,} unidades")

        with ta2:
            if not df_res_ped.empty:
                c1,c2=st.columns(2)
                prf=c1.multiselect("Prioridad",sorted(df_res_ped["PRIORIDAD"].unique()),default=list(df_res_ped["PRIORIDAD"].unique()),key="rp_p")
                esf=c2.multiselect("Estado",sorted(df_res_ped["ESTADO_PEDIDO"].unique()),default=[],key="rp_e")
                dr=df_res_ped[df_res_ped["PRIORIDAD"].isin(prf)]
                if esf: dr=dr[dr["ESTADO_PEDIDO"].isin(esf)]
                st.dataframe(dr.style.map(ces,subset=["ESTADO_PEDIDO"]).format(
                    {"DEMANDA_PEDIDO":"{:,.0f}","TOTAL_ASIGNADO":"{:,.0f}",
                     "PENDIENTE_FINAL":"{:,.0f}","FILL_RATE_%":"{:.2f}%"}),
                    use_container_width=True,hide_index=True)

        with ta3:
            if not df_prior.empty:
                st.dataframe(df_prior.style.format(
                    {"DEMANDA_TOTAL":"{:,.0f}","ASIGNADO_TOTAL":"{:,.0f}",
                     "PENDIENTE_TOTAL":"{:,.0f}","FILL_RATE_%":"{:.2f}%"}),
                    use_container_width=True,hide_index=True)
                fp1,fp2=st.columns(2)
                with fp1:
                    dm=df_prior.melt(id_vars="PRIORIDAD",value_vars=["DEMANDA_TOTAL","ASIGNADO_TOTAL"],
                                     var_name="TIPO",value_name="UNIDADES")
                    fig=px.bar(dm,x="PRIORIDAD",y="UNIDADES",color="TIPO",barmode="group",
                               color_discrete_map={"DEMANDA_TOTAL":"#3498db","ASIGNADO_TOTAL":"#2ecc71"},
                               title="Demanda vs Asignado",text="UNIDADES")
                    fig.update_traces(texttemplate="%{text:,.0f}",textposition="outside")
                    fig.update_layout(height=350,margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig,use_container_width=True)
                with fp2:
                    fig2=px.bar(df_prior,x="PRIORIDAD",y="FILL_RATE_%",color="FILL_RATE_%",
                                color_continuous_scale=["#e74c3c","#f39c12","#2ecc71"],range_color=[0,100],
                                title="Fill Rate por Prioridad",text="FILL_RATE_%")
                    fig2.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
                    fig2.add_hline(y=100,line_dash="dash",line_color="gray",opacity=0.5)
                    fig2.update_layout(height=350,coloraxis_showscale=False,margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig2,use_container_width=True)

        with ta4:
            if df_sin_ped.empty: st.success("✅ Sin producción sin pedido asignado.")
            else:
                st.warning(f"🔴 {len(df_sin_ped)} registros de producción sin pedido")
                st.dataframe(df_sin_ped,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 8 — RECOMENDACIONES IA
# ═══════════════════════════════════════════════════════════════
if res_solver or res_lotes:
    with st.expander("💡 Sección 8 — Recomendaciones & IA", expanded=False):

        # ── 5-7: Optimización de Lotes ────────────────────────
        if res_lotes:
            st.subheader("📦 5-7 · Optimización de Lotes")
            df_res_l = res_lotes["df_lotes_res"]
            if not df_res_l.empty:
                col_sel=st.selectbox("Filtrar color",["(Todos)"]+sorted(df_res_l["COLOR"].unique()),key="rec_c")
                grps = df_res_l.groupby(["COLOR","ANCHO_CORTABLE"])
                for (color,ancho), grp in grps:
                    if col_sel!="(Todos)" and color!=col_sel: continue
                    lbs_v=grp["LBS_TOTAL"].values; total=lbs_v.sum(); media=np.mean(lbs_v)
                    cands=sorted(set([round(v/50)*50 for v in
                        [np.percentile(lbs_v,p) for p in [25,50,75]]+[media] if v>0]))
                    top3=sorted([{"tam":t,"n":int(total//t),"res":round(total-int(total//t)*t,1),
                        "pct":round((total-int(total//t)*t)/total*100,1)} for t in cands if t>0],
                        key=lambda x:x["pct"])[:3]
                    n_frag=int((lbs_v<media*0.6).sum()); pct_f=n_frag/len(lbs_v)*100
                    with st.expander(f"🎨 {color} | {ancho}\" — {len(grp)} lotes · {total:,.0f} lbs",expanded=False):
                        cc1,cc2,cc3=st.columns(3)
                        cc1.metric("Lotes actuales",len(grp))
                        cc2.metric("Media/lote",f"{media:,.0f} lbs")
                        cc3.metric("Fragmentados",f"{n_frag} ({pct_f:.0f}%)",
                                   delta="⚠️ Alto" if pct_f>=20 else None)
                        if top3:
                            df_top=pd.DataFrame([{"Tamaño (lbs)":t["tam"],"N° lotes":t["n"],
                                "Residuo (lbs)":t["res"],"Residuo (%)":t["pct"]} for t in top3])
                            st.dataframe(df_top.style.format(
                                {"Tamaño (lbs)":"{:,.0f}","N° lotes":"{:,.0f}",
                                 "Residuo (lbs)":"{:,.1f}","Residuo (%)":"{:.1f}%"}),
                                use_container_width=True,hide_index=True)
                            st.success(f"✅ Tamaño óptimo: **{top3[0]['tam']:,.0f} lbs** — {top3[0]['n']} lotes · {top3[0]['pct']:.1f}% residuo")
                            lbs_min_sug=round(np.percentile(lbs_v,10)/10)*10
                            st.info(f"🔧 LBS_MIN sugerido: **{lbs_min_sug:,.0f} lbs** (percentil 10)")
                            fig=px.histogram(x=lbs_v,nbins=15,title=f"{color} | {ancho}\"",
                                             labels={"x":"Lbs/lote"},color_discrete_sequence=["#4A90D9"])
                            fig.add_vline(x=media,line_dash="dash",line_color="orange",annotation_text=f"Media {media:,.0f}")
                            fig.add_vline(x=top3[0]["tam"],line_dash="dot",line_color="green",annotation_text=f"Óptimo {top3[0]['tam']:,.0f}")
                            fig.update_layout(height=220,margin=dict(t=40,b=20))
                            st.plotly_chart(fig,use_container_width=True)

        st.divider()

        # ── 8: Alertas de riesgo ──────────────────────────────
        if res_asig and not res_asig["df_resumen_ped"].empty:
            st.subheader("🚨 8 · Alertas de Pedidos en Riesgo")
            rp=res_asig["df_resumen_ped"]
            umbrales={"AHEAD 2":80.0,"AHEAD 3":60.0}
            alertas=[]
            for prio,umb in umbrales.items():
                sub=rp[rp["PRIORIDAD"].str.upper().str.contains(prio.replace(" ",""),na=False)]
                for _,row in sub[sub["FILL_RATE_%"]<umb].iterrows():
                    alertas.append({"PRIORIDAD":prio,"PEDIDO":row["PEDIDO_LINEA"],
                        "ESTILO":row["ESTILO"],"TALLA":row["TALLA"],"COLOR":row["COLOR"],
                        "FILL_RATE_%":row["FILL_RATE_%"],"UMBRAL_%":umb,
                        "DEFICIT_%":round(umb-row["FILL_RATE_%"],2)})
            if not alertas:
                st.success("✅ Ningún pedido por debajo del umbral crítico.")
            else:
                df_al=pd.DataFrame(alertas).sort_values("DEFICIT_%",ascending=False)
                st.error(f"⚠️ {len(df_al)} pedido(s) en riesgo")
                def cal(row):
                    if row["DEFICIT_%"]>30: return ["background-color:#ffcccc"]*len(row)
                    if row["DEFICIT_%"]>15: return ["background-color:#ffe5b4"]*len(row)
                    return ["background-color:#fffde7"]*len(row)
                st.dataframe(df_al.style.apply(cal,axis=1).format(
                    {"FILL_RATE_%":"{:.1f}%","UMBRAL_%":"{:.0f}%","DEFICIT_%":"{:.1f}%"}),
                    use_container_width=True,hide_index=True)

        st.divider()

        # ── 10: Curva de cobertura ────────────────────────────
        if res_lotes and not res_lotes["df_lotes_res"].empty:
            st.subheader("📈 10 · Curva de Cobertura Acumulada")
            dfr_l=res_lotes["df_lotes_res"]
            tc1,tc2=st.tabs(["🌐 Global","🎨 Por Color"])
            with tc1:
                ds=dfr_l.sort_values("LBS_TOTAL",ascending=False).reset_index(drop=True)
                tot_l=ds["LBS_TOTAL"].sum()
                ds["COB"]=ds["LBS_TOTAL"].cumsum()/tot_l*100
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=ds.index+1,y=ds["COB"],mode="lines+markers",
                    line=dict(color="#2980b9",width=2),marker=dict(size=4),name="Cobertura"))
                fig.add_hline(y=80,line_dash="dash",line_color="orange",annotation_text="80%")
                fig.add_hline(y=95,line_dash="dash",line_color="green",annotation_text="95%")
                fig.update_layout(title="Cobertura global",xaxis_title="N° lotes",
                                  yaxis_title="Cobertura (%)",height=320)
                st.plotly_chart(fig,use_container_width=True)
                l80=int((ds["COB"]<=80).sum())+1
                st.caption(f"📌 Los primeros **{l80}** lotes cubren el 80% de la producción.")
            with tc2:
                fig2=go.Figure()
                for color in sorted(dfr_l["COLOR"].dropna().unique()):
                    sc=dfr_l[dfr_l["COLOR"]==color].sort_values("LBS_TOTAL",ascending=False).reset_index(drop=True)
                    tt=sc["LBS_TOTAL"].sum()
                    if tt==0: continue
                    sc["C"]=sc["LBS_TOTAL"].cumsum()/tt*100
                    fig2.add_trace(go.Scatter(x=sc.index+1,y=sc["C"],mode="lines",name=color))
                fig2.add_hline(y=80,line_dash="dash",line_color="gray",annotation_text="80%")
                fig2.update_layout(title="Cobertura por color",xaxis_title="N° lotes",
                                   yaxis_title="Cobertura (%)",height=380)
                st.plotly_chart(fig2,use_container_width=True)

        st.divider()

        # ── 11-13: IA con Claude ──────────────────────────────
        st.subheader("🤖 11-13 · Inteligencia Artificial con Claude")

        def _call_claude(messages, system=""):
            import requests
            payload={"model":"claude-sonnet-4-20250514","max_tokens":1500,"messages":messages}
            if system: payload["system"]=system
            try:
                r=requests.post("https://api.anthropic.com/v1/messages",
                    headers={"Content-Type":"application/json"},json=payload,timeout=60)
                if r.status_code!=200: return f"❌ Error {r.status_code}: {r.text[:200]}"
                return "\n".join(b["text"] for b in r.json().get("content",[]) if b.get("type")=="text")
            except Exception as e: return f"❌ {e}"

        def _ctx():
            parts=[]
            if res_solver:
                for k,lbl in [("df_cumpl","CUMPLIMIENTO"),("df_ddgg","BALANCE DDGG"),("df_comparacion","VS ESTÁNDAR")]:
                    df=res_solver.get(k)
                    if isinstance(df,pd.DataFrame) and not df.empty:
                        parts.append(f"\n=== {lbl} ===\n{df.to_string(index=False,max_rows=25)}")
            if res_lotes:
                dl=res_lotes.get("df_lotes_res")
                if isinstance(dl,pd.DataFrame) and not dl.empty:
                    parts.append(f"\n=== LOTES ({len(dl)}) ===\n{dl.describe().to_string()}")
            if res_asig:
                da=res_asig.get("df_resumen_ped")
                if isinstance(da,pd.DataFrame) and not da.empty:
                    parts.append(f"\n=== PEDIDOS ===\n{da.head(20).to_string(index=False)}")
            return "\n".join(parts) or "Sin datos."

        ia1,ia2,ia3=st.tabs(["📝 Análisis Narrativo","🩺 Diagnóstico Colores","💬 Config Natural"])

        with ia1:
            if st.button("🔍 Analizar resultados completos",key="btn_ia1"):
                with st.spinner("Claude analizando..."):
                    st.session_state["_ia1"]=_call_claude(
                        [{"role":"user","content":f"Analiza:\n\n{_ctx()}"}],
                        "Eres experto en planificación textil. Analiza los datos ST350, identifica 3 problemas críticos y da recomendaciones accionables. Responde en español con emojis."
                    )
            if "_ia1" in st.session_state: st.markdown(st.session_state["_ia1"])

        with ia2:
            cum=res_solver.get("df_cumpl") if res_solver else None
            if isinstance(cum,pd.DataFrame) and not cum.empty:
                n_prob=(cum["FILL_RATE_%"]<85).sum()
                st.caption(f"**{n_prob}** colores con fill rate < 85%")
                if st.button("🩺 Diagnosticar colores problemáticos",key="btn_ia2"):
                    with st.spinner("Claude diagnosticando..."):
                        st.session_state["_ia2"]=_call_claude(
                            [{"role":"user","content":f"Diagnostica:\n\n{cum.to_string(index=False,max_rows=50)}"}],
                            "Eres experto en producción textil. Analiza colores con bajo fill rate, identifica causas (falta markers, capacidad, balance) y sugiere acciones. Responde en español."
                        )
            if "_ia2" in st.session_state: st.markdown(st.session_state["_ia2"])

        with ia3:
            st.caption("Describe tu objetivo y Claude sugerirá parámetros para la tabla de lotes.")
            obj=st.text_area("¿Qué quieres lograr?",
                placeholder="Ej: minimizar residuos en negro, lotes de 600 lbs, máximo 5 estilos.",
                height=90,key="ia3_obj")
            if st.button("💡 Generar parámetros",key="btn_ia3",disabled=not obj.strip()):
                cfg_act=st.session_state.get("_cfg_lotes")
                cfg_str=cfg_act.to_string(index=False) if isinstance(cfg_act,pd.DataFrame) else "No disponible"
                with st.spinner("Claude generando sugerencia..."):
                    st.session_state["_ia3"]=_call_claude(
                        [{"role":"user","content":f"Config actual:\n{cfg_str}\n\nObjetivo: {obj}"}],
                        "Eres experto en corte textil ST350. Sugiere: CATEGORIA, TAMAÑO_LOTE, LBS_MIN, LBS_MAX, MAX_EST_TALLA, CAPAS_MIN_MARKER, MAX_LOTES. Muestra tabla markdown y explica brevemente. Responde en español."
                    )
            if "_ia3" in st.session_state: st.markdown(st.session_state["_ia3"])

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 9 — DESCARGAR REPORTE COMPLETO
# ═══════════════════════════════════════════════════════════════
if res_solver or res_lotes:
    with st.expander("📥 Sección 9 — Descargar Reporte Completo", expanded=False):

        def generar_excel_completo():
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="xlsxwriter") as writer:
                wb=writer.book
                hdr=wb.add_format({"bold":True,"bg_color":"#1F3864","font_color":"white",
                    "border":1,"align":"center","valign":"vcenter","text_wrap":True})
                grn=wb.add_format({"bg_color":"#C6EFCE","font_color":"#276221","border":1,"num_format":"0.00"})
                orn=wb.add_format({"bg_color":"#FFEB9C","font_color":"#9C6500","border":1,"num_format":"0.00"})
                red=wb.add_format({"bg_color":"#FFC7CE","font_color":"#9C0006","border":1,"num_format":"0.00"})

                def ws(df,name,widths=None):
                    if df is None or (isinstance(df,pd.DataFrame) and df.empty):
                        df=pd.DataFrame(columns=df.columns if isinstance(df,pd.DataFrame) else [])
                    df.to_excel(writer,sheet_name=name,index=False,startrow=1,header=False)
                    w=writer.sheets[name]
                    for ci,col in enumerate(df.columns): w.write(0,ci,col,hdr)
                    w.set_row(0,22); w.autofilter(0,0,len(df),len(df.columns)-1); w.freeze_panes(1,0)
                    if widths:
                        for ci,ww in enumerate(widths): w.set_column(ci,ci,ww)
                    return w,len(df)

                # Solver sheets
                if res_solver:
                    br=st.session_state.get("_df_bal_raw",pd.DataFrame())
                    excl=st.session_state.get("_df_excl",pd.DataFrame())
                    if not br.empty: ws(br,"BALANCE_ORIGINAL",[16,20,12,20,14,8])
                    ws(res_solver["df_usos"],"USOS_MARKER",[38,22,8,10,10,8,12,14])
                    wc,nc=ws(res_solver["df_cumpl"],"CUMPLIMIENTO",[34,8,22,10,10,10,12])
                    ws(res_solver["df_ddgg"],"BALANCE_DDGG",[8,10,12,14,16,14,12,14])
                    ws(res_solver["df_comparacion"],"COMPARACION_ESTANDAR",[38,22,8,10,10,16,8,12,8,10,12,12,12,12,12,12,12])
                    if nc>0:
                        fci=res_solver["df_cumpl"].columns.get_loc("FILL_RATE_%")
                        wc.conditional_format(1,fci,nc,fci,{"type":"cell","criteria":">=","value":99,"format":grn})
                        wc.conditional_format(1,fci,nc,fci,{"type":"cell","criteria":"between","minimum":80,"maximum":98.99,"format":orn})
                        wc.conditional_format(1,fci,nc,fci,{"type":"cell","criteria":"<","value":80,"format":red})
                    if not excl.empty:
                        ws(excl[["ESTILO","COLOR","TALLA","GRUPO","DEMANDA","MOTIVO_EXCL"]],"EXCLUIDOS",[16,20,10,8,10,35])
                    ws(res_solver["df_errores"],"ERRORES_SOLVER",[8,22,50])

                # Lotes sheets
                if res_lotes:
                    ws(res_lotes["df_lotes_det"],"DETALLE_LOTES",[8,10,12,20,10,35,14,8,14,12,14,12,12,14,14,12,12,14,14,12,12,14,12])
                    ws(res_lotes["df_lotes_res"],"RESUMEN_LOTES",[8,10,12,20,10,10,12,14,14,12,12,14,14,12])
                    ws(res_lotes["df_incompletos"],"LOTES_INCOMPLETOS",[8,10,12,20,10,12,12,12,12,12,12,12,12,40])
                    cfg=st.session_state.get("_cfg_lotes",pd.DataFrame())
                    if not cfg.empty: ws(cfg,"CONFIG_LOTES",[10,14,12,12,14,16,12,14])

                # Asignacion sheets
                if res_asig:
                    if not res_asig["df_asignacion"].empty:
                        ws(res_asig["df_asignacion"],"ASIGNACION_PEDIDOS",[8,10,12,20,10,16,8,12,20,12,14,14,12,14,14,16])
                    if not res_asig["df_resumen_ped"].empty:
                        ws(res_asig["df_resumen_ped"],"RESUMEN_PEDIDOS",[20,16,8,20,12,14,14,12,14,16])
                    if not res_asig["df_res_prioridad"].empty:
                        ws(res_asig["df_res_prioridad"],"RESUMEN_PRIORIDAD",[14,12,14,14,14,12])
                    if not res_asig["df_sin_pedido"].empty:
                        ws(res_asig["df_sin_pedido"],"SIN_PEDIDO",[8,10,12,20,10,16,8,14,12,40])

            buf.seek(0); return buf.read()

        excel_bytes = generar_excel_completo()
        n_hojas = sum([
            7 if res_solver else 0,
            4 if res_lotes else 0,
            4 if res_asig else 0,
        ])
        st.download_button(
            label="📥 Descargar REPORTE_COMPLETO_ST350.xlsx",
            data=excel_bytes,
            file_name="REPORTE_COMPLETO_ST350.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.caption(f"Reporte unificado · {n_hojas} hojas · Solver + Lotes + Asignación de Pedidos")

elif not archivos_ok:
    st.info("👆 Sube los archivos en la **Sección 1** para comenzar.")


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN  — POST PROCESO
# ═══════════════════════════════════════════════════════════════
from seccion_post_process import render_post_process
render_post_process()
