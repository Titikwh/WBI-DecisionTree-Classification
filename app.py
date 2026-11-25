import json

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

import batch_utils as bu

st.set_page_config(
    page_title="Klasifikasi Pengelolaan Sampah",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1rem;
        color: #222;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem;
        color: #888;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.7rem;
    }
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.1rem;
    }
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 0.95rem;
    }
    [data-testid="stMarkdownContainer"] h4 {
        font-size: 0.9rem;
    }
    [data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load model
@st.cache_resource
def load_model():
    return joblib.load("model_decision_tree.joblib")


model = load_model()


# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("3273-kota-bandung-level-kewilayahan.json", "r") as f:
        data = json.load(f)

    wilayah_list = []
    if 'features' in data:
        for feature in data['features']:
            props = feature.get('properties', {})
            wilayah_list.append({
                'nama': props.get('nama_wilayah', ''),
                'id': props.get('id_wilayah', ''),
                'geometry': feature.get('geometry')
            })
    return wilayah_list, data


# Inisialisasi session state
if 'wilayah_status' not in st.session_state:
    st.session_state.wilayah_status = {}

if 'wilayah_params' not in st.session_state:
    st.session_state.wilayah_params = {}

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# Konstanta label
LABEL_CONFIG = {
    "KRITIS": {
        "color": "#E74C3C",
        "icon": "KRITIS",
        "desc": "Pengelolaan sampah dalam kondisi kritis. Diperlukan tindakan segera."
    },
    "WASPADA": {
        "color": "#F39C12",
        "icon": "WASPADA",
        "desc": "Pengelolaan sampah perlu perhatian lebih. Risiko meningkat."
    },
    "AMAN": {
        "color": "#27AE60",
        "icon": "AMAN",
        "desc": "Pengelolaan sampah berjalan dengan baik."
    },
}

# Output model bisa berupa label Inggris (SAFE/ALERT/CRITICAL) atau
# Indonesia (AMAN/WASPADA/KRITIS) -> dipetakan ke label tampilan app
LABEL_MAP = bu.LABEL_MAP

# Nilai maksimum TOTAL_JARAK_BULAT pada data latih (km)
MAX_TOTAL_JARAK_BULAT = bu.MAX_TOTAL_JARAK_BULAT

DEFAULT_PARAMS = {
    "input": 7.0,
    "angkut": 6.0,
    "diolah": 0.0,
    "sisa": 1.0,
    "jarak_bulat": 63.0
}


def derive_features(vol_input, angkut, diolah, sisa, jarak_bulat):
    """Konversi input mentah -> 4 fitur yang dipakai model."""
    return {
        "rasio_angkut": float(angkut / vol_input) if vol_input else 0.0,
        "rasio_diolah": float(diolah / vol_input) if vol_input else 0.0,
        "rasio_sisa": float(sisa / vol_input) if vol_input else 0.0,
        "indeks_jarak": float(jarak_bulat / MAX_TOTAL_JARAK_BULAT),
    }


def balance_sisa(vol_input, angkut, diolah, sisa):
    """Koreksi SISA mengikuti pipeline training (selisih neraca > 3 m3)."""
    sisa_calc = max(vol_input - angkut - diolah, 0.0)
    if abs(sisa - sisa_calc) > 3:
        return sisa_calc, True
    return sisa, False


# ------------------------------------------------------------------ peta (dipakai kedua mode)
def build_map(geojson_data, color_fn, popup_fn, show_boundary, show_labels, legend_rows,
              label_fn=None):
    m = folium.Map(location=[-6.9146, 107.6098], zoom_start=11)
    folium.TileLayer('openstreetmap').add_to(m)

    if show_boundary and geojson_data:
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            wilayah_name = props.get('nama_wilayah', '')
            if not wilayah_name:
                continue

            fill_color = color_fn(wilayah_name)
            opacity = 0.6 if fill_color != "#95A5A6" else 0.3

            folium.GeoJson(
                feature,
                name=wilayah_name,
                style_function=lambda x, color=fill_color, op=opacity: {
                    'fillColor': color,
                    'color': '#2C3E50',
                    'weight': 1.5,
                    'fillOpacity': op,
                },
                tooltip=wilayah_name,
                popup=folium.Popup(popup_fn(wilayah_name), max_width=300)
            ).add_to(m)

            if show_labels:
                try:
                    geom = feature.get('geometry', {})
                    if geom.get('type') == 'Polygon':
                        coords = geom['coordinates'][0]
                        if coords:
                            lats = [c[1] for c in coords]
                            lons = [c[0] for c in coords]
                            text = label_fn(wilayah_name) if label_fn else wilayah_name
                            folium.Marker(
                                location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                                icon=folium.DivIcon(
                                    html=f'<div style="font-size: 10px; font-weight: bold; background: white; padding: 2px 6px; border-radius: 4px; border: 1px solid {fill_color};">{text}</div>'
                                )
                            ).add_to(m)
                except Exception:
                    pass

    legend_html = (
        '<div style="position: fixed; bottom: 30px; right: 30px; background: white; padding: 8px 12px; '
        'border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 11px; z-index: 1000;">'
        '<b>Status</b><br>'
        + "".join(f'<span style="color:{c};">■</span> {t}<br>' for c, t in legend_rows)
        + '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def show_map(m, width, height):
    """st.iframe (Streamlit terbaru) bila tersedia; kalau tidak, pakai folium_static."""
    if hasattr(st, "iframe"):
        st.iframe(m.get_root().render(), width="stretch", height=height)
    else:
        folium_static(m, width=width, height=height)


# Sidebar
with st.sidebar:
    st.markdown("### Pengaturan Peta")
    show_boundary = st.checkbox("Tampilkan Batas Wilayah", value=True)
    show_labels = st.checkbox("Tampilkan Label Wilayah", value=True)
    st.markdown("---")
    st.markdown("**Keterangan Warna**")
    st.markdown("**Merah**: Kritis")
    st.markdown("**Kuning**: Waspada")
    st.markdown("**Hijau**: Aman")
    st.markdown("**Abu-abu**: Belum Diprediksi")

    if st.button("Reset Data Input Manual", use_container_width=True):
        st.session_state.wilayah_status = {}
        st.session_state.wilayah_params = {}
        st.session_state.prediction_result = None
        st.session_state.show_prediction = False
        st.rerun()

# Header
st.markdown(
    """
    <head>
        <meta name="dicoding:email" content="rizki123.rs1@gmail.com">
        <meta name="dicoding:email" content="andinn1410@gmail.com">
        <meta name="dicoding:email" content="nauvaall.qa@gmail.com">
    </head>
    """,
    unsafe_allow_html=True
)
st.title("Klasifikasi Pengelolaan Sampah")
st.markdown("Kota Bandung — Berbasis Waste Burden Index (WBI)")
st.divider()

# Load data
wilayah_list, geojson_data = load_geojson()
wilayah_names = [w['nama'] for w in wilayah_list if w['nama']]

tab_manual, tab_batch = st.tabs(["Input Manual (satu wilayah)", "Upload Batch (Excel)"])


# ==================================================================== TAB 1: INPUT MANUAL
with tab_manual:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        with st.expander("Panduan Pengisian Fitur", expanded=False):
            st.markdown("""
            | Fitur | Keterangan | Satuan |
            |---|---|---|
            | **INPUT** | Volume sampah masuk | m³ |
            | **ANGKUT** | Volume sampah yang berhasil diangkut | m³ |
            | **DIOLAH** | Volume sampah yang berhasil diolah | m³ |
            | **SISA** | Volume sampah tersisa (dikoreksi otomatis bila neraca selisih > 3 m³) | m³ |
            | **TOTAL_JARAK_BULAT** | Jarak bulat rute TPS ke TPA | km |

            Fitur model dihitung otomatis dari input mentah:
            `rasio_angkut = ANGKUT/INPUT`, `rasio_diolah = DIOLAH/INPUT`,
            `rasio_sisa = SISA/INPUT`, `indeks_jarak = TOTAL_JARAK_BULAT/{max_jarak}`.
            """.format(max_jarak=int(MAX_TOTAL_JARAK_BULAT)))

        if wilayah_names:
            selected_wilayah = st.selectbox("Pilih Wilayah:", wilayah_names)

            saved_params = st.session_state.wilayah_params.get(selected_wilayah, DEFAULT_PARAMS.copy())

            st.markdown("### Input Data Mentah")
            col1, col2 = st.columns(2)

            with col1:
                vol_input = st.number_input(
                    "INPUT (m³)", min_value=0.0, step=0.5,
                    value=saved_params["input"], key=f"input_{selected_wilayah}"
                )
                angkut = st.number_input(
                    "ANGKUT (m³)", min_value=0.0, step=0.5,
                    value=saved_params["angkut"], key=f"angkut_{selected_wilayah}"
                )
                diolah = st.number_input(
                    "DIOLAH (m³)", min_value=0.0, step=0.5,
                    value=saved_params["diolah"], key=f"diolah_{selected_wilayah}"
                )

            with col2:
                sisa_input = st.number_input(
                    "SISA (m³)", min_value=0.0, step=0.5,
                    value=saved_params["sisa"], key=f"sisa_{selected_wilayah}"
                )
                jarak_bulat = st.number_input(
                    "TOTAL_JARAK_BULAT (km)", min_value=0.0, step=1.0,
                    value=saved_params["jarak_bulat"], key=f"jarak_{selected_wilayah}"
                )

            sisa_eff, dikoreksi = balance_sisa(vol_input, angkut, diolah, sisa_input)
            sisa_calc = max(vol_input - angkut - diolah, 0.0)

            if vol_input <= 0:
                st.warning("INPUT harus lebih dari 0 agar rasio dapat dihitung.")
            elif dikoreksi:
                st.info(
                    f"Neraca tidak seimbang (selisih {abs(sisa_input - sisa_calc):.2f} m³) — "
                    f"SISA dikoreksi otomatis menjadi {sisa_eff:.2f} m³."
                )

            fitur = derive_features(vol_input, angkut, diolah, sisa_eff, jarak_bulat)

            with st.expander("Fitur Turunan (dihitung otomatis)", expanded=False):
                fitur_values = [
                    ("Rasio Angkut", fitur["rasio_angkut"]),
                    ("Rasio Diolah", fitur["rasio_diolah"]),
                    ("Rasio Sisa", fitur["rasio_sisa"]),
                    ("Indeks Jarak", fitur["indeks_jarak"]),
                ]
                fitur_cols = st.columns(4)
                for fcol, (flabel, fval) in zip(fitur_cols, fitur_values):
                    fcol.metric(flabel, f"{fval:.3f}")
                st.caption(
                    f"Indeks Jarak = TOTAL_JARAK_BULAT / {int(MAX_TOTAL_JARAK_BULAT)} "
                    "(maksimum data latih); boleh > 1.0 bila jarak melebihi data latih."
                )

            # Tombol Klasifikasi
            if st.button("Klasifikasi dan Prediksi", type="primary", use_container_width=True):
                if vol_input <= 0:
                    st.error("INPUT harus lebih dari 0.")
                else:
                    try:
                        X = np.array([[
                            fitur["rasio_angkut"],
                            fitur["rasio_diolah"],
                            fitur["rasio_sisa"],
                            fitur["indeks_jarak"],
                        ]])
                        prediksi = model.predict(X)[0]
                        label = LABEL_MAP.get(str(prediksi).strip().upper())

                        if label not in LABEL_CONFIG:
                            label = "WASPADA"

                        params_raw = {
                            "input": vol_input,
                            "angkut": angkut,
                            "diolah": diolah,
                            "sisa": sisa_eff,
                            "jarak_bulat": jarak_bulat,
                        }

                        st.session_state.wilayah_status[selected_wilayah] = label
                        st.session_state.wilayah_params[selected_wilayah] = params_raw

                        st.session_state.prediction_result = {
                            "label": label,
                            "cfg": LABEL_CONFIG[label],
                            "params": params_raw,
                            "fitur": fitur,
                            "sisa_awal": sisa_input,
                            "sisa_dikoreksi": dikoreksi,
                        }
                        st.session_state.show_prediction = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

            # TAMPILAN HASIL KLASIFIKASI
            if st.session_state.show_prediction and st.session_state.prediction_result:
                res = st.session_state.prediction_result
                label = res["label"]
                cfg = res["cfg"]
                params = res["params"]
                fitur_res = res["fitur"]

                st.markdown("---")
                st.markdown("#### Hasil Klasifikasi")

                if label == "KRITIS":
                    st.error(f"**STATUS: {label}**")
                elif label == "WASPADA":
                    st.warning(f"**STATUS: {label}**")
                else:
                    st.success(f"**STATUS: {label}**")

                st.write(cfg["desc"])

                if res.get("sisa_dikoreksi"):
                    st.info(
                        f"Nilai SISA dikoreksi dari {res['sisa_awal']:.2f} m³ "
                        f"menjadi {params['sisa']:.2f} m³."
                    )

                rekomendasi = []
                if fitur_res["rasio_sisa"] > 0.3:
                    rekomendasi.append("Rasio sisa tinggi — tambah frekuensi pengangkutan")
                if fitur_res["rasio_angkut"] < 0.7:
                    rekomendasi.append("Rasio angkut rendah — evaluasi armada")
                if fitur_res["rasio_diolah"] < 0.4:
                    rekomendasi.append("Rasio diolah rendah — tingkatkan kapasitas pengolahan")
                if fitur_res["indeks_jarak"] > 0.7:
                    rekomendasi.append("Jarak ke TPA jauh — optimasi rute")
                if not rekomendasi:
                    rekomendasi.append("Semua indikator dalam kondisi baik")

                st.markdown("**Ringkasan Input Mentah:**")
                raw_cols = st.columns(5)
                raw_values = [
                    ("INPUT (m³)", params["input"]),
                    ("ANGKUT (m³)", params["angkut"]),
                    ("DIOLAH (m³)", params["diolah"]),
                    ("SISA (m³)", params["sisa"]),
                    ("Jarak (km)", params["jarak_bulat"]),
                ]
                for rcol, (rlab, rval) in zip(raw_cols, raw_values):
                    rcol.metric(rlab, f"{rval:.2f}")

                st.markdown("**Fitur Model:**")
                fit_cols = st.columns(4)
                fit_values = [
                    ("Rasio Angkut", fitur_res["rasio_angkut"]),
                    ("Rasio Diolah", fitur_res["rasio_diolah"]),
                    ("Rasio Sisa", fitur_res["rasio_sisa"]),
                    ("Indeks Jarak", fitur_res["indeks_jarak"]),
                ]
                for fcol, (flab, fval) in zip(fit_cols, fit_values):
                    fcol.metric(flab, f"{fval:.3f}")

                st.markdown("**Rekomendasi:**")
                for r in rekomendasi:
                    st.write(f"- {r}")

    with col_right:
        st.markdown("### Peta Wilayah Kota Bandung")

        def manual_color(name):
            return {"KRITIS": "#E74C3C", "WASPADA": "#F39C12", "AMAN": "#27AE60"}.get(
                st.session_state.wilayah_status.get(name, ""), "#95A5A6")

        def manual_popup(name):
            params = st.session_state.wilayah_params.get(name, {})
            status_text = st.session_state.wilayah_status.get(name, 'Belum diprediksi')
            if params:
                f = derive_features(
                    params["input"], params["angkut"], params["diolah"],
                    params["sisa"], params["jarak_bulat"]
                )
                raw_html = (
                    f"INPUT: {params['input']:.2f} m³<br>"
                    f"ANGKUT: {params['angkut']:.2f} m³<br>"
                    f"DIOLAH: {params['diolah']:.2f} m³<br>"
                    f"SISA: {params['sisa']:.2f} m³<br>"
                    f"Jarak: {params['jarak_bulat']:.0f} km"
                )
                fitur_html = (
                    f"Rasio Angkut: {f['rasio_angkut']:.3f}<br>"
                    f"Rasio Diolah: {f['rasio_diolah']:.3f}<br>"
                    f"Rasio Sisa: {f['rasio_sisa']:.3f}<br>"
                    f"Indeks Jarak: {f['indeks_jarak']:.3f}"
                )
            else:
                raw_html = "Input mentah belum diisi"
                fitur_html = "-"
            return (f'<div style="min-width: 200px;"><b>{name}</b><br>Status: {status_text}<br><hr>'
                    f'{raw_html}<hr>{fitur_html}</div>')

        m1 = build_map(
            geojson_data, manual_color, manual_popup, show_boundary, show_labels,
            [("#e74c3c", "Kritis"), ("#f39c12", "Waspada"), ("#27ae60", "Aman"), ("#95a5a6", "Belum")],
        )
        show_map(m1, 700, 550)
        st.caption("Klik area untuk melihat detail | Warna berubah setelah klasifikasi")


# ==================================================================== TAB 2: UPLOAD BATCH
@st.cache_data(show_spinner=False)
def process_batch(file_bytes: bytes):
    df = bu.read_upload(file_bytes)
    clean, notes = bu.preprocess(df)
    return clean, notes


with tab_batch:
    st.markdown("### Upload Data Mentah (Excel)")
    st.write(
        "Unggah file Excel data volume sampah dan jarak armada per TPS. Aplikasi membersihkan data, "
        "menghitung fitur, mengklasifikasikan setiap TPS, lalu merangkum hasilnya per SWK pada peta "
        "dan tabel beserta tindakan rekomendasinya."
    )

    with st.expander("Format file yang diterima", expanded=False):
        st.markdown(
            "- **Format asli Dinas** (header 2 baris: NO, SWK, NAMA TPS, INPUT, ANGKUT, DIOLAH, SISA, ... "
            "JUMLAH JARAK - pembulatan), atau\n"
            "- **Template satu baris header**: `SWK`, `NAMA_TPS`, `INPUT`, `ANGKUT`, `DIOLAH`, `SISA`, "
            "`TOTAL_JARAK_BULAT` (volume dalam m³, jarak dalam km).\n\n"
            "Nama SWK yang dikenali di peta: BOJONEGARA, CIBEUNYING, KAREES, KORDOBA, TEGALEGA, UBERMANIK."
        )
        st.download_button(
            "Unduh template Excel", data=bu.make_template(),
            file_name="template_data_tps.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    uploaded = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], key="batch_file")

    with st.expander("Pengaturan ambang status SWK", expanded=False):
        st.caption(
            "Status tiap TPS berasal dari model. Status SWK adalah ringkasan kebijakan di tingkat wilayah "
            "dan dapat disesuaikan di sini."
        )
        thr_k = st.slider("SWK berstatus KRITIS bila % TPS kritis ≥", 1, 50, 10, key="thr_k")
        thr_w = st.slider("SWK berstatus WASPADA bila % TPS waspada ≥ (atau ada TPS kritis)",
                          10, 80, 35, key="thr_w")

    if uploaded is None:
        st.info("Belum ada file. Unggah file Excel untuk memulai.")
    else:
        try:
            clean, notes = process_batch(uploaded.getvalue())
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            clean = None

        if clean is not None and len(clean) == 0:
            st.error("Tidak ada baris data TPS yang valid pada file ini.")
        elif clean is not None:
            result = bu.predict(clean, model)
            summary = bu.summarize_swk(result, thr_k, thr_w)

            # ---- metrik ringkas
            cnt = result["status"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total TPS", len(result))
            c2.metric("AMAN", int(cnt.get("AMAN", 0)))
            c3.metric("WASPADA", int(cnt.get("WASPADA", 0)))
            c4.metric("KRITIS", int(cnt.get("KRITIS", 0)))

            with st.expander("Catatan kualitas data", expanded=False):
                st.write(
                    f"- Baris dibaca: {notes['baris_dibaca']}; duplikat dihapus: {notes['duplikat_dihapus']}; "
                    f"tanpa data jarak (dibuang): {notes['tanpa_jarak_dibuang']}\n"
                    f"- Nilai kosong diisi median per SWK: INPUT {notes['INPUT_kosong_diisi_median']}, "
                    f"ANGKUT {notes['ANGKUT_kosong_diisi_median']}, DIOLAH {notes['DIOLAH_kosong_diisi_median']}\n"
                    f"- SISA dikoreksi (neraca selisih > 3 m³ atau kosong): {notes['sisa_dikoreksi']}\n"
                    f"- TPS dengan INPUT = 0 (rasio dianggap 0): {notes['input_nol']}\n"
                    f"- TPS dengan jarak > {int(bu.MAX_TOTAL_JARAK_BULAT)} km (di luar data latih): "
                    f"{notes['jarak_melebihi_data_latih']}"
                )
                unknown = sorted(set(result["SWK"]) - set(bu.SWK_TO_GEO))
                if unknown:
                    st.warning("SWK berikut tidak ada di peta (hanya tampil di tabel): " + ", ".join(unknown))

            # ---- peta + tabel ringkasan
            swk_info = summary.set_index("SWK")

            def batch_swk_of(geo_name):
                return bu.GEO_TO_SWK.get(geo_name.strip().lower())

            def batch_color(geo_name):
                swk = batch_swk_of(geo_name)
                if swk in swk_info.index:
                    return bu.STATUS_COLOR[swk_info.loc[swk, "Status SWK"]]
                return "#95A5A6"

            def batch_popup(geo_name):
                swk = batch_swk_of(geo_name)
                if swk in swk_info.index:
                    r = swk_info.loc[swk]
                    return (
                        f'<div style="min-width:240px;"><b>SWK {swk}</b> ({geo_name})<br>'
                        f'Status SWK: <b>{r["Status SWK"]}</b><br>'
                        f'TPS: {r["Jumlah TPS"]} | Aman {r["AMAN"]} | Waspada {r["WASPADA"]} | Kritis {r["KRITIS"]}<br>'
                        f'% Kritis: {r["% Kritis"]}% | % Waspada: {r["% Waspada"]}%<hr>'
                        f'{r["Tindakan rekomendasi"]}</div>'
                    )
                return f"<b>{geo_name}</b><br>Tidak ada data pada file"

            def batch_label(geo_name):
                swk = batch_swk_of(geo_name)
                return f"{geo_name} ({swk})" if swk else geo_name

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                st.markdown("### Peta Status per SWK")
                m2 = build_map(
                    geojson_data, batch_color, batch_popup, show_boundary, show_labels,
                    [("#e74c3c", "SWK Kritis"), ("#f39c12", "SWK Waspada"),
                     ("#27ae60", "SWK Aman"), ("#95a5a6", "Tidak ada data")],
                    label_fn=batch_label,
                )
                show_map(m2, 700, 520)
                st.caption("Klik area untuk melihat ringkasan dan rekomendasi SWK. "
                           "Satu SWK dapat terdiri dari beberapa poligon pada peta (mis. UBERMANIK).")
            with col_b:
                st.markdown("### Komposisi Status TPS per SWK")
                chart = summary.set_index("SWK")[["KRITIS", "WASPADA", "AMAN"]]
                st.bar_chart(chart, color=["#E74C3C", "#F39C12", "#27AE60"], stack=True)

            st.markdown("### Ringkasan per SWK dan Tindakan Rekomendasi")

            def color_status(v):
                c = bu.STATUS_COLOR.get(v)
                if not c:
                    return ""
                fg = "#222" if v == "WASPADA" else "white"
                return f"background-color:{c}; color:{fg}; font-weight:bold"

            show_cols = ["Prioritas", "SWK", "Jumlah TPS", "AMAN", "WASPADA", "KRITIS",
                         "% Aman", "% Waspada", "% Kritis", "Status SWK"]
            styler = summary[show_cols].style.format(
                {"% Aman": "{:.1f}", "% Waspada": "{:.1f}", "% Kritis": "{:.1f}"})
            styler = (styler.map if hasattr(styler, "map") else styler.applymap)(
                color_status, subset=["Status SWK"])
            st.dataframe(styler, hide_index=True, use_container_width=True)

            st.markdown("#### Tindakan Rekomendasi per SWK")
            for _, r in summary.iterrows():
                body = f"**{r['Prioritas']}. SWK {r['SWK']}** ({r['Status SWK']}) — {r['Tindakan rekomendasi']}"
                if r["TPS kritis"] != "-":
                    body += f"\n\n*TPS kritis:* {r['TPS kritis']}"
                if r["Status SWK"] == "KRITIS":
                    st.error(body)
                elif r["Status SWK"] == "WASPADA":
                    st.warning(body)
                else:
                    st.success(body)
            st.caption(
                "Prioritas diurutkan dari % TPS kritis tertinggi, lalu % TPS waspada. "
                "Rekomendasi dihasilkan dari aturan sederhana pada TPS non-AMAN (sisa > 0,3; angkut < 0,7; "
                "jarak > 0,7 dari maksimum data latih), sehingga perlu dikonfirmasi dengan kondisi lapangan."
            )

            # ---- detail TPS
            st.markdown("### Detail Status per TPS")
            f1, f2 = st.columns(2)
            swk_filter = f1.multiselect("Filter SWK", sorted(result["SWK"].unique()))
            status_filter = f2.multiselect("Filter status", bu.STATUS_ORDER)
            view = result.copy()
            if swk_filter:
                view = view[view["SWK"].isin(swk_filter)]
            if status_filter:
                view = view[view["status"].isin(status_filter)]
            view["_o"] = view["status"].map({s: i for i, s in enumerate(bu.STATUS_ORDER)})
            view = view.sort_values(["_o", "SWK", "NAMA_TPS"]).drop(columns="_o")

            detail = view[["SWK", "NAMA_TPS", "INPUT", "ANGKUT", "DIOLAH", "SISA", "TOTAL_JARAK_BULAT",
                           "rasio_angkut", "rasio_diolah", "rasio_sisa", "indeks_jarak",
                           "status", "rekomendasi"]].rename(columns={"status": "Status", "rekomendasi": "Rekomendasi"})
            fmt = {c: "{:.3f}" for c in ["rasio_angkut", "rasio_diolah", "rasio_sisa", "indeks_jarak"]}
            fmt.update({c: "{:.2f}" for c in ["INPUT", "ANGKUT", "DIOLAH", "SISA"]})
            dstyler = detail.style.format(fmt)
            dstyler = (dstyler.map if hasattr(dstyler, "map") else dstyler.applymap)(
                color_status, subset=["Status"])
            st.dataframe(dstyler, hide_index=True, use_container_width=True)

            # ---- unduh
            full_detail = result[["SWK", "NAMA_TPS", "INPUT", "ANGKUT", "DIOLAH", "SISA", "TOTAL_JARAK_BULAT",
                                  "rasio_angkut", "rasio_diolah", "rasio_sisa", "indeks_jarak",
                                  "status", "rekomendasi"]].rename(
                columns={"status": "Status", "rekomendasi": "Rekomendasi"})
            st.download_button(
                "Unduh hasil (Excel: ringkasan SWK, detail TPS, catatan data)",
                data=bu.export_excel(full_detail, summary, notes),
                file_name="hasil_klasifikasi_tps.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray;'>Developed by Masoem University – Fakultas Teknik</div>",
    unsafe_allow_html=True
)
