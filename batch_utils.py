"""Fungsi pemrosesan batch (tanpa Streamlit) untuk aplikasi Klasifikasi Pengelolaan Sampah.

Alur mengikuti pipeline pelatihan (notebook v3):
  baca Excel -> cleaning -> rasio fitur -> prediksi model -> ringkasan per SWK + rekomendasi
"""
import io
import re

import numpy as np
import pandas as pd

FEATURE_COLS = ["rasio_angkut", "rasio_diolah", "rasio_sisa", "indeks_jarak"]

# Nilai maksimum TOTAL_JARAK_BULAT pada data latih (km) -- sama dengan app manual
MAX_TOTAL_JARAK_BULAT = 91.0

LABEL_MAP = {
    "SAFE": "AMAN", "ALERT": "WASPADA", "CRITICAL": "KRITIS",
    "AMAN": "AMAN", "WASPADA": "WASPADA", "KRITIS": "KRITIS",
}
STATUS_ORDER = ["KRITIS", "WASPADA", "AMAN"]
STATUS_COLOR = {"KRITIS": "#E74C3C", "WASPADA": "#F39C12", "AMAN": "#27AE60"}

# Pemetaan SWK (data Excel) -> nama wilayah di file GeoJSON.
# CATATAN: GeoJSON memuat 8 wilayah, data Excel 6 SWK. UBERMANIK mencakup 3 poligon.
# Verifikasi ke dokumen resmi pembagian SWK sebelum dipublikasikan.
SWK_TO_GEO = {
    "BOJONEGARA": ["bojonagara", "bojonegara"],
    "CIBEUNYING": ["cibeunying"],
    "KAREES": ["karees"],
    "KORDOBA": ["kordon", "kordoba"],
    "TEGALEGA": ["tegallega", "tegalega"],
    "UBERMANIK": ["ujungberung", "gedebage", "arcamanik", "ubermanik"],
}
GEO_TO_SWK = {g: swk for swk, names in SWK_TO_GEO.items() for g in names}

# Penyebab yang diperiksa pada TPS non-AMAN (aturan sama dengan rekomendasi input manual)
CAUSES = [
    ("rasio_sisa", lambda s: s > 0.3, "sampah tersisa tinggi",
     "tambah frekuensi pengangkutan"),
    ("rasio_angkut", lambda s: s < 0.7, "rasio angkut rendah",
     "evaluasi armada dan efektivitas pengangkutan"),
    ("indeks_jarak", lambda s: s > 0.7, "jarak ke TPA jauh",
     "optimasi rute atau pertimbangkan titik transfer (transfer point)"),
]


# ------------------------------------------------------------------ membaca Excel
def _norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().upper() if pd.notna(x) else ""


def read_upload(file_bytes: bytes) -> pd.DataFrame:
    """Baca Excel mentah. Mendukung (a) format asli Dinas (header 2 baris) dan
    (b) template satu baris header: SWK, NAMA_TPS, INPUT, ANGKUT, DIOLAH, SISA, TOTAL_JARAK_BULAT."""
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    # cari baris yang berisi sel persis "INPUT"
    hdr = None
    for i in range(min(len(raw), 15)):
        if "INPUT" in [_norm(v) for v in raw.iloc[i].tolist()]:
            hdr = i
            break
    if hdr is None:
        raise ValueError("Kolom INPUT tidak ditemukan. Gunakan format asli atau unduh template.")

    top = [_norm(v) for v in raw.iloc[hdr - 1].tolist()] if hdr > 0 else [""] * raw.shape[1]
    sub = [_norm(v) for v in raw.iloc[hdr].tolist()]
    full = [f"{a} {b}".strip() for a, b in zip(top, sub)]

    def find(pred, names):
        for j, n in enumerate(names):
            if pred(n):
                return j
        return None

    c_swk = find(lambda n: n == "SWK" or n.endswith(" SWK") or n.startswith("SWK"), full)
    c_nama = find(lambda n: "NAMA" in n, full)
    c_in = find(lambda n: n == "INPUT", sub)
    c_ang = find(lambda n: n == "ANGKUT", sub)
    c_dio = find(lambda n: n == "DIOLAH", sub)
    c_sis = find(lambda n: n == "SISA", sub)
    c_no = find(lambda n: n == "NO" or n.startswith("NO "), full)

    c_jar, jarak_dibulatkan = find(
        lambda n: "PEMBULATAN" in n or "JARAK_BULAT" in n or "JARAK BULAT" in n, full), True
    if c_jar is None:
        c_jar = find(lambda n: "TOTAL_JARAK" in n or "TOTAL JARAK" in n
                     or ("JUMLAH" in n and "JARAK" in n), full)
        jarak_dibulatkan = False

    missing = [nm for nm, c in [("SWK", c_swk), ("NAMA_TPS", c_nama), ("INPUT", c_in),
                                ("ANGKUT", c_ang), ("DIOLAH", c_dio), ("SISA", c_sis),
                                ("TOTAL_JARAK_BULAT", c_jar)] if c is None]
    if missing:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(missing))

    body = raw.iloc[hdr + 1:].reset_index(drop=True)
    df = pd.DataFrame({
        "SWK": body.iloc[:, c_swk].astype("string").str.strip().str.upper(),
        "NAMA_TPS": body.iloc[:, c_nama].astype("string").str.strip(),
        "INPUT": pd.to_numeric(body.iloc[:, c_in], errors="coerce"),
        "ANGKUT": pd.to_numeric(body.iloc[:, c_ang], errors="coerce"),
        "DIOLAH": pd.to_numeric(body.iloc[:, c_dio], errors="coerce"),
        "SISA": pd.to_numeric(body.iloc[:, c_sis], errors="coerce"),
        "TOTAL_JARAK_BULAT": pd.to_numeric(body.iloc[:, c_jar], errors="coerce"),
    })
    if c_no is not None:
        df = df[body.iloc[:, c_no].notna().values]
    df = df[df["SWK"].notna() & df["NAMA_TPS"].notna()].copy()
    if not jarak_dibulatkan:
        df["TOTAL_JARAK_BULAT"] = np.ceil(df["TOTAL_JARAK_BULAT"])
    df["SWK"] = df["SWK"].astype(str)
    df["NAMA_TPS"] = df["NAMA_TPS"].astype(str)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ cleaning + fitur
def preprocess(df: pd.DataFrame):
    """Cleaning sama dengan pipeline pelatihan. Mengembalikan (df_bersih, catatan)."""
    notes = {}
    d = df.copy()
    notes["baris_dibaca"] = len(d)

    n0 = len(d)
    d = d.drop_duplicates(subset=["NAMA_TPS", "INPUT", "ANGKUT", "DIOLAH", "SISA"], keep="first")
    notes["duplikat_dihapus"] = n0 - len(d)

    n0 = len(d)
    d = d[d["TOTAL_JARAK_BULAT"].notna()].copy()
    notes["tanpa_jarak_dibuang"] = n0 - len(d)

    for c in ["INPUT", "ANGKUT", "DIOLAH"]:
        notes[f"{c}_kosong_diisi_median"] = int(d[c].isna().sum())
        d[c] = d.groupby("SWK")[c].transform(lambda x: x.fillna(x.median()))
        d[c] = d[c].fillna(0)

    sisa_hitung = (d["INPUT"] - d["ANGKUT"] - d["DIOLAH"]).clip(lower=0)
    selisih = (d["SISA"] - sisa_hitung).abs()
    mask = d["SISA"].isna() | (selisih > 3)
    notes["sisa_dikoreksi"] = int(mask.sum())
    d.loc[mask, "SISA"] = sisa_hitung[mask]
    d["SISA"] = d["SISA"].clip(lower=0)

    inp = d["INPUT"].replace(0, np.nan)
    d["rasio_angkut"] = (d["ANGKUT"] / inp).fillna(0)
    d["rasio_diolah"] = (d["DIOLAH"] / inp).fillna(0)
    d["rasio_sisa"] = (d["SISA"] / inp).fillna(0)
    d["indeks_jarak"] = d["TOTAL_JARAK_BULAT"] / MAX_TOTAL_JARAK_BULAT
    notes["input_nol"] = int((d["INPUT"] == 0).sum())
    notes["jarak_melebihi_data_latih"] = int((d["TOTAL_JARAK_BULAT"] > MAX_TOTAL_JARAK_BULAT).sum())
    return d.reset_index(drop=True), notes


def tps_recommendations(row) -> str:
    if row["status"] == "AMAN":
        return "Pertahankan; pantau rutin"
    rec = [f"{c[2]} - {c[3]}" for c in CAUSES if c[1](row[c[0]])]
    return "; ".join(rec) if rec else "Pantau; indikator mendekati ambang"


def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    d = df.copy()
    pred = model.predict(d[FEATURE_COLS].values)
    d["status"] = [LABEL_MAP.get(str(p).strip().upper(), "WASPADA") for p in pred]
    d["rekomendasi"] = d.apply(tps_recommendations, axis=1)
    return d


# ------------------------------------------------------------------ ringkasan SWK
def swk_status(pct_kritis, pct_waspada, thr_kritis, thr_waspada):
    if pct_kritis >= thr_kritis:
        return "KRITIS"
    if pct_kritis > 0 or pct_waspada >= thr_waspada:
        return "WASPADA"
    return "AMAN"


def swk_recommendation(g: pd.DataFrame, status: str) -> str:
    nonsafe = g[g["status"] != "AMAN"]
    pct_aman = (g["status"] == "AMAN").mean() * 100
    causes = []
    if len(nonsafe):
        shares = [(float(c[1](nonsafe[c[0]]).mean()), c) for c in CAUSES]
        shares = sorted(shares, key=lambda t: -t[0])
        causes = [c for s, c in shares if s >= 0.5] or [c for s, c in shares[:1] if s >= 0.3]
    fokus = ""
    if causes:
        fokus = " Penyebab dominan: " + ", ".join(c[2] for c in causes) + \
                ". Tindakan: " + "; ".join(c[3] for c in causes) + "."
    if status == "KRITIS":
        if pct_aman >= 60:
            head = "Prioritas: masalah terkonsentrasi pada TPS tertentu, lakukan intervensi terarah pada TPS kritis (bukan seluruh wilayah)."
        else:
            head = "Prioritas utama: lakukan penanganan menyeluruh di wilayah ini, mulai dari TPS kritis."
    elif status == "WASPADA":
        head = "Lakukan tindakan pencegahan agar TPS waspada tidak menjadi kritis."
    else:
        return "Pertahankan kinerja pengelolaan dan pantau secara rutin."
    return head + fokus


def summarize_swk(d: pd.DataFrame, thr_kritis=10.0, thr_waspada=35.0) -> pd.DataFrame:
    rows = []
    for swk, g in d.groupby("SWK"):
        n = len(g)
        c = g["status"].value_counts()
        n_k, n_w, n_a = int(c.get("KRITIS", 0)), int(c.get("WASPADA", 0)), int(c.get("AMAN", 0))
        pk, pw, pa = n_k / n * 100, n_w / n * 100, n_a / n * 100
        st_ = swk_status(pk, pw, thr_kritis, thr_waspada)
        kritis_names = g.loc[g["status"] == "KRITIS", "NAMA_TPS"].tolist()
        rows.append({
            "SWK": swk, "Jumlah TPS": n,
            "AMAN": n_a, "WASPADA": n_w, "KRITIS": n_k,
            "% Aman": round(pa, 1), "% Waspada": round(pw, 1), "% Kritis": round(pk, 1),
            "Status SWK": st_,
            "TPS kritis": ", ".join(kritis_names) if kritis_names else "-",
            "Tindakan rekomendasi": swk_recommendation(g, st_),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["% Kritis", "% Waspada"], ascending=False).reset_index(drop=True)
    out.insert(0, "Prioritas", range(1, len(out) + 1))
    return out


# ------------------------------------------------------------------ ekspor / template
def make_template() -> bytes:
    df = pd.DataFrame({
        "SWK": ["BOJONEGARA", "KORDOBA"],
        "NAMA_TPS": ["CONTOH TPS A", "CONTOH TPS B"],
        "INPUT": [24, 10],
        "ANGKUT": [3, 9],
        "DIOLAH": [0, 0],
        "SISA": [21, 1],
        "TOTAL_JARAK_BULAT": [52, 60],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Data")
    return buf.getvalue()


def export_excel(detail: pd.DataFrame, summary: pd.DataFrame, notes: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="Ringkasan SWK")
        detail.to_excel(w, index=False, sheet_name="Detail TPS")
        pd.DataFrame({"Catatan": list(notes.keys()), "Nilai": list(notes.values())}).to_excel(
            w, index=False, sheet_name="Catatan data")
    return buf.getvalue()
