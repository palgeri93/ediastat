import os
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Fix prefixek + jelmagyarázat
# ----------------------------
DOMAIN_PREFIXES = {
    "Matematika": ["Mat", "MA", "MD", "MG"],
    "Olvasás": ["Olv", "OA", "OD", "OG"],
    "Természettudomány": ["Term", "TA", "TD", "TG"],
}
EXTRA_PREFIXES = ["GH"]  # Géphaszn.
ALL_PREFIXES = set(sum(DOMAIN_PREFIXES.values(), [])) | set(EXTRA_PREFIXES)

PREFIX_LABELS = {
    "Mat": "Matematika",
    "MA": "Matematika tantárgyi tudás / alkalmazás",
    "MD": "Matematika diszciplináris gondolkodás",
    "MG": "Matematika gondolkodási műveletek",
    "Olv": "Olvasás",
    "OA": "Olvasás alkalmazás",
    "OD": "Olvasás diszciplináris gondolkodás",
    "OG": "Olvasás gondolkodási műveletek",
    "Term": "Természettudomány",
    "TA": "Természettudomány alkalmazás",
    "TD": "Természettudomány diszciplináris gondolkodás",
    "TG": "Természettudomány gondolkodási műveletek",
    "GH": "Géphasználat",
}
SHORT_PREFIX_LABELS = {
    "Mat": "Mat. össz.", "MA": "Mat. alk.", "MD": "Mat. disz.", "MG": "Mat. gond.",
    "Olv": "Olv. össz.", "OA": "Olv. alk.", "OD": "Olv. disz.", "OG": "Olv. gond.",
    "Term": "Term. össz.", "TA": "Term. alk.", "TD": "Term. disz.", "TG": "Term. gond.",
    "GH": "Géphaszn.",
}

MISSING_NAME = "#HIÁNYZIK"
AVERAGE_LINE_VALUE = 500
AVERAGE_LINE_LABEL = "Sokévi átlag"
MAX_PERIODS_TO_SHOW = 6
PERIOD_SHEET_REGEX = r"^\s*\d{4}_\d{4}_\d+\s*$"
CLASS_AVG_REGEX = r"Osztály\s*\((.*?)\)\s*eredménye"
HEADER_REGEX = r"Mérési\s*azonosító"

COMPARISON_ROW_LABELS = {
    "class": "Osztályátlag",
    "region": "Régió",
    "settlement": "Településtípus",
    "national": "Országos",
}


def normalize_class_label(value) -> str:
    """1.a / 1a / 1.A alakok egységesítése: 1.a"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    m = re.match(r"^\s*(\d+)\s*\.?\s*([A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű])\s*$", text)
    if m:
        return f"{int(m.group(1))}.{m.group(2).lower()}"
    return text


def class_label_sort_key(label: str) -> Tuple:
    text = normalize_class_label(label)
    m = re.match(r"^(\d+)\.([A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű])$", text)
    if m:
        return (int(m.group(1)), m.group(2))
    return (999, text)


def read_class_mapping_from_excel(xls) -> Dict[str, str]:
    """
    osztaly_osszerendeles munkalap beolvasása.
    A oszlop: egyértelmű osztálykód / kohorsz-kód, pl. 2025-26A
    B oszlop: felületen megjelenő osztály, pl. 1.a
    """
    mapping: Dict[str, str] = {}
    if "osztaly_osszerendeles" not in xls.sheet_names:
        return mapping
    try:
        df = xls.parse("osztaly_osszerendeles", header=None, dtype=object)
    except Exception:
        return mapping
    if df.empty or df.shape[1] < 2:
        return mapping
    for _, row in df.iloc[:, :2].iterrows():
        code = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        label = normalize_class_label(row.iloc[1])
        if code and label:
            mapping[code] = label
    return mapping


def parse_period_sort_key(sheet_name: str) -> Tuple:
    s = str(sheet_name).strip()
    m = re.match(r"^\s*(\d{4})_(\d{4})_(\d+)\s*$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (9999, 9999, s)


def get_prefix(col_name: str) -> str:
    s = str(col_name).strip()
    if ":" in s:
        return s.split(":", 1)[0].strip()
    m = re.match(r"^([A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)", s)
    return m.group(1) if m else s


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if get_prefix(c) in ALL_PREFIXES]


def read_excel_any_format(uploaded_file) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Excel beolvasás biztonságosan Windows alatt is.
    A feltöltött fájlt memóriából olvassuk, így nem marad zárolt ideiglenes fájl.
    .xls esetén először xlrd-vel próbálkozunk, végső esetben LibreOffice-konverzióval.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()

    if suffix == ".xlsx":
        with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as xls:
            period_sheets = [s for s in xls.sheet_names if re.match(PERIOD_SHEET_REGEX, str(s).strip())]
            period_sheets = sorted(period_sheets, key=parse_period_sort_key)
            dfs = {sh: xls.parse(sh, dtype=object) for sh in period_sheets}
            class_mapping = read_class_mapping_from_excel(xls)
        return period_sheets, dfs, class_mapping

    if suffix == ".xls":
        try:
            with pd.ExcelFile(BytesIO(file_bytes), engine="xlrd") as xls:
                period_sheets = [s for s in xls.sheet_names if re.match(PERIOD_SHEET_REGEX, str(s).strip())]
                period_sheets = sorted(period_sheets, key=parse_period_sort_key)
                dfs = {sh: xls.parse(sh, dtype=object) for sh in period_sheets}
                class_mapping = read_class_mapping_from_excel(xls)
            return period_sheets, dfs, class_mapping
        except Exception:
            pass

        soffice = shutil.which("libreoffice") or shutil.which("soffice")
        if not soffice:
            raise RuntimeError(
                "Az .xls fájl olvasásához telepítsd az xlrd csomagot, vagy mentsd/töltsd fel .xlsx formátumban."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, uploaded_file.name)
            with open(src, "wb") as f:
                f.write(file_bytes)
            subprocess.run(
                [soffice, "--headless", "--convert-to", "xlsx", "--outdir", tmpdir, src],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            read_path = os.path.splitext(src)[0] + ".xlsx"
            with pd.ExcelFile(read_path, engine="openpyxl") as xls:
                period_sheets = [s for s in xls.sheet_names if re.match(PERIOD_SHEET_REGEX, str(s).strip())]
                period_sheets = sorted(period_sheets, key=parse_period_sort_key)
                dfs = {sh: xls.parse(sh, dtype=object) for sh in period_sheets}
                class_mapping = read_class_mapping_from_excel(xls)
            return period_sheets, dfs, class_mapping

    raise RuntimeError("Csak .xlsx vagy .xls fájl tölthető fel.")


@st.cache_data(show_spinner=False)
def read_measurement_sheets(uploaded_file) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, str]]:
    return read_excel_any_format(uploaded_file)


def clean_period_df(df: pd.DataFrame, remove_summary_rows: bool = True) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    out = df.copy()
    out = out.rename(columns={out.columns[0]: "TanulóID", out.columns[1]: "TanulóNév"})
    out["TanulóID"] = out["TanulóID"].apply(normalize_id)
    out["TanulóNév"] = out["TanulóNév"].astype(str).str.strip()

    mask = out["TanulóID"].ne("") & out["TanulóNév"].ne("") & out["TanulóNév"].ne("nan")
    mask &= out["TanulóNév"].ne(MISSING_NAME)
    if remove_summary_rows:
        mask &= ~out["TanulóID"].str.contains("eredménye", case=False, na=False)
        mask &= ~out["TanulóID"].str.contains(HEADER_REGEX, case=False, na=False)
    return out[mask].copy()


def build_long_table(period_order: List[str], period_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for period in period_order:
        df = clean_period_df(period_dfs.get(period), remove_summary_rows=True)
        if df.empty:
            continue
        for metric in metric_columns(df):
            prefix = get_prefix(metric)
            sub = df[["TanulóID", "TanulóNév", metric]].copy().rename(columns={metric: "Érték"})
            sub["Mérés"] = period
            sub["Prefix"] = prefix
            sub["Érték"] = pd.to_numeric(sub["Érték"], errors="coerce")
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["Mérés", "TanulóID", "TanulóNév", "Prefix", "Érték"])
    return pd.concat(rows, ignore_index=True)


def find_class_blocks(period: str, df: pd.DataFrame, class_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Osztályblokkok keresése: tanulók a class average sor fölött, az előző fejléc/üres sor után."""
    if df is None or df.empty:
        return []
    raw = df.copy()
    first_col = raw.columns[0]
    second_col = raw.columns[1] if raw.shape[1] > 1 else None
    blocks = []

    for avg_idx, first_val in raw[first_col].items():
        text = "" if pd.isna(first_val) else str(first_val).strip()
        m = re.search(CLASS_AVG_REGEX, text, flags=re.IGNORECASE)
        if not m:
            continue
        class_code = m.group(1).strip()
        class_name = (class_mapping or {}).get(class_code, normalize_class_label(class_code))

        start_idx = 0
        j = avg_idx - 1
        while j >= 0:
            v = raw.at[j, first_col]
            if is_blank(v) or re.search(HEADER_REGEX, str(v), flags=re.IGNORECASE):
                start_idx = j + 1
                break
            j -= 1

        students_df = raw.loc[start_idx:avg_idx - 1].copy()
        if second_col is not None:
            students_df = students_df.rename(columns={first_col: "TanulóID", second_col: "TanulóNév"})
            students_df["TanulóID"] = students_df["TanulóID"].apply(normalize_id)
            students_df["TanulóNév"] = students_df["TanulóNév"].astype(str).str.strip()
            students_df = students_df[
                students_df["TanulóID"].ne("")
                & students_df["TanulóNév"].ne("")
                & students_df["TanulóNév"].ne("nan")
                & students_df["TanulóNév"].ne(MISSING_NAME)
            ]

        comparison_rows = raw.loc[avg_idx:avg_idx + 3].copy()
        comparison_rows = comparison_rows.rename(columns={first_col: "Mutató"})
        labels = ["class", "region", "settlement", "national"][: len(comparison_rows)]
        comparison_rows["SorTípus"] = labels
        comparison_rows["Mutató"] = comparison_rows["SorTípus"].map(COMPARISON_ROW_LABELS)

        blocks.append(
            {
                "Mérés": period,
                "Osztály": class_name,
                "OsztályKód": class_code,
                "avg_idx": avg_idx,
                "students": students_df,
                "comparison": comparison_rows,
            }
        )
    return blocks


def build_class_blocks(period_order: List[str], period_dfs: Dict[str, pd.DataFrame], class_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
    blocks = []
    for period in period_order:
        blocks.extend(find_class_blocks(period, period_dfs.get(period), class_mapping=class_mapping))
    return blocks


def comparison_pivot(block: Dict, domain: str, include_gh: bool = False) -> pd.DataFrame:
    needed = DOMAIN_PREFIXES[domain].copy()
    if include_gh:
        needed += EXTRA_PREFIXES
    comp = block["comparison"].copy()
    keep_cols = [c for c in comp.columns if get_prefix(c) in needed]
    pv = comp.set_index("Mutató")[keep_cols]
    pv.columns = [SHORT_PREFIX_LABELS.get(get_prefix(c), get_prefix(c)) for c in keep_cols]
    return pv.apply(pd.to_numeric, errors="coerce")


def class_period_pivot(blocks: List[Dict], class_name: str, domain: str, include_gh: bool = False) -> pd.DataFrame:
    needed = DOMAIN_PREFIXES[domain].copy()
    if include_gh:
        needed += EXTRA_PREFIXES
    rows = []
    idx = []
    for b in blocks:
        if b["Osztály"] != class_name:
            continue
        comp = b["comparison"]
        class_row = comp[comp["SorTípus"] == "class"]
        if class_row.empty:
            continue
        data = {}
        for col in comp.columns:
            prefix = get_prefix(col)
            if prefix in needed:
                data[SHORT_PREFIX_LABELS.get(prefix, prefix)] = pd.to_numeric(class_row.iloc[0][col], errors="coerce")
        rows.append(data)
        idx.append(b["Mérés"])
    return pd.DataFrame(rows, index=idx)


def student_ranking(block: Dict, metric_prefix: str) -> pd.DataFrame:
    students_df = block["students"].copy()
    cols = [c for c in students_df.columns if get_prefix(c) == metric_prefix]
    if not cols:
        return pd.DataFrame()
    col = cols[0]
    out = students_df[["TanulóID", "TanulóNév", col]].copy()
    out = out.rename(columns={col: PREFIX_LABELS.get(metric_prefix, metric_prefix)})
    value_col = PREFIX_LABELS.get(metric_prefix, metric_prefix)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col]).sort_values(value_col, ascending=False).reset_index(drop=True)
    out.insert(0, "Helyezés", np.arange(1, len(out) + 1))
    return out


def pivot_for_domain(student_df: pd.DataFrame, periods: List[str], domain: str, include_gh: bool) -> pd.DataFrame:
    needed = DOMAIN_PREFIXES[domain].copy()
    if include_gh:
        needed += EXTRA_PREFIXES
    sub = student_df[student_df["Prefix"].isin(needed)].copy()
    pv = sub.pivot_table(index="Mérés", columns="Prefix", values="Érték", aggfunc="mean")
    pv = pv.reindex(index=periods, columns=needed)
    return pv.rename(columns={p: SHORT_PREFIX_LABELS.get(p, p) for p in pv.columns})


def fig_to_jpg_bytes(fig) -> bytes:
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="jpg", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_grouped_bars_with_labels(pivot_df: pd.DataFrame, title: str, average_line: bool = True) -> plt.Figure:
    periods = pivot_df.index.tolist()
    metrics = pivot_df.columns.tolist()
    values = pivot_df.values.astype(float)
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.8 / max(len(metrics), 1)
    containers = []
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * bar_width, values[:, i], width=bar_width, label=str(metric))
        containers.append(bars)
    if average_line:
        ax.axhline(AVERAGE_LINE_VALUE, linewidth=2.2, linestyle="-", label=AVERAGE_LINE_LABEL)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=0)
    ax.set_ylabel("Érték")
    # Egységes tengely minden diagramon
    ax.set_ylim(0, 900)
    ax.set_yticks(np.arange(0, 901, 100))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    for bars in containers:
        try:
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        except Exception:
            pass
    fig.tight_layout()
    return fig


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))[:80]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EDIA mérés eredmények", layout="wide")
st.title("EDIA mérés eredmények")

uploaded = st.file_uploader("Excel munkafüzet feltöltése (.xlsx vagy .xls)", type=["xlsx", "xls"])
if not uploaded:
    st.info("Tölts fel egy Excel fájlt.")
    st.stop()

try:
    period_order, period_dfs, class_mapping = read_measurement_sheets(uploaded)
except Exception as exc:
    st.error(f"Nem sikerült beolvasni az Excel fájlt: {exc}")
    st.stop()

if not period_order:
    st.error("Nem találtam mérési munkalapokat (várt minta: YYYY_YYYY_N, pl. 2025_2026_1).")
    st.stop()

period_order_last = period_order[-MAX_PERIODS_TO_SHOW:]
long_df = build_long_table(period_order_last, period_dfs)
class_blocks = build_class_blocks(period_order_last, period_dfs, class_mapping=class_mapping)

st.sidebar.header("Beállítások")
selected_domain = st.sidebar.radio("Mérési terület", options=list(DOMAIN_PREFIXES.keys()), index=0)
show_gh = st.sidebar.checkbox("GH (Géphaszn.) hozzáadása", value=False)

tab_student, tab_class = st.tabs(["Tanulói jelentés", "Osztályfelület"])

with tab_student:
    st.subheader("Tanulói teljesítmények")
    students = (
        long_df[["TanulóID", "TanulóNév"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["TanulóNév", "TanulóID"])
    )
    if students.empty:
        st.info("Nem találtam tanulói adatokat.")
    else:
        student_labels = (students["TanulóNév"] + "  (" + students["TanulóID"] + ")").tolist()
        label_to_id = dict(zip(student_labels, students["TanulóID"]))
        selected_label = st.selectbox("Tanuló (kereshető)", options=student_labels, index=0)
        selected_id = label_to_id[selected_label]
        sel_row = students[students["TanulóID"] == selected_id].iloc[0]
        sel_name, sel_id = sel_row["TanulóNév"], sel_row["TanulóID"]

        student_df = long_df[long_df["TanulóID"] == selected_id].copy()
        pv = pivot_for_domain(student_df, period_order_last, selected_domain, include_gh=show_gh).tail(MAX_PERIODS_TO_SHOW)
        chart_title = f"EDIA mérés eredmények – {selected_domain} – {sel_name} ({sel_id})"

        if pv.empty or pv.dropna(how="all").empty:
            st.info("Nincs megjeleníthető adat ehhez a tanulóhoz / területhez az utolsó időszakokban.")
        else:
            fig = plot_grouped_bars_with_labels(pv, chart_title)
            jpg_bytes = fig_to_jpg_bytes(fig)
            st.pyplot(fig)
            st.download_button(
                "Tanulói diagram letöltése JPG-ben",
                data=jpg_bytes,
                file_name=f"EDIA_{safe_filename(selected_domain)}_{safe_filename(sel_name)}_{sel_id}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with st.expander("Tanulói nyers adatok"):
            st.dataframe(student_df.sort_values(["Mérés", "Prefix"]), use_container_width=True)

with tab_class:
    st.subheader("Osztályfelület")
    if not class_blocks:
        st.info("Nem találtam osztályátlag-sorokat az utolsó mérési időszakokban.")
    else:
        mapped_class_options = sorted(set(class_mapping.values()), key=class_label_sort_key)
        present_class_options = sorted({b["Osztály"] for b in class_blocks}, key=class_label_sort_key)
        class_options = mapped_class_options or present_class_options
        selected_class = st.selectbox("Osztály", options=class_options, index=0)

        class_periods = [b["Mérés"] for b in class_blocks if b["Osztály"] == selected_class]
        if not class_periods:
            st.info("Ehhez az osztályhoz nincs adat az utolsó mérési időszakokban.")
            st.stop()

        selected_class_period = st.selectbox("Mérési időszak", options=class_periods, index=len(class_periods) - 1)
        block = next(b for b in class_blocks if b["Osztály"] == selected_class and b["Mérés"] == selected_class_period)
        internal_code_text = f" ({block.get('OsztályKód', '')})" if block.get("OsztályKód") else ""

        st.markdown(f"**{selected_class} osztály – {selected_class_period}**")
        st.caption(f"Belső összerendelési kód: {block.get('OsztályKód', '')}")

        comp_pv = comparison_pivot(block, selected_domain, include_gh=show_gh)
        if comp_pv.empty or comp_pv.dropna(how="all").empty:
            st.info("Ehhez az osztályhoz / területhez nincs megjeleníthető összehasonlító adat.")
        else:
            fig = plot_grouped_bars_with_labels(
                comp_pv,
                f"{selected_class} osztály eredménye – {selected_domain} – {selected_class_period}",
            )
            st.pyplot(fig)
            st.download_button(
                "Osztály-összehasonlító diagram letöltése JPG-ben",
                data=fig_to_jpg_bytes(fig),
                file_name=f"EDIA_osztaly_{safe_filename(selected_class)}_{safe_filename(selected_domain)}_{selected_class_period}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Tanulók sorrendezése mérési szempont alapján")
        ranking_prefixes = DOMAIN_PREFIXES[selected_domain].copy()
        if show_gh:
            ranking_prefixes += EXTRA_PREFIXES
        metric_label_to_prefix = {PREFIX_LABELS.get(p, p): p for p in ranking_prefixes}
        selected_metric_label = st.selectbox("Mérési szempont", options=list(metric_label_to_prefix.keys()))
        selected_metric = metric_label_to_prefix[selected_metric_label]
        ranking = student_ranking(block, selected_metric)
        if ranking.empty:
            st.info("Ehhez a mérési szemponthoz nincs tanulói adat az osztályblokkban.")
        else:
            st.dataframe(ranking, hide_index=True, use_container_width=True)
            csv = ranking.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Rangsor letöltése CSV-ben",
                data=csv,
                file_name=f"EDIA_rangsor_{safe_filename(selected_class)}_{safe_filename(selected_metric_label)}_{selected_class_period}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Osztályátlagok összehasonlítása mérési időszakonként")
        class_trend = class_period_pivot(class_blocks, selected_class, selected_domain, include_gh=show_gh)
        class_trend = class_trend.reindex([p for p in period_order_last if p in class_trend.index]).tail(MAX_PERIODS_TO_SHOW)
        if class_trend.empty or class_trend.dropna(how="all").empty:
            st.info("Nem található több mérési időszakhoz osztályátlag-adat.")
        else:
            fig = plot_grouped_bars_with_labels(
                class_trend,
                f"{selected_class} osztályátlagok mérési időszakonként – {selected_domain}",
            )
            st.pyplot(fig)
            st.download_button(
                "Idősoros osztályátlag-diagram letöltése JPG-ben",
                data=fig_to_jpg_bytes(fig),
                file_name=f"EDIA_osztaly_idosor_{safe_filename(selected_class)}_{safe_filename(selected_domain)}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
            with st.expander("Osztályátlagok táblázata"):
                st.dataframe(class_trend, use_container_width=True)

        with st.expander("Osztályblokk ellenőrző adatai"):
            st.write("Tanulók száma az osztályblokkban:", len(block["students"]))
            st.dataframe(block["students"], use_container_width=True)
            st.dataframe(block["comparison"], use_container_width=True)
