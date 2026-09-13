
from __future__ import annotations

import json
import os
import subprocess
import sys
from html import escape as html_escape
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import DEFAULT_DATASET, DEFAULT_MODEL
from core.validator import (
    find_label_column,
    get_feature_columns,
    load_model,
    overall_status,
    run_validation_pipeline,
)
from reports.report_generator import generate_report
from utils.logger import logger


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "runtime" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "reports" / "test_report.docx"
REPORT_META_PATH = ROOT / "runtime" / "report_meta.json"
UI_STATUS_PATH = ROOT / "runtime" / "ui_status.json"

st.set_page_config(
    page_title="ModelGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{--mg-bg:#050b14;--mg-panel:#091422;--mg-panel2:#0d1c2e;--mg-border:rgba(102,225,255,.16);--mg-cyan:#62e6ff;--mg-blue:#4d8dff;--mg-green:#57e5a1;--mg-yellow:#ffd166;--mg-red:#ff6477;--mg-text:#edf7ff;--mg-muted:#8fa6ba}
header[data-testid="stHeader"]{background:#040a12!important;border-bottom:1px solid rgba(98,230,255,.10)!important;box-shadow:none!important}
header[data-testid="stHeader"] *{color:#dcecf7!important}
div[data-testid="stToolbar"]{background:transparent!important}
[data-testid="stAppViewContainer"]{background:transparent!important}
main[data-testid="stMain"]{background:transparent!important}
div[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(180deg,#0a1727,#08121f)!important;border:1px solid rgba(98,230,255,.12)!important;border-radius:16px!important;box-shadow:0 12px 28px rgba(0,0,0,.18)!important}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{border-color:rgba(98,230,255,.26)!important}
.stApp{background:radial-gradient(circle at 75% 8%,rgba(0,196,255,.10),transparent 18%),radial-gradient(circle at 10% 18%,rgba(74,110,255,.08),transparent 24%),linear-gradient(180deg,#040a12 0%,#06111d 56%,#040b13 100%);color:var(--mg-text)}
.block-container{max-width:1540px;padding-top:1rem;padding-bottom:4rem}#MainMenu{visibility:hidden}footer{visibility:hidden}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#040b14 0%,#071425 55%,#050c16 100%);border-right:1px solid rgba(98,230,255,.14)}
section[data-testid="stSidebar"] *{color:#e9f6ff!important}section[data-testid="stSidebar"] .stRadio label{padding:11px 13px;margin:3px 0;border:1px solid transparent;border-radius:13px;transition:.16s ease}section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(98,230,255,.07);border-color:rgba(98,230,255,.15);transform:translateX(2px)}
.command-strip{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:2px 0 12px;padding:8px 2px;color:#8ea7bd;font-size:11px;letter-spacing:.9px;text-transform:uppercase}.command-left{display:flex;align-items:center;gap:9px}.live-dot{width:9px;height:9px;border-radius:50%;display:inline-block;background:var(--mg-green);box-shadow:0 0 16px rgba(87,229,161,.75)}.command-chip{padding:5px 9px;border-radius:999px;background:rgba(98,230,255,.06);border:1px solid rgba(98,230,255,.12)}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:30px 34px 28px;background:linear-gradient(125deg,rgba(7,22,39,.99),rgba(8,46,66,.98) 58%,rgba(5,77,87,.95));border:1px solid rgba(98,230,255,.22);box-shadow:0 22px 65px rgba(0,0,0,.33),inset 0 0 45px rgba(98,230,255,.025);margin-bottom:20px}.hero-grid{display:grid;grid-template-columns:1fr 280px;gap:20px;align-items:center;position:relative;z-index:2}.hero-title{font-size:44px;font-weight:950;letter-spacing:-1.2px;line-height:1;margin:0 0 8px}.hero-subtitle{color:#bed1e1;font-size:15px;margin-bottom:14px}.hero-badge{display:inline-flex;gap:7px;align-items:center;padding:7px 12px;border-radius:999px;background:rgba(98,230,255,.10);border:1px solid rgba(98,230,255,.25);color:#bff5ff;font-size:11px;font-weight:850;letter-spacing:.8px}.hero-orbit{width:235px;height:235px;border-radius:50%;border:1px solid rgba(98,230,255,.20);box-shadow:0 0 0 24px rgba(98,230,255,.025),0 0 0 55px rgba(98,230,255,.012);margin:auto;position:relative}.hero-orbit:before{content:"";position:absolute;inset:33px;border-radius:50%;border:1px dashed rgba(98,230,255,.20)}.hero-orbit:after{content:"";position:absolute;width:10px;height:10px;border-radius:50%;top:34px;left:50%;background:var(--mg-cyan);box-shadow:0 0 18px rgba(98,230,255,.9)}.hero-scanline{position:absolute;left:-15%;right:-15%;top:47%;height:1px;background:linear-gradient(90deg,transparent,rgba(98,230,255,.55),transparent);box-shadow:0 0 14px rgba(98,230,255,.35);transform:rotate(-7deg)}
.section-title{color:#edf7ff;font-size:25px;font-weight:900;letter-spacing:-.4px;margin:26px 0 10px}.subtle{color:#8ea5ba;font-size:13px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:12px 0 6px}.kpi{background:linear-gradient(180deg,#0b1a2c,#081320);border:1px solid rgba(98,230,255,.14);border-radius:18px;padding:18px 18px 16px;min-height:112px;box-shadow:0 12px 34px rgba(0,0,0,.19)}.kpi-label{color:#8ea5ba;text-transform:uppercase;letter-spacing:.8px;font-size:10px;font-weight:800}.kpi-value{font-size:31px;font-weight:950;letter-spacing:-.7px;margin-top:8px;color:#f0f8ff}.kpi-accent{color:var(--mg-cyan)}.kpi.good{border-color:rgba(87,229,161,.18)}.kpi.warn{border-color:rgba(255,209,102,.20)}.kpi.fail{border-color:rgba(255,100,119,.22)}
.status-ribbon{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px 15px;border-radius:16px;background:linear-gradient(90deg,rgba(87,229,161,.07),rgba(98,230,255,.04));border:1px solid rgba(87,229,161,.14);margin:10px 0 22px}.status-ribbon .big{font-size:18px;font-weight:900}.status-ribbon .small{color:#8ea5ba;font-size:12px}
.pipeline{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:12px 0 22px}.pipe-step{position:relative;padding:13px 10px;min-height:83px;border-radius:14px;background:#081522;border:1px solid rgba(98,230,255,.10);text-align:center}.pipe-step:not(:last-child):after{content:"";position:absolute;width:8px;height:1px;right:-8px;top:50%;background:rgba(98,230,255,.20)}.pipe-icon{font-size:18px}.pipe-name{margin-top:6px;font-size:11px;font-weight:850;color:#dbeaf6}.pipe-state{margin-top:4px;font-size:9px;text-transform:uppercase;letter-spacing:.6px;font-weight:850}.pipe-pass{color:var(--mg-green)}.pipe-warn{color:var(--mg-yellow)}.pipe-fail{color:var(--mg-red)}.pipe-muted{color:#71879b}
.validation-card{background:linear-gradient(180deg,#0a1727,#08121f);border:1px solid rgba(98,230,255,.12);border-radius:16px;padding:16px 17px;margin-bottom:11px;min-height:88px;transition:.16s ease}.validation-card:hover{transform:translateY(-2px);border-color:rgba(98,230,255,.28);box-shadow:0 15px 32px rgba(0,0,0,.22)}.status-pass{color:var(--mg-green);font-weight:900;text-shadow:0 0 12px rgba(87,229,161,.18)}.status-fail{color:var(--mg-red);font-weight:900}.status-warning{color:var(--mg-yellow);font-weight:900}.status-notrun{color:#768ca1;font-weight:900}
div[data-testid="stMetric"]{background:linear-gradient(180deg,#0b1a2c,#081320);border:1px solid rgba(98,230,255,.12);border-radius:17px;padding:15px 17px}div[data-testid="stMetricLabel"]{color:#8ea5ba!important}div[data-testid="stMetricValue"]{color:#f0f8ff!important;font-weight:950}.stButton>button{border-radius:12px;min-height:44px;font-weight:850;color:#e6f8ff;background:#0c1b2d;border:1px solid rgba(98,230,255,.18)}.stButton>button:hover{transform:translateY(-1px);border-color:rgba(98,230,255,.42);box-shadow:0 0 20px rgba(98,230,255,.09)}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#0b7e97,#235fd1);color:#fff;border:0;box-shadow:0 8px 25px rgba(35,95,209,.20)}div[data-testid="stDownloadButton"] button{background:linear-gradient(135deg,#0c819a,#235fcf)!important;color:#fff!important;border:0!important;box-shadow:0 8px 22px rgba(35,95,209,.20)}div[data-testid="stFileUploader"]{background:#071422;border:1px dashed rgba(98,230,255,.33);border-radius:16px;padding:8px}div[data-testid="stFileUploader"] button{background:#0c7893!important;color:#fff!important;border:0!important}div[data-testid="stFileUploader"] small,div[data-testid="stFileUploader"] section,div[data-testid="stFileUploader"] span{color:#a8bdd0!important}div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,textarea{background:#081522!important;color:#edf7ff!important;border-color:rgba(98,230,255,.18)!important}label,.stCheckbox label,.stRadio label,.stSelectbox label{color:#cbdbe8!important}.stAlert{border-radius:14px}div[data-testid="stExpander"]{background:#081522;border:1px solid rgba(98,230,255,.12);border-radius:14px;overflow:hidden}[data-testid="stDataFrame"]{border:1px solid rgba(98,230,255,.12);border-radius:14px;overflow:hidden}hr{border-color:rgba(98,230,255,.10)!important}
.score-panel{background:linear-gradient(180deg,#0b1b2c,#07131f);border:1px solid rgba(98,230,255,.14);border-radius:20px;padding:20px;min-height:215px;box-shadow:0 15px 40px rgba(0,0,0,.18)}.score-ring{width:138px;height:138px;border-radius:50%;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;background:conic-gradient(var(--mg-cyan) calc(var(--score)*1%),rgba(98,230,255,.10) 0);box-shadow:0 0 35px rgba(98,230,255,.08);position:relative}.score-ring:after{content:"";position:absolute;inset:11px;border-radius:50%;background:#081521}.score-ring span{position:relative;z-index:2;font-size:27px;font-weight:950}.score-caption{text-align:center;color:#8ea5ba;font-size:11px;text-transform:uppercase;letter-spacing:.8px}
@media (max-width:1100px){.hero-grid{grid-template-columns:1fr}.hero-orbit{display:none}.kpi-grid{grid-template-columns:repeat(2,1fr)}.pipeline{grid-template-columns:repeat(3,1fr)}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="command-bar">
      <div><span class="command-dot"></span>MODEL GUARD / CONTROL CENTER</div>
      <div>LOCAL RUNTIME • OFFLINE MODE • AUTOMATED VALIDATION</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def save_uploaded_file(uploaded, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded.name).name
    destination = destination_dir / safe_name
    destination.write_bytes(uploaded.getbuffer())
    return destination


def status_html(status: str) -> str:
    cls = {
        "PASS": "status-pass",
        "FAIL": "status-fail",
        "WARNING": "status-warning",
        "NOT RUN": "status-notrun",
        "SKIPPED": "status-notrun",
    }.get(status, "status-notrun")
    return f'<span class="{cls}">{status}</span>'


def get_configured_paths():
    model_path = st.session_state.get("model_path", DEFAULT_MODEL)
    dataset_path = st.session_state.get("dataset_path", DEFAULT_DATASET)
    return Path(model_path), Path(dataset_path)


def _file_fingerprint(path: Path | None) -> str:
    """Return a fingerprint that changes when an uploaded file is replaced."""
    if not path:
        return "NONE"
    try:
        stat = Path(path).stat()
        return f"{Path(path).resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        return f"MISSING|{Path(path).resolve()}"


def _validation_signature(model_path: Path, dataset_path: Path, label_column, allow_imputation, impute_strategy, reference_dataset_path):
    """Signature for the exact model/dataset/reference configuration being validated."""
    return "|".join(
        [
            _file_fingerprint(model_path),
            _file_fingerprint(dataset_path),
            str(label_column),
            str(bool(allow_imputation)),
            str(impute_strategy),
            _file_fingerprint(Path(reference_dataset_path) if reference_dataset_path else None),
        ]
    )


def _read_report_meta():
    if not REPORT_META_PATH.exists():
        return None
    try:
        return json.loads(REPORT_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def report_is_current(model_path=None, dataset_path=None, label_column=None, allow_imputation=True, impute_strategy="mean", reference_dataset_path=None) -> bool:
    """Return True only when the Word report belongs to the current inputs/configuration."""
    if not REPORT_PATH.exists():
        return False
    if model_path is None or dataset_path is None:
        model_path, dataset_path = get_configured_paths()

    current_sig = _validation_signature(
        Path(model_path), Path(dataset_path), label_column, allow_imputation, impute_strategy,
        Path(reference_dataset_path) if reference_dataset_path else None,
    )

    # Prefer the in-memory signature for the current Streamlit session.
    # This avoids a false "PENDING" display immediately after a successful
    # report generation when Streamlit's widget/session values rerun.
    session_report_status = st.session_state.get("report_status")
    session_report_sig = st.session_state.get("report_validation_signature")
    if session_report_status == "PASS" and session_report_sig == current_sig:
        return True

    meta = _read_report_meta()
    if not meta:
        return False

    return meta.get("validation_signature") == current_sig


def _write_report_meta(signature: str):
    REPORT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_META_PATH.write_text(
        json.dumps({
            "validation_signature": signature,
            "report_mtime_ns": REPORT_PATH.stat().st_mtime_ns if REPORT_PATH.exists() else None,
        }, indent=2),
        encoding="utf-8",
    )


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def read_dataset(path: Path) -> pd.DataFrame:
    """
    Cached CSV read. Streamlit reruns this whole script on every
    interaction, including a plain sidebar navigation click, and without
    caching that meant re-reading and re-parsing the CSV from disk on
    every single rerun regardless of which page was clicked. Keyed on
    (path, mtime) so a re-upload / edited file still invalidates it.
    """
    return _read_csv_cached(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _load_model_cached(path_str: str, mtime: float):
    return load_model(path_str)


def read_model(path: Path):
    """
    Cached model load. Avoids re-deserializing the joblib/pickle file on
    every rerun (e.g. every sidebar click), same rationale as
    read_dataset() above.
    """
    return _load_model_cached(str(path), path.stat().st_mtime)


def load_last_results():
    return st.session_state.get("validation_results")


def read_ui_status():
    if not UI_STATUS_PATH.exists():
        return {"status": "NOT RUN", "message": "Selenium UI validation has not been run yet."}
    try:
        return json.loads(UI_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "NOT RUN", "message": "UI status file could not be read."}


# True only for the browser session that Selenium itself is driving (see
# tests/test_ui_selenium.py, which appends ?selenium_driver=1 to its first
# page load). Used to stop that automated session from chaining into
# launching ANOTHER Selenium subprocess when it reaches the Validation page
# -- without this guard, each Selenium run spawns another one recursively.
IS_SELENIUM_DRIVEN_SESSION = st.query_params.get("selenium_driver") == "1"


with st.sidebar:
    st.markdown("<div style='font-size:28px;font-weight:850;letter-spacing:-.5px'>🛡️ ModelGuard</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;opacity:.72;margin-top:-3px;margin-bottom:16px'>AUTOMATED ML TESTING</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Dataset", "Model", "Validation", "Reports", "Logs"],
        label_visibility="visible",
    )
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.14);margin:15px 0'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;letter-spacing:.8px;opacity:.65;font-weight:800'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.success("● System Ready")
    st.caption(f"Python {sys.version_info.major}.{sys.version_info.minor}  •  Streamlit  •  PyTest")
    if IS_SELENIUM_DRIVEN_SESSION:
        st.warning("🤖 Selenium-driven session (auto Selenium chaining disabled)")

st.markdown(
    """
<div class="command-strip">
  <div class="command-left"><span class="live-dot"></span><span>ModelGuard Sentinel Console</span><span class="command-chip">LOCAL EXECUTION</span></div>
  <div>ML ASSURANCE / SECURITY TESTING</div>
</div>
<div class="hero">
  <div class="hero-grid">
    <div>
      <div class="hero-title">🛡️ ModelGuard</div>
      <div class="hero-subtitle">ML Model Validation & Testing Framework</div>
      <div class="hero-badge">● AUTOMATED <span>•</span> OFFLINE-READY <span>•</span> SECURITY-AWARE</div>
    </div>
    <div class="hero-orbit"><div class="hero-scanline"></div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":
    model_path, dataset_path = get_configured_paths()
    results = load_last_results()
    ui_status = read_ui_status()

    model_status = "PASS" if model_path.exists() else "FAIL"
    dataset_status = "PASS" if dataset_path.exists() else "FAIL"

    passed = failed = warnings = skipped = 0
    if results:
        statuses = [r.get("status", "NOT RUN") for r in results.values()]
        passed = statuses.count("PASS")
        failed = statuses.count("FAIL")
        warnings = statuses.count("WARNING")
        skipped = statuses.count("SKIPPED")

    final = "NOT RUN"
    if results:
        final = overall_status(results)
        if ui_status.get("status") == "FAIL":
            final = "FAIL"
        elif ui_status.get("status") != "PASS":
            final = "NOT COMPLETE"
        elif final == "PASS":
            final = "PASS"

    total_core = len(results) if results else 0
    score = int(round((passed / total_core) * 100)) if total_core else 0
    record_count = 0
    if dataset_path.exists():
        try:
            record_count = len(read_dataset(dataset_path))
        except Exception:
            record_count = 0

    st.markdown('<div class="section-title">◈ Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Live snapshot of model readiness, data health and automated assurance coverage.</div>', unsafe_allow_html=True)

    model_text = "READY" if model_status == "PASS" else "MISSING"
    dataset_text = "READY" if dataset_status == "PASS" else "MISSING"
    status_cls = "good" if final == "PASS" else ("warn" if final == "WARNING" else ("fail" if final == "FAIL" else ""))
    st.markdown(
        f"""<div class="kpi-grid">
          <div class="kpi good"><div class="kpi-label">Model Asset</div><div class="kpi-value">{model_text}</div><div class="subtle">joblib / pickle ready</div></div>
          <div class="kpi good"><div class="kpi-label">Dataset</div><div class="kpi-value">{dataset_text}</div><div class="subtle">{record_count} records loaded</div></div>
          <div class="kpi"><div class="kpi-label">Validation Coverage</div><div class="kpi-value">{passed}/{total_core}</div><div class="subtle">{warnings} warning · {skipped} skipped · {failed} failed</div></div>
          <div class="kpi {status_cls}"><div class="kpi-label">Overall Decision</div><div class="kpi-value">{final}</div><div class="subtle">Selenium UI: {ui_status.get('status','NOT RUN')}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    ribbon_color = "#57e5a1" if final == "PASS" else ("#ffd166" if final == "WARNING" else "#ff6477")
    ribbon_copy = "READY FOR REVIEW" if final == "PASS" else ("REVIEW WARNINGS BEFORE RELEASE" if final == "WARNING" else "ACTION REQUIRED")
    report_word = "AVAILABLE" if report_is_current(
        model_path=model_path,
        dataset_path=dataset_path,
        label_column=st.session_state.get("label_column"),
        allow_imputation=st.session_state.get("allow_imputation", True),
        impute_strategy=st.session_state.get("impute_strategy", "mean"),
        reference_dataset_path=st.session_state.get("reference_dataset_path"),
    ) else "PENDING"
    st.markdown(
        f"""<div class="status-ribbon" style="border-color:{ribbon_color}33">
          <div><div class="big" style="color:{ribbon_color}">● {ribbon_copy}</div><div class="small">Core automated checks completed in the local ModelGuard environment.</div></div>
          <div class="command-chip">REPORT • {report_word}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    pipeline_map = [
        ("📥","INGEST","Dataset Upload & Validation"),
        ("🧹","PREP","Data Imputation"),
        ("⚙️","INFER","Model Inference"),
        ("🛡️","RESILIENCE","Robustness Testing"),
        ("🔐","SECURITY","Data Poisoning Detection"),
        ("📄","REPORT","Automated Word Report"),
    ]
    p_html = []
    for icon, label, key in pipeline_map:
        if key == "Automated Word Report":
            item = {
                "status": "PASS" if report_is_current(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    label_column=st.session_state.get("label_column"),
                    allow_imputation=st.session_state.get("allow_imputation", True),
                    impute_strategy=st.session_state.get("impute_strategy", "mean"),
                    reference_dataset_path=st.session_state.get("reference_dataset_path"),
                ) else "NOT RUN"
            }
        else:
            item = results.get(key, {}) if results else {}
        stt = item.get("status", "NOT RUN")
        state_cls = "pipe-pass" if stt == "PASS" else ("pipe-warn" if stt == "WARNING" else ("pipe-fail" if stt == "FAIL" else "pipe-muted"))
        p_html.append(f'<div class="pipe-step"><div class="pipe-icon">{icon}</div><div class="pipe-name">{label}</div><div class="pipe-state {state_cls}">{stt}</div></div>')
    st.markdown('<div class="pipeline">' + ''.join(p_html) + '</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""<div class="score-panel">
              <div class="kpi-label" style="text-align:center;margin-bottom:9px">VALIDATION COVERAGE</div>
              <div class="score-ring" style="--score:{score}"><span>{score}%</span></div>
              <div class="score-caption">PASS stages / reported core stages</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with right:
        ref = st.session_state.get("reference_dataset_path")
        st.markdown('<div class="card"><div class="kpi-label">LIVE TELEMETRY</div>', unsafe_allow_html=True)
        telemetry = [
            ("Model", model_path.name if model_path.exists() else "Not configured"),
            ("Validation dataset", dataset_path.name if dataset_path.exists() else "Not configured"),
            ("Reference baseline", Path(ref).name if ref else "Not supplied"),
            ("UI automation", ui_status.get("status", "NOT RUN")),
        ]
        for label, value in telemetry:
            st.markdown(f'<div style="display:flex;justify-content:space-between;gap:18px;padding:10px 0;border-bottom:1px solid rgba(98,230,255,.07)"><span class="subtle">{label}</span><span style="font-weight:800;color:#e6f6ff;text-align:right">{html_escape(str(value))}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">⌁ Validation Matrix</div>', unsafe_allow_html=True)
    if results:
        display_results = dict(results)
    else:
        display_results = {
            "Model Loading": {"status": "NOT RUN"},
            "Dataset Upload & Validation": {"status": "NOT RUN"},
            "Data Imputation": {"status": "NOT RUN"},
            "Preprocessing Testing": {"status": "NOT RUN"},
            "Model Inference": {"status": "NOT RUN"},
            "Result Validation": {"status": "NOT RUN"},
            "Robustness Testing": {"status": "NOT RUN"},
            "Out-of-Distribution Detection": {"status": "NOT RUN"},
            "Distribution Shift Detection": {"status": "NOT RUN"},
            "Data Poisoning Detection": {"status": "NOT RUN"},
            "Model Poisoning Detection": {"status": "NOT RUN"},
            "Selenium UI Validation": {"status": ui_status.get("status", "NOT RUN")},
            "Automated Word Report": {"status": "NOT RUN"},
        }

    items = list(display_results.items())
    for i in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (name, item) in zip(cols, items[i:i + 3]):
            with col:
                status = item.get("status", "NOT RUN")
                details = item.get("details") or []
                icon = "✓" if status == "PASS" else ("!" if status == "WARNING" else ("×" if status == "FAIL" else "•"))

                with st.container(border=True):
                    top_left, top_right = st.columns([8, 1])
                    with top_left:
                        st.markdown(f"**{html_escape(name)}**")
                    with top_right:
                        st.markdown(
                            f'<div style="text-align:right;color:#8096aa;font-weight:900">{icon}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(status_html(status), unsafe_allow_html=True)

                    if status in ("FAIL", "WARNING") and details:
                        st.caption(str(details[0]))

    failed_stages = [name for name, item in display_results.items() if item.get("status") == "FAIL"]
    warning_stages = [name for name, item in display_results.items() if item.get("status") == "WARNING"]
    if failed_stages:
        st.error(f"{len(failed_stages)} core stage(s) failed: " + ", ".join(failed_stages) + ". Open the Validation page for the full reason.")
    elif warning_stages:
        st.warning(f"{len(warning_stages)} core stage(s) returned warnings: " + ", ".join(warning_stages) + ".")
    elif results and ui_status.get("status") != "PASS":
        st.info("Core validation passed. Run Selenium UI validation to complete the end-to-end workflow.")


# ==================================================
# DATASET
# ==================================================
elif page == "Dataset":
    st.markdown('<div class="section-title">📁 Dataset Management</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload test CSV dataset", type=["csv"], key="dataset_upload")

    if uploaded is not None:
        try:
            path = save_uploaded_file(uploaded, UPLOAD_DIR)
            df = read_dataset(path)
            st.session_state.dataset_path = path
            logger.info("Dataset uploaded: %s", path)
            st.success(f"Dataset uploaded: {path.name}")
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
            df = None
    else:
        model_path, dataset_path = get_configured_paths()
        df = read_dataset(dataset_path) if dataset_path.exists() else None
        if df is not None:
            st.info(f"Using configured dataset: {dataset_path}")

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isna().sum().sum()))
        c4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), width='stretch')

        st.markdown("### Label Column")
        candidates = ["None"] + list(df.columns)
        current = st.session_state.get("label_column")
        default_index = candidates.index(current) if current in candidates else (
            candidates.index(find_label_column(df)) if find_label_column(df) in candidates else 0
        )
        selected = st.selectbox(
            "Select ground-truth label column (choose None if the dataset is unlabeled)",
            candidates,
            index=default_index,
        )
        st.session_state.label_column = None if selected == "None" else selected

        st.markdown("### Dataset Validation Preview")
        if df.empty:
            st.error("Dataset is empty.")
        elif df.duplicated().sum():
            st.warning(f"{int(df.duplicated().sum())} duplicate rows detected.")
        elif df.isna().sum().sum():
            st.warning(f"{int(df.isna().sum().sum())} missing values detected.")
        else:
            st.success("Basic dataset checks passed.")

        st.markdown("### Training / Reference Dataset (optional, strongly recommended)")
        st.caption(
            "Upload the original dataset the model was trained on. When provided, it is used "
            "for three things: (1) Out-of-Distribution Detection, (2) Distribution Shift "
            "Detection — both SKIPPED without it — and (3) as the source of statistics "
            "(mean/median/most-frequent/KNN neighbors) for Missing Value Handling below, "
            "instead of deriving fill values from the dataset being validated itself."
        )
        reference_upload = st.file_uploader(
            "Upload training/reference CSV dataset",
            type=["csv"],
            key="reference_dataset_upload",
        )
        if reference_upload is not None:
            try:
                ref_path = save_uploaded_file(reference_upload, UPLOAD_DIR / "reference")
                st.session_state.reference_dataset_path = ref_path
                st.success(f"Training/reference dataset uploaded: {ref_path.name}")
            except Exception as exc:
                st.error(f"Could not read training/reference dataset: {exc}")
        elif st.session_state.get("reference_dataset_path"):
            st.info(f"Using training/reference dataset: {st.session_state.reference_dataset_path}")
            if st.button("Clear training/reference dataset"):
                del st.session_state["reference_dataset_path"]

        st.markdown("### Missing Value Handling")
        allow_imputation = st.checkbox(
            "Automatically impute missing values instead of failing validation",
            value=st.session_state.get("allow_imputation", True),
            key="allow_imputation",
        )
        if allow_imputation:
            st.selectbox(
                "Imputation strategy",
                ["mean", "median", "most_frequent", "knn"],
                index=["mean", "median", "most_frequent", "knn"].index(
                    st.session_state.get("impute_strategy", "mean")
                ),
                key="impute_strategy",
                help="mean/median/most_frequent use per-column statistics; "
                "knn imputes each missing value from its 5 nearest complete rows.",
            )
            if st.session_state.get("reference_dataset_path"):
                st.caption(
                    "A training/reference dataset is set above, so imputation will use its "
                    "statistics rather than the validation dataset's own values, wherever "
                    "the feature columns match."
                )
            else:
                st.caption(
                    "No training/reference dataset is set above, so imputation will fall back "
                    "to statistics computed from the dataset being validated itself."
                )


# ==================================================
# MODEL
# ==================================================
elif page == "Model":
    st.markdown('<div class="section-title">🤖 Model Management</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload ML model (.joblib, .pkl or .pickle)",
        type=["joblib", "pkl", "pickle"],
        key="model_upload",
    )

    if uploaded is not None:
        try:
            path = save_uploaded_file(uploaded, UPLOAD_DIR)
            model = read_model(path)
            st.session_state.model_path = path
            logger.info("Model uploaded and loaded: %s", path)
            st.success(f"Model loaded successfully: {path.name}")

            st.write(f"Model type: `{type(model).__name__}`")
            st.write(f"Has predict(): `{hasattr(model, 'predict')}`")
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is not None:
                st.write("Expected features:", list(feature_names))
        except Exception as exc:
            st.error(f"Model could not be loaded: {exc}")
    else:
        model_path, _ = get_configured_paths()
        if model_path.exists():
            try:
                model = read_model(model_path)
                st.info(f"Using configured model: {model_path}")
                st.write(f"Model type: `{type(model).__name__}`")
                st.write(f"Has predict(): `{hasattr(model, 'predict')}`")
            except Exception as exc:
                st.error(f"Configured model could not be loaded: {exc}")


# ==================================================
# VALIDATION
# ==================================================
elif page == "Validation":
    st.markdown('<div class="section-title">🧪 Model Validation</div>', unsafe_allow_html=True)

    model_path, dataset_path = get_configured_paths()

    st.write(f"Model: `{model_path}`")
    st.write(f"Dataset: `{dataset_path}`")

    if not model_path.exists() or not dataset_path.exists():
        st.warning("Please provide a valid model and dataset first.")
    else:
        df = read_dataset(dataset_path)
        inferred = find_label_column(df, st.session_state.get("label_column"))
        label_options = ["None"] + list(df.columns)
        selected_index = label_options.index(inferred) if inferred in label_options else 0

        label = st.selectbox(
            "Ground-truth label column",
            label_options,
            index=selected_index,
            key="validation_label",
        )
        label_column = None if label == "None" else label
        st.session_state.label_column = label_column

        allow_imputation = st.session_state.get("allow_imputation", True)
        impute_strategy = st.session_state.get("impute_strategy", "mean")
        reference_dataset_path = st.session_state.get("reference_dataset_path")

        if reference_dataset_path:
            st.caption(f"Reference dataset for OOD / distribution shift checks: `{reference_dataset_path}`")
        else:
            st.caption(
                "No reference dataset set — upload one on the Dataset page to enable "
                "Distribution Shift Detection and stronger Out-of-Distribution Detection."
            )

        auto_run = st.checkbox(
            "Automatically run validation (including the Selenium UI test) as soon as a "
            "model and dataset are ready — no button click needed",
            value=False if IS_SELENIUM_DRIVEN_SESSION else st.session_state.get("auto_run_validation", True),
            key="auto_run_validation",
            disabled=IS_SELENIUM_DRIVEN_SESSION,
            help=(
                "Disabled in this Selenium-driven session to avoid the automated "
                "browser test re-triggering another Selenium test recursively."
                if IS_SELENIUM_DRIVEN_SESSION
                else None
            ),
        )

        def _run_selenium_ui_validation():
            env = os.environ.copy()
            env["SELENIUM_BASE_URL"] = "http://localhost:8501"
            env["MODEL_PATH"] = str(model_path.resolve())
            env["DATASET_PATH"] = str(dataset_path.resolve())
            env["PYTHONPATH"] = str(ROOT)
            # Offline-only Selenium: never ask Selenium Manager to download a driver.
            # Resolution order: explicit env var -> project drivers/ folder -> fixed
            # local install path used on this machine.
            _project_driver = ROOT / "drivers" / "chromedriver.exe"
            _fixed_driver = Path(r"C:\WebDriver\chromedriver.exe")
            if os.environ.get("CHROMEDRIVER_PATH"):
                local_driver = os.environ["CHROMEDRIVER_PATH"]
            elif _project_driver.is_file():
                local_driver = str(_project_driver)
            else:
                local_driver = str(_fixed_driver)
            env["CHROMEDRIVER_PATH"] = local_driver
            logger.info("Starting Selenium UI validation.")
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "pytest", "-m", "ui", "-q"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                st.session_state.selenium_last_output = completed.stdout + "\n" + completed.stderr
                st.session_state.selenium_last_passed = completed.returncode == 0
                if completed.returncode == 0:
                    logger.info("Selenium UI validation passed.")
                else:
                    logger.error("Selenium UI validation failed.")
            except Exception as exc:
                logger.exception("Selenium validation could not start.")
                st.session_state.selenium_last_output = f"Could not run Selenium validation: {exc}"
                st.session_state.selenium_last_passed = False

        def _run_full_validation():
            logger.info("Validation started. Model=%s Dataset=%s", model_path, dataset_path)
            validation_sig = _validation_signature(
                model_path, dataset_path, label_column, allow_imputation, impute_strategy, reference_dataset_path
            )
            try:
                results = run_validation_pipeline(
                    model_path,
                    dataset_path,
                    label_column,
                    allow_imputation=allow_imputation,
                    impute_strategy=impute_strategy,
                    reference_dataset_path=reference_dataset_path,
                )
                st.session_state.validation_results = results

                # Create the Word report from the real results.
                report_status = "PASS"
                try:
                    generate_report(
                        results,
                        REPORT_PATH,
                        model_path=model_path,
                        dataset_path=dataset_path,
                        label_column=label_column,
                    )
                    _write_report_meta(validation_sig)
                    st.session_state.report_validation_signature = validation_sig
                    logger.info("Word report generated: %s", REPORT_PATH)
                except Exception as exc:
                    report_status = "FAIL"
                    logger.exception("Report generation failed.")
                    st.session_state.report_generation_error = str(exc)

                st.session_state.report_status = report_status
                logger.info("Validation completed. Overall core status=%s", overall_status(results))
            except Exception as exc:
                logger.exception("Validation pipeline failed.")
                st.session_state.pipeline_error = str(exc)
                return

            # Chain straight into the Selenium UI test — no separate click required.
            # Never do this in a session that Selenium itself is driving, or each
            # Selenium run would recursively spawn another one.
            if not IS_SELENIUM_DRIVEN_SESSION:
                _run_selenium_ui_validation()

        # A fingerprint of the actual files plus validation settings. Including
        # file modification time/size is important when a user uploads a new file
        # with the same filename as the previous one.
        run_signature = _validation_signature(
            model_path, dataset_path, label_column, allow_imputation, impute_strategy, reference_dataset_path
        )

        manual_trigger = st.button("🚀 Run ModelGuard Validation (+ Selenium UI test)", width='stretch')

        should_run = manual_trigger or (
            auto_run and st.session_state.get("last_auto_run_signature") != run_signature
        )

        if should_run:
            st.session_state.pop("pipeline_error", None)
            st.session_state.pop("report_generation_error", None)
            with st.spinner(
                "Running ModelGuard validation, then the automated Selenium UI test..."
            ):
                _run_full_validation()
            st.session_state.last_auto_run_signature = run_signature
            st.rerun()

        if st.session_state.get("pipeline_error"):
            st.error(f"Validation failed: {st.session_state.pipeline_error}")
        if st.session_state.get("report_generation_error"):
            st.error(f"Report generation failed: {st.session_state.report_generation_error}")

        results = st.session_state.get("validation_results")

        if results:
            st.markdown("### Validation Results")
            for name, item in results.items():
                status = item["status"]
                with st.expander(f"{name} — {status}", expanded=(status != "PASS")):
                    for detail in item.get("details", []):
                        st.write(f"• {detail}")

                    if item.get("metrics"):
                        st.write("Metrics:", item["metrics"])

            st.markdown("### Automated Word Report")
            report_current = report_is_current(
                model_path=model_path,
                dataset_path=dataset_path,
                label_column=label_column,
                allow_imputation=allow_imputation,
                impute_strategy=impute_strategy,
                reference_dataset_path=reference_dataset_path,
            )
            report_status = "PASS" if report_current else st.session_state.get("report_status", "NOT RUN")
            st.write(status_html(report_status), unsafe_allow_html=True)

            st.markdown("### Selenium UI Validation")
            ui_status = read_ui_status()
            st.write(
                f"Current UI test status: {status_html(ui_status['status'])}",
                unsafe_allow_html=True,
            )

            if "selenium_last_output" in st.session_state:
                st.code(st.session_state.selenium_last_output, language="text")
                if st.session_state.selenium_last_passed:
                    st.success("Selenium UI validation passed.")
                else:
                    st.error("Selenium UI validation failed. Check the output above.")

            if st.button("🌐 Re-run Selenium UI Validation only", width='stretch'):
                with st.spinner("Running the automated Selenium UI test..."):
                    _run_selenium_ui_validation()
                st.rerun()

            st.info(
                "Selenium is configured for offline execution. A local Chrome/ChromeDriver "
                "installation is required; no internet download is used."
            )


# ==================================================
# REPORTS
# ==================================================
elif page == "Reports":
    st.markdown('<div class="section-title">📄 Reports</div>', unsafe_allow_html=True)

    report_model_path, report_dataset_path = get_configured_paths()
    report_label = st.session_state.get("label_column")
    report_allow_imputation = st.session_state.get("allow_imputation", True)
    report_impute_strategy = st.session_state.get("impute_strategy", "mean")
    report_reference = st.session_state.get("reference_dataset_path")
    report_current = report_is_current(
        model_path=report_model_path,
        dataset_path=report_dataset_path,
        label_column=report_label,
        allow_imputation=report_allow_imputation,
        impute_strategy=report_impute_strategy,
        reference_dataset_path=report_reference,
    )

    if report_current:
        st.success("Current ModelGuard test report is available for the active model/dataset/reference configuration.")
        report_mtime = REPORT_PATH.stat().st_mtime_ns
        with open(REPORT_PATH, "rb") as file:
            report_bytes = file.read()
        st.download_button(
            "⬇️ Download Current Test Report",
            data=report_bytes,
            file_name="ModelGuard_Test_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width='stretch',
            key=f"download_report_{report_mtime}",
        )
    elif REPORT_PATH.exists():
        st.warning(
            "A previous report exists, but it belongs to an older model/dataset/reference configuration. "
            "Run ModelGuard validation again to generate the current report before downloading."
        )
    else:
        st.info("Run ModelGuard validation to generate the Word report.")


# ==================================================
# LOGS
# ==================================================
elif page == "Logs":
    st.markdown('<div class="section-title">📝 System Logs</div>', unsafe_allow_html=True)

    log_path = ROOT / "logs" / "modelguard.log"
    if log_path.exists():
        st.code(log_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("No log entries yet. Run a validation to create the log.")
