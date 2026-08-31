"""Presentation constants and CSS for the underwriting console."""

BRAND = "Meridian Credit"
PRODUCT = "Underwriting Console"

# Semantic colours for decision bands.
BAND = {
    "LOW": {
        "color": "#0f7b34",
        "bg": "#eaf6ed",
        "border": "#0f7b34",
        "decision": "Approve",
        "detail": "Within standard risk appetite. Proceed on standard terms.",
    },
    "MEDIUM": {
        "color": "#8a5a00",
        "bg": "#fdf5e4",
        "border": "#c98a12",
        "decision": "Refer to underwriter",
        "detail": "Outside automatic approval. Manual review required before offer.",
    },
    "HIGH": {
        "color": "#a3232b",
        "bg": "#fbeced",
        "border": "#a3232b",
        "decision": "Decline",
        "detail": "Exceeds risk appetite. Adverse action notice required if declined.",
    },
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1150px;}

.mc-header {
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid #dfe4ea; padding-bottom:.9rem; margin-bottom:1.6rem;
}
.mc-brand {font-size:1.35rem; font-weight:700; color:#1f4e79; letter-spacing:-.01em;}
.mc-brand span {font-weight:400; color:#5b6672; margin-left:.55rem; font-size:1rem;}
.mc-env {
  font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
  background:#eef2f7; color:#41556b; padding:.28rem .6rem; border-radius:4px;
  border:1px solid #d7dee7;
}

.mc-section {
  font-size:.76rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.09em; color:#5b6672; margin:.4rem 0 .5rem;
}

.mc-decision {border-radius:10px; padding:1.4rem 1.6rem; margin:.4rem 0 1rem;}
.mc-decision .verdict {font-size:1.85rem; font-weight:700; line-height:1.15;}
.mc-decision .band {font-size:.78rem; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; opacity:.85; margin-bottom:.35rem;}
.mc-decision .detail {margin-top:.55rem; font-size:.94rem; color:#3d4854;}

.mc-ref {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.8rem; color:#5b6672; background:#f4f6f9;
  border:1px solid #e3e8ee; border-radius:6px; padding:.55rem .75rem;
}
.mc-fact {border-left:3px solid #d7dee7; padding:.15rem 0 .15rem .7rem; margin:.35rem 0;}
.mc-fact b {color:#1c2530;}
.mc-note {font-size:.83rem; color:#6a7480;}
</style>
"""
