import html as htmllib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"
MIN_GAP = 13        # seconds between requests (free tier allows about 5 a minute)
DAILY_LIMIT = 20    # free tier requests per day for this model (check ai.dev/rate-limit)
DEMO_FILE = Path(__file__).with_name("demo_result.json")
SHARED_DAILY_BUDGET = 16   # requests per day we allow on a shared key (keep a few spare)
SESSION_RUN_CAP = 3        # runs per visit on a shared key

AGENTS = [
    {"name": "Market analyst", "focus": "demand, customers and competitors"},
    {"name": "Finance lead", "focus": "unit economics, costs and break-even logic"},
    {"name": "Operations", "focus": "execution, capability, delivery and timing"},
    {"name": "Risk officer", "focus": "what could go wrong and how badly"},
]

DRIVER_KEYS = ["upfront_cost", "monthly_benefit", "monthly_cost", "ramp_months"]
TAGS = ("Sourced", "Estimated", "Assumption")

# ---------- prompts ----------

RESEARCH_PROMPT = """Use Google Search to find current, concrete facts that matter for this business decision: <<QUESTION>>
Write 6 to 8 short factual statements, one per line. Prefer numbers, benchmarks, dates, market sizes, typical costs and rates, and recent developments. Only state what the search results support. No opinions and no recommendations."""

SPECIALIST_SHAPE = """{"finding": "2 or 3 specific sentences from your angle only", "key_assumption": "the one assumption your finding depends on", "tag": "Sourced, Estimated or Assumption", "source_ids": ["F1"]}"""

SPECIALIST_RULES = """Rules for specialists: use the tag "Sourced" only if the finding relies on a research fact above, and then list its id in source_ids. Never invent an id and never invent statistics. If no fact is used, use "Estimated" or "Assumption" and leave source_ids empty. Each specialist speaks only from their own angle, is specific to this question and takes a clear position. Specialists should not simply agree with each other."""

MODEL_SHAPE = """{"unit": "...", "horizon_months": 12,
 "upfront_cost": {"label": "what this one-time cost is", "worst": 0, "base": 0, "best": 0, "why": "one sentence", "tag": "Sourced, Estimated or Assumption", "source_ids": []},
 "monthly_benefit": {"label": "extra monthly profit or savings once fully running", "worst": 0, "base": 0, "best": 0, "why": "one sentence", "tag": "Sourced, Estimated or Assumption", "source_ids": []},
 "monthly_cost": {"label": "extra monthly running cost", "worst": 0, "base": 0, "best": 0, "why": "one sentence", "tag": "Sourced, Estimated or Assumption", "source_ids": []},
 "ramp_months": {"label": "months until the benefit reaches full size", "worst": 0, "base": 0, "best": 0, "why": "one sentence", "tag": "Sourced, Estimated or Assumption", "source_ids": []}}"""

MODEL_RULES = """Rules for the model: choose one currency unit for the question and put it in "unit" (for example "INR lakh" for an Indian business, otherwise "USD thousand"). "worst" is the value that is worst for the decision (highest for costs, lowest for benefits, longest for ramp months) and "best" is the most favourable. Use plain numbers only. horizon_months must be 12, 24 or 36. Use "Sourced" only when the number comes directly from a research fact, and list that fact id in source_ids. Never invent an id. Use "Estimated" only if the number follows from typical industry ranges, otherwise "Assumption". If the size of the business is unknown, assume a small to mid-sized business and say so in the "why" of upfront_cost."""

SPECIALIST_PROMPT = """You are the <<NAME>> on a consulting board. Your focus: <<FOCUS>>.
Decision question: <<QUESTION>>
Research facts found on the web (each has an id):
<<FACTS>>
Return ONLY valid JSON with exactly these keys:
<<SPECIALIST_SHAPE>>
<<SPECIALIST_RULES>>"""

ECON_PROMPT = """You are the finance modeller on a consulting board.
Decision question: <<QUESTION>>
Specialist findings:
<<FINDINGS>>
Research facts found on the web (each has an id):
<<FACTS>>
Build a simple 4-driver model of doing what the question asks, compared with not doing it.
Return ONLY valid JSON with exactly this shape:
<<MODEL_SHAPE>>
<<MODEL_RULES>>"""

BOARD_PROMPT = """You are a consulting board of four specialists and a finance modeller, working on one decision.
Decision question: <<QUESTION>>
Research facts found on the web (each has an id):
<<FACTS>>
Step 1. Each specialist gives a finding from their own angle only:
- Market analyst: demand, customers and competitors
- Finance lead: unit economics, costs and break-even logic
- Operations: execution, capability, delivery and timing
- Risk officer: what could go wrong and how badly
Step 2. The finance modeller builds a simple 4-driver model of doing what the question asks, compared with not doing it.
Return ONLY valid JSON with exactly this shape:
{"specialists": {"Market analyst": <S>, "Finance lead": <S>, "Operations": <S>, "Risk officer": <S>}, "model": <M>}
where every <S> has this shape:
<<SPECIALIST_SHAPE>>
and <M> has this shape:
<<MODEL_SHAPE>>
<<SPECIALIST_RULES>>
<<MODEL_RULES>>"""

CLOSE_PROMPT = """You are, first, the devil's advocate and then the managing partner of a consulting board.
Decision question: <<QUESTION>>
Specialist findings:
<<FINDINGS>>
Computed financial results (done by code, treat as the base for the memo):
<<NUMBERS>>
Step 1. As the devil's advocate, attack the weakest claim or number.
Step 2. As the managing partner, write the final memo, taking the challenge into account.
Return ONLY valid JSON with exactly this shape:
{"devil": {"target": "name of the specialist or model input you challenge", "challenge": "2 sentences on why it may be wrong", "evidence_needed": "one sentence on what evidence would settle it"},
 "memo": {"answer": "the recommendation in two sentences, consistent with the computed numbers", "verdict": "exactly one word: go, pilot or pause", "numbers_comment": "one sentence interpreting the payback and the downside case", "disagreement": "2 sentences on where the board disagreed and how you resolved it", "risks": ["risk 1", "risk 2", "risk 3"], "what_would_change_this": "one sentence", "next_steps": ["step 1", "step 2", "step 3"]}}
Rules: do not invent statistics. Do not contradict the computed numbers."""

CONSTANTS = {
    "SPECIALIST_SHAPE": SPECIALIST_SHAPE,
    "SPECIALIST_RULES": SPECIALIST_RULES,
    "MODEL_SHAPE": MODEL_SHAPE,
    "MODEL_RULES": MODEL_RULES,
}


def fill(text, **values):
    for k, v in CONSTANTS.items():
        text = text.replace(f"<<{k}>>", v)
    for k, v in values.items():
        text = text.replace(f"<<{k}>>", v)
    return text


def get_secret(name):
    try:
        return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def get_key():
    return get_secret("GEMINI_API_KEY")


PUBLIC_MODE = get_secret("PUBLIC_MODE") == "1"


@st.cache_resource
def global_usage():
    return {"day": "", "calls": 0}


def usage_today():
    """Requests sent through this server today, shared by all visitors (resets if the app restarts)."""
    u = global_usage()
    today = (datetime.now(timezone.utc) - timedelta(hours=7)).date().isoformat()  # roughly Pacific time
    if u["day"] != today:
        u["day"] = today
        u["calls"] = 0
    return u


# ---------- talking to Gemini, politely ----------

class QuotaError(Exception):
    pass


def is_quota(msg):
    m = msg.lower()
    return "429" in m or "resource_exhausted" in m or "quota" in m


def is_daily(msg):
    m = msg.lower()
    return "perday" in m or "per day" in m or "daily" in m


def pace(note=None):
    last = st.session_state.get("last_call", 0.0)
    wait = MIN_GAP - (time.time() - last)
    if wait > 0:
        if note:
            note(f"Pausing {int(wait) + 1}s to stay inside the free-tier limit of about 5 requests a minute...")
        time.sleep(wait)


def ask(client, prompt, temperature=0.5, search=False, patient=True, note=None):
    last_error = None
    for attempt in range(3):
        pace(note)
        st.session_state["calls"] = st.session_state.get("calls", 0) + 1
        usage_today()["calls"] += 1
        st.session_state["last_call"] = time.time()
        try:
            if search:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=temperature,
                )
            else:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=temperature
                )
            return client.models.generate_content(model=MODEL, contents=prompt, config=config)
        except Exception as e:
            last_error = e
            msg = str(e)
            if is_quota(msg):
                if is_daily(msg) or not patient or attempt >= 1:
                    raise QuotaError(msg)
                if note:
                    note("Per-minute limit reached. Waiting 65 seconds, then trying once more...")
                time.sleep(65)
            else:
                time.sleep(5 * (attempt + 1))
    raise last_error


def ask_json(client, prompt, temperature=0.5, note=None):
    for _ in range(2):
        reply = ask(client, prompt, temperature, note=note)
        try:
            return json.loads(reply.text)
        except Exception:
            continue
    raise ValueError("The AI returned something unreadable twice. Please run again.")


# ---------- live research (Google Search through Gemini) ----------

def run_research(client, question, note=None):
    try:
        reply = ask(
            client, fill(RESEARCH_PROMPT, QUESTION=question), temperature=0.2,
            search=True, patient=False, note=note,
        )
        cand = reply.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        chunks = getattr(gm, "grounding_chunks", None) or []
        supports = getattr(gm, "grounding_supports", None) or []

        srcs = []
        for c in chunks:
            w = getattr(c, "web", None)
            srcs.append({
                "title": (getattr(w, "title", "") or "source"),
                "uri": (getattr(w, "uri", "") or ""),
            })

        facts, seen = [], set()
        for s in supports:
            seg = getattr(s, "segment", None)
            text = (getattr(seg, "text", "") or "").strip()
            idx = list(getattr(s, "grounding_chunk_indices", None) or [])
            links = [srcs[i] for i in idx if 0 <= i < len(srcs) and srcs[i]["uri"]]
            if text and links and text not in seen:
                seen.add(text)
                facts.append({"text": text, "links": links})
        facts = facts[:10]
        for i, f in enumerate(facts, 1):
            f["id"] = f"F{i}"
        if not facts:
            return [], "The search ran but returned no facts with sources."
        return facts, ""
    except Exception as e:
        return [], str(e)[:250]


def facts_block(facts):
    if not facts:
        return "None available. You have no live web access, so never claim a source."
    return "\n".join(f"{f['id']}: {f['text']}" for f in facts)


def links_for(ids, facts_by_id):
    seen, out = set(), []
    for i in ids:
        f = facts_by_id.get(i)
        if not f:
            continue
        for l in f["links"]:
            if l["uri"] not in seen:
                seen.add(l["uri"])
                out.append(l)
    return out


def link_line(links):
    return "Sources: " + " · ".join(f"[{l['title']}]({l['uri']})" for l in links[:4])


# ---------- cleaning what the AI returns ----------

def clean_ids(raw, valid_ids):
    if not isinstance(raw, list):
        return []
    return [i for i in raw if isinstance(i, str) and i in valid_ids]


def clean_tag(tag, ids):
    if tag not in TAGS:
        tag = "Assumption"
    if tag == "Sourced" and not ids:
        tag = "Estimated"
    return tag


def normalise_specialist(out, valid_ids):
    ids = clean_ids(out.get("source_ids"), valid_ids)
    return {
        "finding": str(out.get("finding", "")),
        "key_assumption": str(out.get("key_assumption", "")),
        "tag": clean_tag(out.get("tag", "Assumption"), ids),
        "source_ids": ids,
    }


def norm_driver(d, valid_ids, is_ramp=False):
    vals = {k: float(d[k]) for k in ("worst", "base", "best")}
    if is_ramp:
        vals = {k: float(max(1, round(v))) for k, v in vals.items()}
    ids = clean_ids(d.get("source_ids"), valid_ids)
    return {
        "label": str(d.get("label", "")),
        "why": str(d.get("why", "")),
        "tag": clean_tag(d.get("tag", "Assumption"), ids),
        "source_ids": ids,
        **vals,
    }


def normalise_model(raw, valid_ids):
    unit = str(raw.get("unit", "units"))[:20]
    horizon = int(max(6, min(36, float(raw.get("horizon_months", 12)))))
    model = {k: norm_driver(raw[k], valid_ids, is_ramp=(k == "ramp_months")) for k in DRIVER_KEYS}
    return unit, horizon, model


def get_board(client, q, facts, valid_ids, note=None):
    """Quick Take: four specialists and the model in ONE request."""
    for _ in range(2):
        raw = ask_json(client, fill(BOARD_PROMPT, QUESTION=q, FACTS=facts_block(facts)), 0.5, note)
        try:
            specs = {}
            for a in AGENTS:
                specs[a["name"]] = normalise_specialist(raw["specialists"][a["name"]], valid_ids)
            unit, horizon, model = normalise_model(raw["model"], valid_ids)
            return specs, unit, horizon, model
        except Exception:
            continue
    raise ValueError("The board's answer was incomplete twice. Please run again.")


def get_model(client, q, findings_text, facts, valid_ids, note=None):
    """Full Engagement: the finance modeller in its own request."""
    for _ in range(2):
        raw = ask_json(
            client,
            fill(ECON_PROMPT, QUESTION=q, FINDINGS=findings_text, FACTS=facts_block(facts)),
            0.3, note,
        )
        try:
            return normalise_model(raw, valid_ids)
        except Exception:
            continue
    raise ValueError("The finance modeller returned numbers that could not be read. Please run again.")


def get_close(client, q, findings_text, nums, note=None):
    """Devil's advocate and managing partner in ONE request."""
    raw = ask_json(
        client,
        fill(CLOSE_PROMPT, QUESTION=q, FINDINGS=findings_text, NUMBERS=nums),
        0.4, note,
    )
    devil, memo = raw.get("devil"), raw.get("memo")
    if not isinstance(devil, dict) or not isinstance(memo, dict):
        raise ValueError("The final memo was incomplete. Please run again.")
    return devil, memo


# ---------- the numbers (plain Python, no AI) ----------

def project(upfront, benefit, cost, ramp, horizon):
    cum = -upfront
    series = [cum]
    payback = None
    for m in range(1, horizon + 1):
        factor = min(1.0, m / ramp) if ramp > 0 else 1.0
        cum += benefit * factor - cost
        series.append(cum)
        if payback is None and cum >= 0:
            payback = m
    return series, payback


def run_scenarios(model, horizon):
    out = {}
    for name in ("worst", "base", "best"):
        series, payback = project(
            model["upfront_cost"][name], model["monthly_benefit"][name],
            model["monthly_cost"][name], model["ramp_months"][name], horizon,
        )
        out[name] = {"series": series, "payback": payback, "final": series[-1]}
    return out


def fmt(v):
    return f"{v:,.1f}"


def payback_text(p, horizon):
    return f"month {p}" if p else f"not within {horizon} months"


def numbers_text(unit, horizon, scen):
    lines = [f"Currency unit: {unit}. Horizon: {horizon} months."]
    for name in ("worst", "base", "best"):
        s = scen[name]
        lines.append(
            f"{name.capitalize()} case: payback {payback_text(s['payback'], horizon)}, "
            f"cumulative result at month {horizon} = {fmt(s['final'])} {unit}."
        )
    return "\n".join(lines)


def compute_confidence(specialists, model, scen, verdict, has_sources=False):
    tags = [o.get("tag", "Assumption") for o in specialists.values()]
    tags += [model[k]["tag"] for k in DRIVER_KEYS]
    # "Estimated" is the AI's own label. With no sources behind it, it counts for less.
    weight = {"Sourced": 0.0, "Estimated": 0.5 if has_sources else 0.75, "Assumption": 1.0}
    share = sum(weight.get(t, 1.0) for t in tags) / len(tags)
    n_s = sum(1 for t in tags if t == "Sourced")
    n_e = sum(1 for t in tags if t == "Estimated")
    n_a = len(tags) - n_s - n_e
    parts = [("Starting point", 70)]
    parts.append((
        f"Evidence quality: {n_s} sourced, {n_e} estimated, {n_a} assumed",
        -round(25 * share),
    ))
    base_loses = scen["base"]["final"] < 0
    v = str(verdict).strip().lower()
    if v == "go":
        if base_loses:
            parts.append(("Recommends going, but the base case loses money", -20))
        elif scen["worst"]["final"] < 0:
            parts.append(("Recommends going, but the all-worst case loses money", -5))
        else:
            parts.append(("Recommends going, and even the all-worst case ends positive", 10))
    elif v == "pilot":
        if base_loses:
            parts.append(("A pilot fits a base case that loses money", 10))
        elif scen["worst"]["final"] < 0:
            parts.append(("A pilot fits a positive base case with a deep downside", 5))
        else:
            parts.append(("Recommends a pilot although even the all-worst case ends positive", -10))
    else:
        if base_loses:
            parts.append(("Caution matches a base case that loses money", 10))
        else:
            parts.append(("Recommends pausing although the base case makes money", -15))
    parts.append(("The devil's advocate raised a challenge", -5))
    total = max(10, min(90, sum(p[1] for p in parts)))
    return total, parts


# ---------- look and feel ----------

ACCENT = "#C8F169"
INK = "#E9E7E2"
MUTED = "#9AA4B2"
PANEL = "#161C24"
LINE = "#263040"
AGENT_COLORS = {
    "Market analyst": "#F2B84B",
    "Finance lead": "#4FD1C5",
    "Operations": "#6CB6FF",
    "Risk officer": "#FF7A66",
    "Devil's advocate": "#A593FF",
}
TAG_COLORS = {"Sourced": ACCENT, "Estimated": "#F2B84B", "Assumption": "#FF7A66"}
VERDICTS = {"go": ("GO", ACCENT), "pilot": ("PILOT", "#F2B84B"), "pause": ("PAUSE", "#FF7A66")}

EXAMPLES = [
    ("Should a D2C skincare brand expand to Tier-2 cities?", "EXPANSION"),
    ("Should a B2B SaaS startup move from monthly to annual pricing?", "PRICING"),
    ("Should a restaurant chain open a cloud kitchen?", "NEW CHANNEL"),
    ("Should a logistics startup build its own warehouses or rent them?", "BUILD VS RENT"),
    ("Should an edtech company launch a free tier?", "GROWTH"),
    ("Should a retailer cut prices to answer a new low-cost competitor?", "COMPETITION"),
]

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp{background:#0E1217;color:#E9E7E2;font-family:'Instrument Sans',system-ui,sans-serif}
.stApp p,.stApp li,.stApp label{font-family:'Instrument Sans',system-ui,sans-serif}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
header[data-testid="stHeader"]{background:transparent;height:0}
.block-container{max-width:1120px;padding:0 32px 72px!important}
.mono{font-family:'JetBrains Mono',monospace}
a{color:#C8F169}
.dl-top{display:flex;flex-wrap:wrap;align-items:center;gap:14px 28px;padding:22px 0;border-bottom:1px solid #263040;margin-bottom:36px}
.dl-logo{display:flex;align-items:center;gap:12px;font-family:'JetBrains Mono',monospace;font-size:15px;letter-spacing:.04em}
.dl-dia{width:12px;height:12px;background:#C8F169;border-radius:3px;transform:rotate(45deg);display:inline-block}
.dl-steps{display:flex;gap:8px;flex-wrap:wrap;flex-grow:1}
.dl-step{display:flex;gap:8px;align-items:center;padding:6px 12px;border-radius:999px;font-size:13px;border:1px solid #263040;color:#9AA4B2}
.dl-step b{font-family:'JetBrains Mono',monospace;font-weight:400}
.dl-step.on{border-color:#C8F169;color:#C8F169;background:rgba(200,241,105,.08)}
.dl-step.done{border-color:#3A4658;color:#E9E7E2}
.dl-note{font-size:12px;color:#9AA4B2;border:1px dashed #3A4658;padding:6px 12px;border-radius:8px}
.dl-eyebrow{font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:.08em;color:#C8F169}
.dl-h1{font-weight:400!important;padding:0!important;font-family:'Newsreader',Georgia,serif;font-weight:400;font-size:clamp(34px,5.2vw,60px)!important;line-height:1.05;letter-spacing:-.02em;margin:14px 0 16px;color:#E9E7E2}
.dl-h2{font-weight:400!important;padding:0!important;font-family:'Newsreader',Georgia,serif;font-size:clamp(26px,3.4vw,38px)!important;line-height:1.15;margin:10px 0 6px;color:#E9E7E2}.dl-lede{font-size:18px;line-height:1.5;color:#9AA4B2;max-width:760px;margin-bottom:28px}
.dl-label{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.06em;color:#9AA4B2;margin:28px 0 10px}
.dl-card{background:#161C24;border:1px solid #263040;border-radius:14px;padding:18px 20px;margin-bottom:14px}
.dl-card-h{display:flex;justify-content:space-between;align-items:center;font-size:12px;letter-spacing:.06em}
.dl-status{font-family:'JetBrains Mono',monospace;font-size:11px;color:#9AA4B2;text-transform:uppercase}
.dl-sub{color:#9AA4B2;font-size:13px;margin:4px 0 10px}
.dl-body{font-size:15.5px;line-height:1.55;color:#E9E7E2}
.dl-tagrow{margin-top:12px}
.dl-chip{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.04em;border:1px solid;border-radius:999px;padding:3px 10px}
.dl-dep{font-size:13px;color:#9AA4B2;margin-top:10px;line-height:1.45}
.dl-src{font-size:12.5px;color:#9AA4B2;margin-top:8px}
.dl-memo{background:#161C24;border:1px solid #263040;border-radius:18px;padding:28px 30px;margin:6px 0 18px}
.dl-verdict{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.08em;border:1px solid;border-radius:6px;padding:4px 10px;margin-bottom:14px}
.dl-answer{font-family:'Newsreader',Georgia,serif;font-size:clamp(21px,2.4vw,27px);line-height:1.4;color:#E9E7E2;margin:0 0 14px}
.dl-sec{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.08em;color:#C8F169;margin:26px 0 8px}
.dl-text{font-size:16px;line-height:1.6;color:#E9E7E2}
.dl-list{margin:0;padding-left:20px;color:#E9E7E2;line-height:1.6;font-size:16px}
.dl-conf{margin-top:18px}
.dl-bar{height:8px;background:#263040;border-radius:99px;overflow:hidden;margin-top:8px}
.dl-bar div{height:100%;background:#C8F169;border-radius:99px}
.dl-table{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0 6px}
.dl-table th{font-family:'JetBrains Mono',monospace;font-weight:400;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#9AA4B2;text-align:left;padding:10px 12px;border-bottom:1px solid #263040}
.dl-table td{padding:12px;border-bottom:1px solid #1d2531;vertical-align:top;color:#E9E7E2;line-height:1.45}
.dl-table td.n{font-family:'JetBrains Mono',monospace;text-align:right;white-space:nowrap}
.dl-explain div.t{font-family:'JetBrains Mono',monospace;font-size:12px;margin-bottom:8px}
.dl-explain div.d{color:#9AA4B2;line-height:1.5;font-size:15px}
.stTextArea textarea,.stTextInput input{background:#161C24!important;color:#E9E7E2!important;font-size:17px!important;font-family:'Instrument Sans',system-ui,sans-serif!important}
[data-baseweb="textarea"],[data-baseweb="base-input"],[data-baseweb="input"]{background:#161C24!important;border-color:#3A4658!important;border-radius:12px!important}
[data-baseweb="select"]>div{background:#161C24!important;border-color:#3A4658!important;border-radius:12px!important}
[data-testid="stBaseButton-primary"]{background:#C8F169!important;color:#0E1217!important;border:0!important;border-radius:12px!important;font-weight:600!important;min-height:52px;padding:0 28px!important}
[data-testid="stBaseButton-primary"] p{color:#0E1217!important;font-weight:600!important}
[data-testid="stBaseButton-secondary"]{background:#161C24!important;color:#E9E7E2!important;border:1px solid #263040!important;border-radius:12px!important}
[data-testid="stBaseButton-secondary"]:hover{border-color:#C8F169!important;color:#C8F169!important}
.st-key-examples [data-testid="stBaseButton-secondary"]{min-height:104px;width:100%;justify-content:flex-start;text-align:left;padding:16px 18px!important;border-radius:14px!important}
.st-key-examples [data-testid="stBaseButton-secondary"] p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;text-align:left;font-size:15.5px;line-height:1.4}
div[role="radiogroup"] label{background:#161C24;border:1px solid #263040;border-radius:999px;padding:6px 14px;margin-right:6px}
div[role="radiogroup"] label:has(input:checked){border-color:#C8F169}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid #263040}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#9AA4B2;padding:12px 16px}
.stTabs [aria-selected="true"]{color:#C8F169!important}
.stTabs [data-baseweb="tab-highlight"]{background:#C8F169!important}
[data-testid="stMetric"]{background:#161C24;border:1px solid #263040;border-radius:14px;padding:14px 18px}
[data-testid="stMetricLabel"] p{font-family:'JetBrains Mono',monospace;font-size:12px;color:#9AA4B2;letter-spacing:.04em}
[data-testid="stMetricValue"]{font-family:'Newsreader',Georgia,serif;font-size:34px}
[data-testid="stExpander"]{background:#161C24;border:1px solid #263040!important;border-radius:12px}
[data-testid="stAlert"]{background:#161C24;border:1px solid #263040;border-radius:12px;color:#E9E7E2}
@media (max-width:760px){.block-container{padding:0 16px 56px!important}.dl-memo{padding:20px 18px}.dl-top{margin-bottom:24px}}
</style>"""


def e(x):
    return htmllib.escape(str(x), quote=True)


def safe_href(u):
    u = str(u)
    return u if u.startswith(("http://", "https://")) else ""


def src_html(links):
    items = " · ".join(
        f'<a href="{e(safe_href(l["uri"]))}" target="_blank" rel="noopener">{e(l["title"])}</a>'
        for l in (links or [])[:4] if safe_href(l["uri"])
    )
    return f'<div class="dl-src">Sources: {items}</div>' if items else ""


def chip(tag):
    color = TAG_COLORS.get(tag, MUTED)
    return f'<span class="dl-chip" style="color:{color};border-color:{color}">{e(tag)}</span>'


def agent_card(name, sub, status, body="", tag="", assumption="", links=None):
    color = AGENT_COLORS.get(name, ACCENT)
    parts = [
        f'<div class="dl-card" style="border-left:3px solid {color}">',
        f'<div class="dl-card-h"><span class="mono" style="color:{color}">{e(name.upper())}</span>'
        f'<span class="dl-status">{e(status)}</span></div>',
        f'<div class="dl-sub">{e(sub)}</div>',
    ]
    if body:
        parts.append(f'<div class="dl-body">{e(body)}</div>')
    if tag:
        parts.append(f'<div class="dl-tagrow">{chip(tag)}</div>')
    if assumption:
        parts.append(f'<div class="dl-dep">Depends on: {e(assumption)}</div>')
    parts.append(src_html(links))
    parts.append("</div>")
    return "".join(parts)


def draw_card(slot, name, sub, status, body="", tag="", assumption="", links=None):
    slot.markdown(agent_card(name, sub, status, body, tag, assumption, links), unsafe_allow_html=True)


def top_bar(step):
    labels = ["Decision", "Board", "Memo"]
    pills = []
    for i, label in enumerate(labels, 1):
        cls = "on" if i == step else ("done" if i < step else "")
        pills.append(f'<div class="dl-step {cls}"><b>0{i}</b>{label}</div>')
    return (
        '<div class="dl-top"><div class="dl-logo"><span class="dl-dia"></span>DECISIONLAB</div>'
        f'<div class="dl-steps">{"".join(pills)}</div>'
        '<div class="dl-note">Simulated thinking, not verified advice.</div></div>'
    )


# ---------- result pieces ----------

def numbers_table(model, show_facts):
    head = "<tr><th>Driver</th><th>Worst</th><th>Base</th><th>Best</th><th>Tag</th>"
    head += "<th>Facts</th>" if show_facts else ""
    head += "<th>Why</th></tr>"
    rows = []
    for k in DRIVER_KEYS:
        d = model[k]
        cells = (
            f'<td>{e(d["label"])}</td><td class="n">{d["worst"]:g}</td><td class="n">{d["base"]:g}</td>'
            f'<td class="n">{d["best"]:g}</td><td>{chip(d["tag"])}</td>'
        )
        if show_facts:
            cells += f'<td class="mono">{e(", ".join(d["source_ids"]))}</td>'
        cells += f'<td style="color:#9AA4B2">{e(d["why"])}</td>'
        rows.append(f"<tr>{cells}</tr>")
    return f'<div style="overflow-x:auto"><table class="dl-table">{head}{"".join(rows)}</table></div>'


def tab_memo(res):
    m = res["memo"]
    conf, parts = compute_confidence(
        res["specialists"], res["model"], res["scen"], m.get("verdict", ""), bool(res["facts"])
    )
    label, color = VERDICTS.get(str(m.get("verdict", "")).strip().lower(), ("MEMO", ACCENT))
    risks = "".join(f"<li>{e(r)}</li>" for r in m.get("risks", []))
    steps = "".join(f"<li>{e(s)}</li>" for s in m.get("next_steps", []))
    st.markdown(
        '<div class="dl-memo">'
        f'<div class="dl-verdict" style="color:{color};border-color:{color}">{label}</div>'
        f'<p class="dl-answer">{e(m.get("answer", ""))}</p>'
        f'<div class="dl-text" style="color:#9AA4B2">{e(m.get("numbers_comment", ""))}</div>'
        '<div class="dl-conf">'
        f'<div class="mono" style="font-size:12px;letter-spacing:.06em;color:#9AA4B2">CONFIDENCE {conf}% · CALCULATED, NOT GUESSED</div>'
        f'<div class="dl-bar"><div style="width:{conf}%"></div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How this confidence is calculated"):
        for lab, pts in parts:
            st.markdown(f"- {lab}: **{pts:+d}**")
        st.caption("The answer uses the base case. Your sliders only change the chart on the Numbers tab.")
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="dl-sec">WHERE THE BOARD DISAGREED</div><div class="dl-text">{e(m.get("disagreement", ""))}</div>'
        f'<div class="dl-sec">WHAT WOULD CHANGE THIS ANSWER</div><div class="dl-text">{e(m.get("what_would_change_this", ""))}</div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="dl-sec">BIGGEST RISKS</div><ul class="dl-list">{risks}</ul>'
        f'<div class="dl-sec">NEXT STEPS</div><ol class="dl-list">{steps}</ol>',
        unsafe_allow_html=True,
    )


def tab_board(res):
    facts_by_id = {f["id"]: f for f in res["facts"]}
    cols = st.columns(2)
    for i, a in enumerate(AGENTS):
        out = res["specialists"][a["name"]]
        cols[i % 2].markdown(
            agent_card(
                a["name"], a["focus"], "done", out.get("finding", ""), out.get("tag", ""),
                out.get("key_assumption", ""), links_for(out.get("source_ids", []), facts_by_id),
            ),
            unsafe_allow_html=True,
        )
    d = res["devil"]
    color = AGENT_COLORS["Devil's advocate"]
    st.markdown(
        f'<div class="dl-card" style="border-left:3px solid {color}">'
        f'<div class="dl-card-h"><span class="mono" style="color:{color}">DEVIL\'S ADVOCATE</span>'
        f'<span class="dl-status">challenges {e(d.get("target", ""))}</span></div>'
        f'<div class="dl-body" style="margin-top:10px">{e(d.get("challenge", ""))}</div>'
        f'<div class="dl-dep">Evidence that would settle it: {e(d.get("evidence_needed", ""))}</div></div>',
        unsafe_allow_html=True,
    )


def tab_numbers(res):
    unit, horizon, model, scen = res["unit"], res["horizon"], res["model"], res["scen"]
    rid = st.session_state.get("run_id", 0)
    st.caption(f"All amounts in {unit}. Horizon: {horizon} months. Estimates by the AI, calculated by code.")
    st.markdown(numbers_table(model, bool(res["facts"])), unsafe_allow_html=True)

    st.markdown('<div class="dl-sec">TEST YOUR OWN ASSUMPTIONS</div>', unsafe_allow_html=True)
    mine = {}
    for k in DRIVER_KEYS:
        d = model[k]
        lo, hi = min(d["worst"], d["best"]), max(d["worst"], d["best"])
        if k == "ramp_months":
            smin, smax = 1, int(max(hi * 1.5, 6))
            val = int(min(max(round(d["base"]), smin), smax))
            mine[k] = st.slider(f"{d['label']} (months)", smin, smax, val, key=f"sl_{k}_{rid}")
        else:
            step = 0.5 if hi >= 10 else 0.1
            smin = math.floor(max(0.0, lo * 0.5) / step) * step
            smax = math.ceil(max(hi * 1.5, smin + 1.0) / step) * step
            val = round(min(max(d["base"], smin), smax) / step) * step
            mine[k] = st.slider(
                f"{d['label']} ({unit})", float(smin), float(smax), float(val),
                step=float(step), key=f"sl_{k}_{rid}",
            )

    series, payback = project(
        mine["upfront_cost"], mine["monthly_benefit"], mine["monthly_cost"], mine["ramp_months"], horizon
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Payback (yours)", f"Month {payback}" if payback else "Not in horizon")
    c2.metric(f"Yours, month {horizon}", fmt(series[-1]))
    c3.metric(f"All-worst, month {horizon}", fmt(scen["worst"]["final"]))

    months = list(range(horizon + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=scen["worst"]["series"], name="All-worst case", line=dict(color="#FF7A66", width=2)))
    fig.add_trace(go.Scatter(x=months, y=scen["base"]["series"], name="Base case", line=dict(color="#6CB6FF", width=3)))
    fig.add_trace(go.Scatter(x=months, y=scen["best"]["series"], name="All-best case", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=months, y=series, name="Your settings", line=dict(color=INK, width=2, dash="dash")))
    fig.add_hline(y=0, line_color="#3A4658", line_width=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Instrument Sans, sans-serif"),
        title=dict(text="Cumulative cash from the decision", font=dict(color=INK, size=16)),
        xaxis=dict(title="Month", gridcolor=LINE, zeroline=False),
        yaxis=dict(title=unit, gridcolor=LINE, zeroline=False),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=10, r=10, t=50, b=10), height=400,
    )
    st.plotly_chart(fig, width="stretch")


def tab_research(res):
    if not res.get("research_on"):
        st.markdown(
            '<div class="dl-text" style="color:#9AA4B2">Live research was switched off for this run, '
            "so nothing is marked Sourced.</div>", unsafe_allow_html=True,
        )
        return
    if not res["facts"]:
        reason = f' Reason: {e(res.get("research_error", ""))}' if res.get("research_error") else ""
        st.markdown(
            f'<div class="dl-text" style="color:#9AA4B2">No live sources this run, so nothing is marked Sourced.{reason}</div>',
            unsafe_allow_html=True,
        )
        return
    st.caption("Facts found by Google Search. Click a link to check the source yourself.")
    for f in res["facts"]:
        st.markdown(
            f'<div class="dl-card"><div class="mono" style="color:{ACCENT};font-size:12px">{e(f["id"])}</div>'
            f'<div class="dl-body" style="margin-top:6px">{e(f["text"])}</div>{src_html(f["links"])}</div>',
            unsafe_allow_html=True,
        )


# ---------- state helpers ----------

def go_home():
    for k in ("result6", "job"):
        st.session_state.pop(k, None)
    st.session_state["screen"] = "home"


def set_question(text):
    st.session_state["question"] = text


def load_demo():
    st.session_state["result6"] = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
    st.session_state["screen"] = "result"


def need_for(mode, research):
    return (2 if mode == "Quick Take" else 6) + (1 if research else 0)


def start_job():
    q = st.session_state.get("question", "").strip()
    mode = st.session_state.get("mode", "Quick Take")
    research = bool(st.session_state.get("research", False))
    user_key = st.session_state.get("user_key", "")
    shared = get_key()
    key = user_key or shared
    using_shared = bool(shared) and not user_key
    need = need_for(mode, research)

    if not q:
        st.session_state["notice"] = "Type a decision first, or pick one of the examples."
    elif not key:
        st.session_state["notice"] = "Paste your Gemini key in the key box first."
    elif using_shared and st.session_state.get("runs", 0) >= SESSION_RUN_CAP:
        st.session_state["notice"] = (
            "You have used the free runs for this visit. Paste your own free Gemini key to keep going, "
            "or open the saved example."
        )
    elif using_shared and SHARED_DAILY_BUDGET - usage_today()["calls"] < need:
        st.session_state["notice"] = (
            "Today's shared free runs are used up. Paste your own free Gemini key, or open the saved example."
        )
    else:
        st.session_state["job"] = {
            "q": q, "mode": mode, "research": research,
            "horizon": int(st.session_state.get("horizon", 12)), "key": key.strip(),
        }
        st.session_state["screen"] = "running"


# ---------- screens ----------

def screen_home():
    st.markdown(top_bar(1), unsafe_allow_html=True)
    st.markdown(
        '<div class="dl-eyebrow">AGENTIC CONSULTING ROOM</div>'
        '<h1 class="dl-h1">Bring a business decision. Watch the AI board argue about it.</h1>'
        '<div class="dl-lede">Four specialists and a devil\'s advocate build the numbers, challenge each other, '
        "and hand you a memo that shows its evidence and its doubts.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("notice"):
        st.warning(st.session_state.pop("notice"))

    st.text_area(
        "Your decision", key="question", height=110, label_visibility="collapsed",
        placeholder="Type your decision, or pick one below",
    )

    st.markdown('<div class="dl-label">BOARD SETTINGS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 1, 1.3])
    c1.radio(
        "How deep should the board go?", ["Quick Take", "Full Engagement"], key="mode",
        captions=["2 requests. The whole board answers at once.", "6 requests. Watch each agent work."],
    )
    c2.selectbox("Time horizon", [12, 24, 36], key="horizon", format_func=lambda m: f"{m} months")
    c3.checkbox("Live web research with sources (+1 request, often unavailable on the free tier)", key="research")

    shared = get_key()
    if shared:
        with st.expander("Use your own Gemini key (optional)"):
            st.text_input(
                "Your key stays in this browser session. Leave empty to use the shared free runs.",
                type="password", key="user_key",
            )
        st.caption(f"Shared free runs: up to {SESSION_RUN_CAP} per visit.")
    else:
        st.text_input("Gemini API key (stays in this browser session only)", type="password", key="user_key")

    need = need_for(st.session_state.get("mode", "Quick Take"), st.session_state.get("research", False))
    used = st.session_state.get("calls", 0)
    st.caption(
        f"This run needs about {need} requests. Sent from this browser session so far: {used}. "
        f"The free tier allows about {DAILY_LIMIT} a day per key."
    )
    b1, b2 = st.columns([1, 3])
    b1.button("Run the board", type="primary", on_click=start_job)
    if DEMO_FILE.exists():
        b2.button("Open the saved example (no requests used)", on_click=load_demo)

    st.markdown('<div class="dl-label">OR TRY ONE OF THESE</div>', unsafe_allow_html=True)
    with st.container(key="examples"):
        for row in range(0, len(EXAMPLES), 3):
            cols = st.columns(3)
            for col, (text, tag) in zip(cols, EXAMPLES[row:row + 3]):
                col.button(text, key=f"ex_{tag}", on_click=set_question, args=(text,))

    st.markdown(
        '<div class="dl-explain" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));'
        'gap:24px;padding-top:28px;margin-top:28px;border-top:1px solid #263040">'
        '<div><div class="t" style="color:#F2B84B">01 BRIEF</div><div class="d">Type any business decision, or start from an example.</div></div>'
        '<div><div class="t" style="color:#4FD1C5">02 DEBATE</div><div class="d">Specialists give findings tagged Sourced, Estimated or Assumption. A devil\'s advocate attacks the weakest one.</div></div>'
        '<div><div class="t" style="color:#A593FF">03 TEST</div><div class="d">Move the sliders and watch worst, base and best cases change. Confidence is calculated, not guessed.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def screen_running():
    job = st.session_state["job"]
    q, mode, research_on, horizon_choice = job["q"], job["mode"], job["research"], job["horizon"]
    st.markdown(top_bar(2), unsafe_allow_html=True)
    st.markdown(
        '<div class="dl-eyebrow">THE BOARDROOM, LIVE</div>'
        f'<h2 class="dl-h2">{e(q)}</h2>',
        unsafe_allow_html=True,
    )
    client = genai.Client(api_key=job["key"])
    calls_before = st.session_state.get("calls", 0)
    status_slot = st.empty()
    note = status_slot.info

    slots = {}
    if mode == "Full Engagement":
        cols = st.columns(2)
        for i, a in enumerate(AGENTS):
            slots[a["name"]] = cols[i % 2].empty()
            draw_card(slots[a["name"]], a["name"], a["focus"], "waiting")

    try:
        facts, research_error = [], ""
        if research_on:
            note("The researcher is searching the live web...")
            facts, research_error = run_research(client, q, note)
        valid_ids = {f["id"] for f in facts}
        facts_by_id = {f["id"]: f for f in facts}

        if mode == "Quick Take":
            note("The board is meeting (specialists and finance modeller, one request)...")
            specialists, unit, horizon, model = get_board(client, q, facts, valid_ids, note)
        else:
            specialists = {}
            for a in AGENTS:
                note(f"{a['name']} is working...")
                draw_card(slots[a["name"]], a["name"], a["focus"], "working")
                prompt = fill(
                    SPECIALIST_PROMPT, NAME=a["name"], FOCUS=a["focus"],
                    QUESTION=q, FACTS=facts_block(facts),
                )
                out = normalise_specialist(ask_json(client, prompt, 0.5, note), valid_ids)
                specialists[a["name"]] = out
                draw_card(
                    slots[a["name"]], a["name"], a["focus"], "done",
                    out["finding"], out["tag"], out["key_assumption"],
                    links_for(out["source_ids"], facts_by_id),
                )
            findings_tmp = "\n".join(
                f"- {n}: {o['finding']} (assumes: {o['key_assumption']})" for n, o in specialists.items()
            )
            note("The finance modeller is building the numbers...")
            unit, horizon, model = get_model(client, q, findings_tmp, facts, valid_ids, note)

        horizon = horizon_choice  # the visitor's choice wins; the monthly numbers stay the same
        findings_text = "\n".join(
            f"- {name}: {o['finding']} (assumes: {o['key_assumption']})"
            for name, o in specialists.items()
        )
        scen = run_scenarios(model, horizon)
        nums = numbers_text(unit, horizon, scen)

        note("The devil's advocate and the managing partner are writing the memo...")
        devil, memo = get_close(client, q, findings_text, nums, note)

        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        st.session_state["result6"] = {
            "mode": mode, "research_on": research_on,
            "calls_used": st.session_state.get("calls", 0) - calls_before,
            "facts": facts, "research_error": research_error,
            "specialists": specialists, "devil": devil, "memo": memo,
            "unit": unit, "horizon": horizon, "model": model, "scen": scen,
            "question": q,
        }
        st.session_state.pop("job", None)
        st.session_state["screen"] = "result"
    except QuotaError:
        st.error(
            "Gemini says the free quota for this key is used up. It usually resets once a day. "
            "You can wait, or use a key from a different Google AI Studio project."
        )
        st.button("Back", on_click=go_home)
        st.stop()
    except Exception as ex:
        st.error("Something went wrong. Try again in a minute.")
        st.caption(str(ex)[:300])
        st.button("Back", on_click=go_home)
        st.stop()

    st.rerun()


def screen_result():
    res = st.session_state["result6"]
    st.markdown(top_bar(3), unsafe_allow_html=True)
    left, right = st.columns([4, 1])
    left.markdown(
        '<div class="dl-eyebrow">DECISIONLAB MEMO</div>'
        f'<h2 class="dl-h2">{e(res.get("question", "Saved example"))}</h2>'
        f'<div class="dl-status" style="margin-bottom:18px">{e(res.get("mode", ""))} · '
        f'{e(res.get("calls_used", "?"))} requests · {e(res["horizon"])}-month horizon</div>',
        unsafe_allow_html=True,
    )
    right.button("Ask a new question", on_click=go_home, width="stretch")

    t1, t2, t3, t4 = st.tabs(["Memo", "Boardroom", "Numbers", "Research"])
    with t1:
        tab_memo(res)
    with t2:
        tab_board(res)
    with t3:
        tab_numbers(res)
    with t4:
        tab_research(res)

    if not PUBLIC_MODE:
        with st.expander("Owner tools"):
            st.caption("Saves this result so visitors can open it without using any requests.")
            if st.button("Save this result as the saved example"):
                DEMO_FILE.write_text(json.dumps(res), encoding="utf-8")
                st.success("Saved as demo_result.json, next to app.py.")


# ---------- the page ----------

st.set_page_config(page_title="DecisionLab", page_icon="◆", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# keep these widget values alive while another screen is showing
for _k in ("question", "mode", "horizon", "research", "user_key"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

_screen = st.session_state.get("screen", "home")
if _screen == "running" and "job" in st.session_state:
    screen_running()
elif _screen == "result" and "result6" in st.session_state:
    screen_result()
else:
    screen_home()
