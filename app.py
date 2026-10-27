# ==============================================================================
# JOURNEY / AI — Customer Funnel Intelligence Agent | ONE Google Colab cell
# Find the leaks. Fix the journey. Grow revenue.
# ==============================================================================
import subprocess, sys, os, re, zipfile, html, warnings
warnings.filterwarnings("ignore")

def _deps():
    try:
        from importlib.metadata import version
        if int(version("gradio").split(".")[0]) not in (4, 5): raise RuntimeError("gradio")
        for p in ("plotly", "scikit-learn", "requests", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "requests", "pandas", "numpy"])
        except Exception as e: print("Install warning:", type(e).__name__, "- check internet access and re-run.")
_deps()
try:
    import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr, requests
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as e:
    raise SystemExit("Libraries could not be imported (" + type(e).__name__ + "). Restart the runtime and re-run this cell.")

C = dict(bg="#111522", sf="#1B1E2E", la="#B9A5FF", orc="#E3A5F0", co="#F6B6BB", vi="#9879FF", tx="#F5F3FA", mu="#A9ADC4", ok="#8EE3C2", am="#F2C48D")
G = {}; TESTS = []
BLANK = ["", "nan", "none", "n/a", "na", "-", "null"]
esc = lambda x: html.escape(str(x))
norm = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
money = lambda v: "—" if nul(v) else ("-$" + format(abs(v), ",.0f") if v < 0 else "$" + format(v, ",.0f"))
pc0 = lambda v: "—" if nul(v) else format(v, ".0%")
pc1 = lambda v: "—" if nul(v) else format(v, ".1%")
fx = lambda v, d=2: "—" if nul(v) else format(v, "." + str(d) + "f")
n0 = lambda v: "—" if nul(v) else format(v, ",.0f")
x2 = lambda v: "—" if nul(v) else format(v, ".2f") + "x"
def sd(a, b):
    return float(a) / float(b) if (not nul(a) and not nul(b) and float(b) > 0) else np.nan
def dv(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    with np.errstate(all="ignore"): return np.where(y > 0, x / np.where(y > 0, y, 1.0), np.nan)
def mcol(v): return "—" if nul(v) else "<span style='color:%s'>%s</span>" % (C["ok"] if v >= 0 else C["co"], money(v))
def cond(fn, ref, hi=True):
    def f(v):
        if nul(v): return "—"
        s = fn(v)
        if nul(ref) or ref == 0: return s
        r = v / ref; good = r >= 1.2 if hi else r <= .8; bad = r <= .8 if hi else r >= 1.2
        col = C["ok"] if good else (C["co"] if bad else None)
        return "<span style='color:%s'>%s</span>" % (col, s) if col else s
    return f
TAGC = {"High volume · low quality": C["co"], "High volume · high quality": C["ok"], "Low volume · high quality": C["la"], "Low volume · low quality": C["am"], "Mid": C["mu"], "Insufficient data": C["mu"]}
def tagf(v): return "<span class='pill' style='background:%s26;color:%s'>%s</span>" % (TAGC.get(v, C["mu"]), TAGC.get(v, C["mu"]), esc(v))
def note(t, k="info"):
    col = {"info": C["la"], "ok": C["ok"], "warn": C["am"], "err": C["co"]}[k]
    return "<div class='note' style='border-left:4px solid %s'>%s</div>" % (col, t)
def card(ic, lab, val, desc, col=C["vi"]):
    return "<div class='kc' style='border-top:3px solid %s'><div class='ki'>%s</div><div class='kl'>%s</div><div class='kv'>%s</div><div class='kd'>%s</div></div>" % (col, ic, lab, val, desc)
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows():
        h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=380, t=""):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["tx"]), height=h, title=dict(text=t, x=.02),
                    margin=dict(l=50, r=20, t=55, b=50), legend=dict(orientation="h", y=-.25), colorway=[C["vi"], C["orc"], C["la"], C["co"]])
    return f
def empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(color=C["mu"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    return sty(f, 300)
def wilson(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - a) / d, (c + a) / d)

# ------------------------------ schema ------------------------------
STG = ["exposure", "visit", "lead", "mql", "sql", "opp", "won"]; CHAIN = STG[1:]
LBL = dict(exposure="Campaign exposure", visit="Website visit", lead="Lead generated", mql="Marketing-qualified lead", sql="Sales-qualified lead", opp="Opportunity", won="Closed-won customer")
ACOL = dict(exposure="impressions", visit="visits", lead="leads", mql="mql", sql="sql", opp="opps", won="wins")
BASE = ["spend", "impressions", "clicks", "visits", "leads", "mql", "sql", "opps", "wins", "revenue", "refunds", "net_rev"]
SYN = {"timestamp": ["timestamp", "event_timestamp", "event_time", "datetime"], "date": ["date", "day", "week", "month", "period", "order_date"],
       "uid": ["customer_id", "visitor_id", "lead_id", "user_id", "contact_id", "session_id"], "event": ["event_name", "event", "stage", "touchpoint", "step"],
       "channel": ["channel", "source", "medium", "platform", "utm_source"], "campaign": ["campaign", "campaign_name", "campaign_id", "utm_campaign"],
       "spend": ["campaign_cost", "spend", "cost", "ad_spend", "media_cost"], "impressions": ["impressions", "impr", "views"], "clicks": ["clicks", "link_clicks"],
       "visits": ["sessions", "landing_page_views", "visits", "website_visits"], "leads": ["leads", "form_submissions", "leads_generated"],
       "mql": ["marketing_qualified_leads", "mqls"], "sql": ["sales_qualified_leads", "sqls"], "opps": ["opportunities", "opps", "opportunities_created"],
       "wins": ["closed_won", "conversions", "customers", "won", "orders"], "value": ["deal_value", "revenue", "amount", "order_value"], "refunds": ["refunds"],
       "lead_score": ["lead_score"], "segment": ["industry", "segment", "customer_segment"], "csize": ["company_size", "size"]}

def stage_of(x):
    s = re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")
    if re.search(r"won|purchase|conversion|order", s): return "won"
    if "opportunit" in s or s == "opp": return "opp"
    if "sales_qualified" in s or re.search(r"(^|_)sql($|_)", s): return "sql"
    if "marketing_qualified" in s or re.search(r"(^|_)mql($|_)", s): return "mql"
    if "lead" in s or "form" in s or "signup" in s: return "lead"
    if re.search(r"visit|page_?view|session|landing|click|touch", s): return "visit"
    return None
def clean_str(s):
    t = s.astype(str).str.strip(); t = t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan); return t
def to_num(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%₹€£\s]", "", regex=True), errors="coerce")
def to_dt(s):
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(file):
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    if df.empty: raise ValueError("The file contains no rows.")
    if len(df) > 400000: raise ValueError("File too large (limit 400,000 rows).")
    return df

# ------------------------------ FICTIONAL synthetic data ------------------------------
CFG = [("Paid Search", "Brand Search", 700, .10, .55, .55, .60, .45, 9000, 35, 14000, 2.2, .08, .01, 70), ("Paid Search", "Generic Search", 1200, .14, .35, .45, .55, .30, 8000, 50, 40000, 4.5, .03, .02, 55),
       ("Paid Social", "LinkedIn Lead Gen", 1500, .28, .22, .35, .50, .22, 9500, 70, 60000, 6.5, .006, .02, 40), ("Paid Social", "Meta Prospecting", 2200, .30, .10, .30, .45, .18, 5000, 45, 30000, 1.5, .01, .03, 30),
       ("Display", "Programmatic Display", 1800, .12, .12, .30, .40, .20, 4500, 55, 22000, 1.1, .004, .05, 35), ("Content", "Whitepaper Download", 900, .35, .25, .40, .50, .28, 7500, 60, 12000, 3.0, .012, .01, 45),
       ("Content", "Webinar Series", 450, .20, .60, .60, .65, .40, 12000, 75, 18000, 5.0, .02, .01, 72), ("Email", "Nurture Email", 600, .12, .50, .55, .60, .38, 7000, 30, 3000, .8, .03, .01, 65),
       ("Partner", "Partner Referrals", 300, .30, .65, .65, .65, .50, 15000, 60, 9000, 6.0, .02, .00, 78), ("Event", "Trade Show", 350, .45, .45, .50, .55, .30, 13000, 90, 35000, 20.0, .03, .02, 60),
       ("Organic", "Organic Search", 1000, .09, .40, .50, .55, .35, 7500, 40, 4000, 2.0, .05, .03, 60), ("Affiliate", "Review Sites", 500, .22, .30, .40, .45, .25, 6000, 40, 8000, 3.5, .02, .04, 45)]
def make_synth(seed=42):
    r = np.random.default_rng(seed); START = pd.Timestamp("2026-01-05"); CUT = pd.Timestamp("2026-09-30")
    IND = {"SaaS": 1.15, "Finance": 1.0, "Healthcare": .9, "Retail": .8, "Manufacturing": .95}; SZ = {"SMB": (.6, .8), "Mid-market": (1.0, 1.0), "Enterprise": (1.9, 1.15)}
    tw = np.array([c[2] for c in CFG], float); tw = tw / tw.sum(); rows = []; n = 0
    def add(t, uid, nm, ch, cp, val=np.nan, ref=np.nan, sc=np.nan, ind=None, sz=None): rows.append((t, uid, nm, ch, cp, val, ref, sc, ind, sz))
    for ci, (ch, cp, N, lead, mql, sql, opp, win, adv, cyc, sp, cpc, ctr, unk, sm) in enumerate(CFG):
        for _ in range(N):
            n += 1; uid = "V%06d" % n; t0 = START + pd.Timedelta(days=float(r.uniform(0, 180))); isl = r.random() < lead; nt = 1 + r.poisson(1.2 if isl else .35)
            times = [t0 + pd.Timedelta(days=float(o)) for o in np.sort(r.uniform(0, 14 if isl else 5, nt))]
            for j, t in enumerate(times):
                cj = ci if ((j == nt - 1 and isl) or (j == 0 and not isl)) else int(r.choice(len(CFG), p=tw))
                if r.random() < CFG[cj][13]: add(t, uid, "website_visit", None, None)
                else: add(t, uid, "website_visit", CFG[cj][0], CFG[cj][1])
            if not isl: continue
            t = times[-1] + pd.Timedelta(minutes=float(r.integers(1, 60))); score = float(np.clip(r.normal(sm, 14), 1, 99)); ind = str(r.choice(list(IND))); sz = str(r.choice(list(SZ), p=[.45, .4, .15]))
            chf = (None, None) if r.random() < unk else (ch, cp); add(t, uid, "lead_generated", chf[0], chf[1], sc=round(score, 1), ind=ind, sz=sz)
            steps = [("marketing_qualified_lead", min(.95, mql * (.55 + .9 * score / 100) * IND[ind]), .12), ("sales_qualified_lead", min(.95, sql * SZ[sz][1]), .2), ("opportunity_created", opp, .25), ("closed_won", min(.95, win * SZ[sz][1]), .43)]
            for nm, p, fr in steps:
                if r.random() >= p: break
                t = t + pd.Timedelta(days=float(r.exponential(cyc * fr)))
                if t > CUT: break
                if nm == "closed_won":
                    v = round(adv * SZ[sz][0] * float(r.lognormal(0, .3)), 2); add(t, uid, nm, chf[0], chf[1], v, round(v * float(r.uniform(.1, .3)), 2) if r.random() < .05 else 0.0)
                else: add(t, uid, nm, chf[0], chf[1])
    for k in range(150): add(START + pd.Timedelta(days=float(r.uniform(10, 200))), "X%04d" % k, "closed_won", None, None, round(8000 * float(r.lognormal(0, .4)), 2), 0.0)
    df = pd.DataFrame(rows, columns=["timestamp", "visitor_id", "event_name", "channel", "campaign", "deal_value", "refunds", "lead_score", "industry", "company_size"])
    df = pd.concat([df, df.sample(frac=.01, random_state=1)]).sort_values("timestamp").reset_index(drop=True); df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cost = pd.DataFrame([dict(campaign=c[1], channel=c[0], campaign_cost=c[10], clicks=int(c[10] / c[11]), impressions=int(c[10] / c[11] / c[12])) for c in CFG])
    return df, cost
SAMPLE_E, SAMPLE_C = make_synth()

# ------------------------------ ingestion ------------------------------
def map_cols(df, mapping, W):
    df = df.copy(); df.columns = [norm(c) for c in df.columns]; ov = {}
    for p in re.split(r"[,;\n]", mapping or ""):
        if "=" in p: k, v = p.split("=", 1); ov[norm(k)] = norm(v)
    bad = [k + "=" + v for k, v in ov.items() if k not in SYN or v not in df.columns]
    if bad: W.append("Mapping ignored (unknown field or column): " + ", ".join(bad) + ".")
    ren = {}
    for k, al in SYN.items():
        c = ov[k] if ov.get(k) in df.columns else next((a for a in al if a in df.columns and a not in ren), None)
        if c is None or c in ren: continue
        if c != k and k in df.columns: df = df.drop(columns=[k])
        ren[c] = k
    df = df.rename(columns=ren); return df.loc[:, ~df.columns.duplicated()]

def build_users(E):
    t = E.groupby(["uid", "stage"])["ts"].min().unstack().reindex(columns=CHAIN)
    U = pd.DataFrame({"t_" + s: pd.to_datetime(t[s]) for s in CHAIN}); U["t_first"] = E.groupby("uid")["ts"].min()
    fr = E.sort_values("ts", kind="mergesort").drop_duplicates("uid").set_index("uid"); src = fr[["channel", "campaign"]].copy()
    lr = E[E["stage"] == "lead"].sort_values("ts", kind="mergesort").drop_duplicates("uid").set_index("uid")
    if len(lr): src.loc[lr.index, ["channel", "campaign"]] = lr[["channel", "campaign"]].values
    U = U.join(src)
    for c in ["lead_score", "segment", "csize"]: U[c] = lr[c] if len(lr) else np.nan
    w = E[E["stage"] == "won"]; U["value"] = w.groupby("uid")["value"].sum(min_count=1); U["refunds"] = w.groupby("uid")["refunds"].sum(min_count=1)
    for c in ["channel", "campaign"]: U[c] = U[c].fillna("(untracked)")
    U["segment"] = U["segment"].fillna("(unknown)"); U["csize"] = U["csize"].fillna("(unknown)"); return U

def ingest_cost(craw, mapping):
    W = []; d = map_cols(craw, mapping, W)
    if "spend" not in d.columns or "campaign" not in d.columns: raise ValueError("Cost file needs campaign and spend (campaign_cost) columns.")
    for c in ["spend", "impressions", "clicks"]:
        d[c] = to_num(d[c]) if c in d.columns else np.nan
        n = int((d[c] < 0).sum())
        if n: d.loc[d[c] < 0, c] = np.nan; W.append("%d negative values in cost file column '%s' set to missing." % (n, c))
    d["campaign"] = d["campaign"].astype(str).str.strip(); d["channel"] = clean_str(d["channel"]) if "channel" in d.columns else "(untracked)"
    return d[["campaign", "channel", "spend", "impressions", "clicks"]], W

def ingest(raw, mapping="", craw=None):
    W = []; df = map_cols(raw, mapping, W); dq = dict(rows_in=len(raw)); cost = None
    if craw is not None:
        cost, w2 = ingest_cost(craw, mapping); W += w2
    ev = all(k in df.columns for k in ("uid", "event")) and ("timestamp" in df.columns or "date" in df.columns)
    if ev:
        tc = "timestamp" if "timestamp" in df.columns else "date"
        E = pd.DataFrame({"uid": df["uid"].astype(str).str.strip(), "raw": df["event"], "ts": to_dt(df[tc])})
        for c in ["channel", "campaign", "segment", "csize"]: E[c] = clean_str(df[c]) if c in df.columns else np.nan
        for c in ["value", "refunds", "lead_score"]: E[c] = to_num(df[c]) if c in df.columns else np.nan
        nb = int(E["ts"].isna().sum()); E = E[E["ts"].notna() & ~E["uid"].str.lower().isin(BLANK)]
        dq["bad_ts"] = nb
        if nb: W.append("%d rows with unparseable timestamps or missing IDs were removed." % nb)
        for c in ["value", "refunds"]:
            n = int((E[c] < 0).sum())
            if n: E.loc[E[c] < 0, c] = np.nan; W.append("%d negative values in '%s' set to missing." % (n, c))
        E["stage"] = E["raw"].map(stage_of); un = E[E["stage"].isna()]["raw"].astype(str).value_counts()
        dq["unmapped"] = int(un.sum())
        if len(un): W.append("%d events with unrecognised names were ignored (e.g. %s)." % (un.sum(), ", ".join(map(str, un.index[:4]))))
        E = E[E["stage"].notna()]; n0_ = len(E); E = E.drop_duplicates(["uid", "stage", "ts", "value"]).reset_index(drop=True); dq["dups"] = n0_ - len(E)
        if dq["dups"]: W.append("%d duplicate event rows removed (same user, stage, timestamp and value)." % dq["dups"])
        if E.empty: raise ValueError("No usable events after validation.")
        dq["miss_ch"] = float(E["channel"].isna().mean()); dq["miss_cp"] = float(E["campaign"].isna().mean())
        if dq["miss_ch"] > 0: W.append("%.1f%% of events have no channel/campaign: reported as '(untracked)', not as drop-off." % (dq["miss_ch"] * 100))
        U = build_users(E); cut = E["ts"].max(); dq["users"] = len(U); dq["ts_min"] = E["ts"].min(); dq["ts_max"] = cut
        dq["wo_lead"] = int((U["t_won"].notna() & U["t_lead"].isna()).sum()); dq["avg_touch"] = float(E[E["stage"] == "visit"].groupby("uid").size().mean()) if (E["stage"] == "visit").any() else np.nan
        if dq["wo_lead"]: W.append("%d customers closed-won with no lead event (offline or untracked sales): visible in open funnels, excluded from closed funnels." % dq["wo_lead"])
        av = {s: bool(U["t_" + s].notna().any()) for s in CHAIN}; av["exposure"] = bool(cost is not None and cost["impressions"].notna().any())
        if cost is None: W.append("No cost file: spend, CPC, CPL, CAC, ROAS and ROI are unavailable (never assumed zero).")
        return dict(mode="event", E=E, U=U, cutoff=cut, cost=cost, A=None, avail=av, dq=dq, W=W, raw_n=len(raw))
    for c in ["spend", "impressions", "clicks", "visits", "leads", "mql", "sql", "opps", "wins", "value", "refunds"]:
        df[c] = to_num(df[c]) if c in df.columns else np.nan
    if not df[["visits", "leads", "mql", "sql", "opps", "wins"]].notna().any().any() and not df["clicks"].notna().any():
        raise ValueError("Could not recognise the data. Event-level files need an ID, event name and timestamp; aggregated files need stage counts (leads, mql, sql, opportunities, closed_won...). Use the mapping box, e.g. leads=form_fills, campaign=ad_name.")
    if not ({"campaign", "channel"} & set(df.columns)): raise ValueError("Aggregated files need a campaign or channel column.")
    if "campaign" not in df.columns: df["campaign"] = df["channel"]; W.append("No campaign column: channel used as campaign.")
    if "channel" not in df.columns: df["channel"] = "(untracked)"
    for c in ["channel", "campaign"]: df[c] = clean_str(df[c]).fillna("(untracked)")
    num = ["spend", "impressions", "clicks", "visits", "leads", "mql", "sql", "opps", "wins", "value", "refunds"]; neg = int((df[num] < 0).sum().sum())
    if neg: df[num] = df[num].mask(df[num] < 0); W.append("%d negative values set to missing." % neg)
    df["date"] = to_dt(df["date"]) if "date" in df.columns else pd.NaT
    n0_ = len(df); df = df.drop_duplicates().reset_index(drop=True); dq["dups"] = n0_ - len(df)
    if dq["dups"]: W.append("%d exact duplicate rows removed." % dq["dups"])
    if df["visits"].isna().all() and df["clicks"].notna().any(): df["visits"] = df["clicks"]; W.append("No visits/sessions column: clicks used as website visits.")
    if df["date"].isna().all(): W.append("No date column: trends and date filters are unavailable.")
    df["revenue"] = df["value"]; df["net_rev"] = df["revenue"] - df["refunds"].fillna(0)
    for a, b in [("leads", "visits"), ("mql", "leads"), ("sql", "mql"), ("opps", "sql"), ("wins", "opps")]:
        n = int((df[a] > df[b]).sum())
        if n: W.append("%d rows where %s exceeds %s (check definitions or timing)." % (n, a, b))
    W.append("Aggregated data: stage-level conversion only. Individual journeys, attribution paths and lead-level models cannot be reconstructed.")
    av = {s: bool(df[ACOL[s]].notna().any()) for s in STG}; dq["users"] = np.nan
    return dict(mode="agg", E=None, U=None, cutoff=None, cost=None, A=df, avail=av, dq=dq, W=W, raw_n=len(raw))

# ------------------------------ funnel engine ------------------------------
def reach(U, stages, closed, win):
    M = {}; prev = None; t0 = None
    for s in stages:
        t = U["t_" + s]; has = t.notna()
        if prev is None: m = has; t0 = t
        elif closed:
            m = has & M[prev] & (t >= U["t_" + prev])
            if win > 0: m = m & ((t - t0) <= pd.Timedelta(days=win))
        else: m = has
        M[s] = m; prev = s
    return M

def event_funnel(U, stages, closed, win, imp):
    ev = [s for s in stages if s != "exposure"]; M = reach(U, ev + (["won"] if "won" not in ev else []), closed, win)
    net = U["value"] - U["refunds"].fillna(0); counts, vals, days = [], [], []
    for i, s in enumerate(stages):
        if s == "exposure": counts.append(imp); vals.append(net.where(M["won"]).sum(min_count=1)); days.append(np.nan); continue
        counts.append(int(M[s].sum())); vals.append(net.where(M[s] & M["won"]).sum(min_count=1))
        nxt = stages[i + 1] if i + 1 < len(stages) else None
        if nxt and nxt != "exposure":
            d = (U["t_" + nxt] - U["t_" + s]).dt.total_seconds() / 86400; d = d[M[nxt]]; days.append(float(d.mean()) if len(d) else np.nan)
        else: days.append(np.nan)
    return counts, vals, days

def funnel_df(stages, names, counts, vals, days, wins, adv, totrev):
    rows = []
    for i, s in enumerate(stages):
        c = counts[i]; p = counts[i - 1] if i else np.nan; conv = sd(c, p) if i else np.nan
        drop = (p - c) if i and not nul(p) and not nul(c) else np.nan; vs = np.nan
        if i and stages[i - 1] != "exposure" and not nul(drop) and drop > 0 and not nul(c) and c > 0 and not nul(wins) and wins > 0 and not nul(adv): vs = drop * min(1.0, wins / c) * adv
        v = vals[i]; rows.append(dict(key=s, stage=names.get(s, LBL[s]), count=c, conv=conv, cum=sd(c, counts[0]), drop=drop, dpct=(1 - conv) if i and not nul(conv) else np.nan,
                                      value=v, rev_per=sd(v if not nul(v) else totrev, c), days=days[i], vs=vs))
    return pd.DataFrame(rows)

def ct_event(U, M, av, cost):
    D = U[["channel", "campaign"]].copy()
    for s in CHAIN: D[ACOL[s]] = M[s].astype(float) if (s in M and av.get(s)) else np.nan
    won = M["won"] if "won" in M else pd.Series(False, index=U.index); v = U["value"]; rf = U["refunds"].fillna(0)
    D["revenue"] = v.where(won); D["refunds"] = rf.where(won & v.notna()); D["net_rev"] = (v - rf).where(won)
    cyc = (U["t_won"] - U["t_lead"]).dt.total_seconds() / 86400; ok = won & cyc.notna(); D["cyc_sum"] = cyc.where(ok); D["cyc_n"] = ok.astype(float)
    g = D.groupby(["channel", "campaign"], as_index=False).sum(min_count=1)
    if v.notna().any():
        for c in ["revenue", "refunds", "net_rev"]: g[c] = g[c].where(g["wins"].fillna(1) != 0, 0.0)
    cg = cost.groupby("campaign")[["spend", "impressions", "clicks"]].sum(min_count=1) if cost is not None else None
    for c in ["spend", "impressions", "clicks"]: g[c] = g["campaign"].map(cg[c]) if cg is not None else np.nan
    return g

def metrics(d, margin):
    d = d.copy()
    for c in BASE:
        if c not in d.columns: d[c] = np.nan
    d["ctr"] = dv(d["clicks"], d["impressions"]); d["cpc"] = dv(d["spend"], d["clicks"]); d["cpl"] = dv(d["spend"], d["leads"])
    d["mql_rate"] = dv(d["mql"], d["leads"]); d["sql_rate"] = dv(d["sql"], d["mql"]); d["opp_rate"] = dv(d["opps"], d["sql"]); d["win_rate"] = dv(d["wins"], d["opps"])
    d["l2c"] = dv(d["wins"], d["leads"]); d["cac"] = dv(d["spend"], d["wins"]); d["adv"] = dv(d["net_rev"], d["wins"]); d["rpl"] = dv(d["net_rev"], d["leads"])
    d["roas"] = dv(d["net_rev"], d["spend"]); d["roi"] = dv(d["net_rev"] * margin - d["spend"], d["spend"])
    d["cycle"] = dv(d["cyc_sum"], d["cyc_n"]) if "cyc_sum" in d.columns else np.nan; return d

def tag_rows(d, minleads=30):
    tot = {c: d[c].sum(min_count=1) for c in ["leads", "mql", "wins", "opps"]}; ol = sd(tot["wins"], tot["leads"]); om = sd(tot["mql"], tot["leads"]); ow = sd(tot["wins"], tot["opps"])
    med = d["leads"].median(); cyc_med = d["cycle"].median() if "cycle" in d else np.nan; tags, fl = [], []
    for _, r in d.iterrows():
        f = []
        if nul(r["leads"]) or r["leads"] < minleads or nul(r["l2c"]) or nul(ol): tags.append("Insufficient data"); fl.append(""); continue
        hv = r["leads"] >= med; lo = r["l2c"] < .8 * ol; hi = r["l2c"] >= 1.2 * ol
        tags.append(("High volume" if hv else "Low volume") + " · " + ("low quality" if lo else ("high quality" if hi else "")) if (lo or hi) else "Mid")
        if not nul(r["mql_rate"]) and not nul(om) and r["mql_rate"] < .8 * om: f.append("weak qualification")
        if not nul(r["win_rate"]) and not nul(ow) and r["win_rate"] < .8 * ow: f.append("weak sales conversion")
        if not nul(r["cycle"]) and not nul(cyc_med) and r["cycle"] > 1.25 * cyc_med: f.append("long sales cycle")
        if r["channel"] == "(untracked)" or r["campaign"] == "(untracked)": f.append("incomplete tracking")
        fl.append(", ".join(f))
    d["tag"] = tags; d["flags"] = fl; return d

def totals(ct): return {c: (float(ct[c].sum(min_count=1)) if c in ct.columns and ct[c].notna().any() else np.nan) for c in BASE}

def do_compute(ch, cp, seg, d1, d2, stg, order, rename, ftype, win, mature, margin):
    if "B" not in G: return note("Load data first.", "warn")
    try:
        B = G["B"]; mode = B["mode"]; W = []; win = int(win or 0); closed = (ftype or "").startswith("Closed"); margin = float(margin if margin is not None else .7)
        ch, cp, seg = list(ch or []), list(cp or []), list(seg or []); D1 = pd.to_datetime(d1, errors="coerce") if d1 else pd.NaT; D2 = pd.to_datetime(d2, errors="coerce") if d2 else pd.NaT
        names = {}
        for p in (rename or "").split(","):
            if "=" in p and p.split("=")[0].strip() in STG: names[p.split("=")[0].strip()] = p.split("=", 1)[1].strip()
        ordl = [x.strip() for x in (order or "").split(",") if x.strip() in STG]; sel = [s for s in STG if B["avail"].get(s) and (not stg or s in stg)]
        stages = [s for s in ordl if s in sel] + [s for s in sel if s not in ordl]
        if "exposure" in stages: stages.remove("exposure"); stages.insert(0, "exposure")
        V = dict(mode=mode, closed=closed, win=win, margin=margin, names=names, flt=dict(ch=ch, cp=cp, seg=seg, d1=d1, d2=d2))
        if mode == "event":
            U = B["U"]; m = pd.Series(True, index=U.index)
            if ch: m &= U["channel"].isin(ch)
            if cp: m &= U["campaign"].isin(cp)
            if seg: m &= U["segment"].isin(seg)
            if pd.notna(D1): m &= U["t_first"] >= D1
            if pd.notna(D2): m &= U["t_first"] <= D2 + pd.Timedelta(days=1)
            U = U[m]; cut = B["cutoff"]
            if U.empty: raise ValueError("No customers match the current filters.")
            imm = (U["t_first"] > cut - pd.Timedelta(days=win)) if win > 0 else pd.Series(False, index=U.index)
            V["n_imm"] = int(imm.sum()); V["n_all"] = len(U)
            if mature and win > 0:
                U = U[~imm]; W.append("%d recent customers (first seen within %d days of the data end) excluded so unfinished journeys are not counted as drop-off." % (V["n_imm"], win))
            elif V["n_imm"] > 0: W.append("%.0f%% of customers are still inside the %d-day window: their downstream stages may not have happened yet, so drop-off is overstated." % (100 * V["n_imm"] / V["n_all"], win))
            if U.empty: raise ValueError("No mature customers remain; shorten the window.")
            cost = B["cost"]
            if cost is not None:
                if ch: cost = cost[cost["channel"].isin(ch)]
                if cp: cost = cost[cost["campaign"].isin(cp)]
                if pd.notna(D1) or pd.notna(D2) or seg: W.append("Cost data has no dates or segments, so spend/impressions are not narrowed by date or segment filters.")
            av = B["avail"]; chain = [s for s in CHAIN if av.get(s)]; M = reach(U, chain, closed, win)
            CT = ct_event(U, M, av, cost); tot = totals(CT); imp = cost["impressions"].sum(min_count=1) if cost is not None else np.nan
            if "exposure" in stages and nul(imp): stages.remove("exposure")
            counts, vals, days = event_funnel(U, stages, closed, win, imp)
            lt = U["t_lead"]; V["imm_leads"] = int((lt.notna() & (lt > cut - pd.Timedelta(days=max(win, 1)))).sum()); V["leads_n"] = int(lt.notna().sum())
            V["offline_wins"] = int((U["t_won"].notna() & U["t_lead"].isna()).sum()); V["offline_rev"] = float((U["value"] - U["refunds"].fillna(0))[U["t_won"].notna() & U["t_lead"].isna()].sum(min_count=1)) if U["value"].notna().any() else np.nan
            tr = pd.DataFrame({"m": U["t_first"].dt.to_period("M").astype(str), "leads": M["lead"].astype(int) if "lead" in M else 0, "wins": M["won"].astype(int) if "won" in M else 0}).groupby("m", as_index=False).sum()
            V.update(U=U, E=B["E"], M=M, cutoff=cut)
        else:
            A = B["A"]; m = pd.Series(True, index=A.index)
            if ch: m &= A["channel"].isin(ch)
            if cp: m &= A["campaign"].isin(cp)
            if pd.notna(D1): m &= A["date"] >= D1
            if pd.notna(D2): m &= A["date"] <= D2
            A = A[m]
            if A.empty: raise ValueError("No rows match the current filters.")
            CT = A.groupby(["channel", "campaign"], as_index=False)[["spend", "impressions", "clicks", "visits", "leads", "mql", "sql", "opps", "wins", "revenue", "refunds"]].sum(min_count=1)
            CT["net_rev"] = CT["revenue"] - CT["refunds"].fillna(0); CT["cyc_sum"] = np.nan; CT["cyc_n"] = np.nan; tot = totals(CT)
            stages = [s for s in stages if not nul(tot[ACOL[s]])]; counts = [tot[ACOL[s]] for s in stages]; vals = [np.nan] * len(stages); days = [np.nan] * len(stages)
            tr = A.dropna(subset=["date"]).assign(m=lambda x: x["date"].dt.to_period("M").astype(str)).groupby("m", as_index=False)[["leads", "wins"]].sum(min_count=1) if A["date"].notna().any() else pd.DataFrame(columns=["m", "leads", "wins"])
            V.update(n_imm=0, n_all=len(A), imm_leads=np.nan, leads_n=tot["leads"], offline_wins=np.nan, offline_rev=np.nan, U=None, E=None, M=None)
        if len(stages) < 2: raise ValueError("Enable at least two stages that exist in the data.")
        adv = sd(tot["net_rev"], tot["wins"]); fun = funnel_df(stages, names, counts, vals, days, tot["wins"], adv, tot["net_rev"])
        CTm = tag_rows(metrics(CT, margin)); CH = metrics(CT.groupby("channel", as_index=False)[[c for c in BASE + ["cyc_sum", "cyc_n"]]].sum(min_count=1), margin)
        V.update(stages=stages, fun=fun, CT=CT, CTm=CTm, CH=CH, tot=tot, adv=adv, tr=tr, W=W)
        G["V"] = V; G.pop("paths", None); G.pop("scn_out", None); G.pop("lead_model", None)
        return note("Analysis updated: %d stages · %s funnel · %s customers/rows in view." % (len(stages), "closed" if closed else "open", format(V["n_all"] if not mature else len(V["U"]) if mode == "event" else V["n_all"], ",")), "ok")
    except Exception as e:
        G.pop("V", None); return note("❌ " + esc(e), "err")

# ------------------------------ scenarios & leakage ------------------------------
KEYS = ["lead", "mql", "sql", "opp", "won"]
def scen_core(tot, d, advp=0.0, scale=1.0):
    ks = [k for k in KEYS if not nul(tot[ACOL[k]]) and tot[ACOL[k]] > 0]
    if "lead" not in ks or "won" not in ks: return None
    c0 = [float(tot[ACOL[k]]) for k in ks]; c1 = [c0[0]]
    for i in range(1, len(ks)):
        r0 = min(1.0, c0[i] / c0[i - 1]); c1.append(c1[-1] * min(1.0, max(0.0, r0 + scale * d.get(ks[i], 0) / 100.0)))
    adv = sd(tot["net_rev"], tot["wins"]); r0v = c0[-1] * adv if not nul(adv) else np.nan; r1v = c1[-1] * adv * (1 + scale * advp / 100.0) if not nul(adv) else np.nan
    return dict(ks=ks, c0=c0, c1=c1, rev0=r0v, rev1=r1v, adv=adv)

def leak_rows(tot, d, advp, margin):
    ks = [k for k in KEYS if not nul(tot[ACOL[k]]) and tot[ACOL[k]] > 0]; rows = []; adv = sd(tot["net_rev"], tot["wins"]); W = tot["wins"]
    nm = dict(mql="Failed qualification (lead → MQL)", sql="Failed sales acceptance (MQL → SQL)", opp="No opportunity created (SQL → opportunity)", won="Opportunities not closed (opportunity → won)")
    fd = dict(mql="lead score, form fields, ICP match, lead source", sql="MQL rejection reasons, sales follow-up time, lead owner", opp="discovery-call outcomes, demo bookings, deal stage history", won="loss reasons, competitor, price, stage ageing, deal size")
    for i in range(1, len(ks)):
        k, p = ks[i], ks[i - 1]; cf, ct_ = tot[ACOL[p]], tot[ACOL[k]]; drop = cf - ct_; wd = min(1.0, W / ct_) if ct_ > 0 else np.nan
        vs = drop * wd * adv if not nul(adv) and drop > 0 else np.nan; add = cf * d.get(k, 0) / 100.0 * wd; rev = add * adv if not nul(adv) else np.nan
        rows.append(dict(point=nm.get(k, k), vol=drop, ev="%s of %s %s did not reach %s (%.1f%% progression)" % (n0(drop), n0(cf), LBL[p].lower() + "s", LBL[k].lower(), 100 * ct_ / cf), vs=vs, wins=add, rev=rev, contrib=rev * margin if not nul(rev) else np.nan, fields=fd.get(k, "")))
    return rows

def scope_tot(V, scope):
    CT = V["CT"]
    if scope and scope.startswith("Channel: "): CT = CT[CT["channel"] == scope[9:]]
    elif scope and scope.startswith("Campaign: "): CT = CT[CT["campaign"] == scope[10:]]
    return totals(CT)

def render_scn(scope, d_mql, d_sql, d_opp, d_won, d_adv):
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data"), empty("No data")
    tot = scope_tot(V, scope); d = dict(mql=d_mql or 0, sql=d_sql or 0, opp=d_opp or 0, won=d_won or 0); s = scen_core(tot, d, d_adv or 0)
    if not s: return note("This scope has no leads or no wins to simulate (needs lead and closed-won counts).", "warn"), empty("No data"), empty("No data")
    ks = s["ks"]; mg = V["margin"]; dr = s["rev1"] - s["rev0"] if not nul(s["rev0"]) else np.nan
    h = "<div class='kg'>" + card("🧲", "Additional qualified leads", n0(s["c1"][1] - s["c0"][1]) if len(ks) > 2 else "—", "scenario − baseline (first qualification stage)") + \
        card("🧭", "Additional expected opportunities", n0(s["c1"][ks.index("opp")] - s["c0"][ks.index("opp")]) if "opp" in ks else "—", "modeled", C["la"]) + \
        card("🏆", "Additional expected wins", fx(s["c1"][-1] - s["c0"][-1], 1), "modeled", C["ok"]) + card("💰", "Projected revenue change", money(dr), "scenario − baseline (net of refunds)", C["ok"] if not nul(dr) and dr >= 0 else C["co"]) + \
        card("📊", "Contribution change", money(dr * mg) if not nul(dr) else "—", "at %.0f%% assumed margin; spend held constant" % (mg * 100), C["orc"]) + "</div>"
    h += note("<b>How it works.</b> Improvement in a stage's conversion (percentage points) is applied to that transition; downstream stages keep their <b>baseline conditional conversion rates</b>, so the gain propagates consistently (no double counting). Rates are clipped to 0–100%%. "
              "Revenue = wins × baseline average deal value × (1 + deal-value change). <b>This is a modeled scenario, not a forecast</b>; confirm with a controlled test.", "warn")
    names = [V["names"].get(k, LBL[k]) for k in ks]
    f1 = go.Figure([go.Bar(y=names, x=s["c0"], orientation="h", name="Baseline", marker_color=C["mu"]), go.Bar(y=names, x=s["c1"], orientation="h", name="Scenario", marker_color=C["vi"])]); f1.update_layout(barmode="group"); f1 = sty(f1, 380, "Baseline vs scenario funnel"); f1.update_yaxes(autorange="reversed")
    sens = [(m, scen_core(tot, d, d_adv or 0, m)) for m in (.5, 1.0, 1.5)]; rows = [dict(m="%gx of assumed improvement" % m, rev=x["rev1"] - x["rev0"], wins=x["c1"][-1] - x["c0"][-1]) for m, x in sens]
    h += "<div class='sec'>Sensitivity range</div>" + tbl(pd.DataFrame(rows), [("m", "Assumption strength"), ("wins", "Additional wins"), ("rev", "Revenue change")], dict(wins=lambda v: fx(v, 1), rev=mcol))
    f2 = sty(go.Figure(go.Bar(x=[r["m"] for r in rows], y=[r["rev"] for r in rows], marker_color=C["orc"])), 320, "Revenue change under different improvement strengths"); f2.update_yaxes(title="Revenue change ($)")
    G["scn_out"] = pd.DataFrame(dict(stage=names, baseline=s["c0"], scenario=s["c1"])); return h, f1, f2

def render_leak(d_mql, d_sql, d_opp, d_won, d_adv):
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data")
    tot = V["tot"]; d = dict(mql=d_mql or 0, sql=d_sql or 0, opp=d_opp or 0, won=d_won or 0); rows = leak_rows(tot, d, d_adv, V["margin"]); h = ""
    if not rows: return note("Leakage analysis needs lead and closed-won counts.", "warn"), empty("No data")
    h += note("<b>Value at stake (upper bound)</b> = dropped records × downstream win rate of those who progressed × average deal value. <b>Modeled opportunity</b> = stage volume × assumed improvement (pp) × downstream win rate × average deal value. "
              "Rows are alternative views of the same leads and <b>must not be added together</b>; the combined scenario below propagates all assumptions once. All figures are modeled, not actual lost revenue.", "warn")
    h += tbl(pd.DataFrame(rows), [("point", "Leak point"), ("vol", "Records lost"), ("ev", "Evidence"), ("vs", "Value at stake (upper bound)"), ("wins", "Extra wins (assumption)"), ("rev", "Modeled revenue opportunity"), ("contrib", "Contribution"), ("fields", "Fields to investigate")],
             dict(vol=n0, vs=money, wins=lambda v: fx(v, 1), rev=money, contrib=money))
    s = scen_core(tot, d, d_adv)
    if s and not nul(s["rev1"]): h += note("<b>Combined scenario (propagated, no double counting):</b> +%s wins and %s revenue (%s contribution at %.0f%% margin)." % (fx(s["c1"][-1] - s["c0"][-1], 1), money(s["rev1"] - s["rev0"]), money((s["rev1"] - s["rev0"]) * V["margin"]), V["margin"] * 100), "ok")
    o = []; CT = V["CTm"]; ol = sd(tot["wins"], tot["leads"]); adv = V["adv"]
    if V["mode"] == "event":
        lm = V["leads_n"]; ml = sd(tot["wins"], tot["leads"]); pend = V["imm_leads"] * ml * adv if not nul(ml) and not nul(adv) else np.nan
        o.append(dict(k="Delayed / not yet matured", v="%s leads are still inside the observation window" % n0(V["imm_leads"]), m=pend, f="lead timestamps, stage timestamps, sales-cycle distribution"))
        o.append(dict(k="Tracking and attribution gaps", v="%s closed-won customers have no lead event (%s revenue); %.1f%% of events lack campaign data" % (n0(V["offline_wins"]), money(V["offline_rev"]), 100 * G["B"]["dq"]["miss_cp"]), m=np.nan, f="UTM capture, CRM lead source, offline-to-online matching"))
    g = CT[(CT["wins"].fillna(0) >= 5) & CT["adv"].notna()]; gap = (g["wins"] * (adv - g["adv"]).clip(lower=0)).sum() if not nul(adv) and len(g) else np.nan
    o.append(dict(k="Low average deal value", v="%d campaigns have below-average deal value" % int((g["adv"] < adv).sum()) if len(g) else "insufficient wins", m=gap, f="deal size by segment, product mix, discounting"))
    m = CT[CT["leads"].fillna(0) >= 30]; med = m["l2c"].median(); cg = (m["leads"] * (med - m["l2c"]).clip(lower=0) * adv).sum() if len(m) and not nul(med) and not nul(adv) else np.nan
    o.append(dict(k="Weak campaigns vs median campaign", v="leads in campaigns converting below the median lead-to-customer rate (%s)" % pc1(med), m=cg, f="lead source quality, targeting, form qualification"))
    h += "<div class='sec'>Other leakage categories (modeled)</div>" + tbl(pd.DataFrame(o), [("k", "Category"), ("v", "Evidence"), ("m", "Modeled value"), ("f", "Fields to investigate")], dict(m=money))
    f = go.Figure(go.Bar(x=[r["point"] for r in rows], y=[r["vs"] for r in rows], marker_color=C["co"], name="Value at stake"))
    f = sty(f, 360, "Value at stake by leak point (upper bound)"); f.update_yaxes(title="$"); return h, f

# ------------------------------ attribution ------------------------------
def wts(n, tdays, method, hl=7.0):
    if n == 1: return [1.0]
    if method == "first": return [1.0] + [0.0] * (n - 1)
    if method == "last": return [0.0] * (n - 1) + [1.0]
    if method == "linear": return [1.0 / n] * n
    if method == "position": return [.5, .5] if n == 2 else [.4] + [.2 / (n - 2)] * (n - 2) + [.4]
    raw = [2 ** (-max(t, 0) / hl) for t in tdays]; s = sum(raw); return [x / s for x in raw]

def build_journeys(E, U, dim):
    ev = E[(E["stage"] == "visit") & E["uid"].isin(U.index)].copy(); ev[dim] = ev[dim].fillna("(untracked)")
    ev = ev.merge(U[["t_won"]], left_on="uid", right_index=True, how="left"); ev = ev[ev["t_won"].isna() | (ev["ts"] <= ev["t_won"])].sort_values(["uid", "ts"], kind="mergesort")
    g = ev.groupby("uid").agg(path=(dim, list), times=("ts", list)); J = U[["t_won"]].join(g); J["conv"] = J["t_won"].notna(); J["rev"] = U["value"] - U["refunds"].fillna(0)
    J = J[J["path"].notna() | J["conv"]].copy(); J["path"] = [p if isinstance(p, list) else ["(no tracked touches)"] for p in J["path"]]; J["times"] = [t if isinstance(t, list) else [tw] for t, tw in zip(J["times"], J["t_won"])]; return J

def attribute(J, method, hl=7.0):
    cr, rv = {}, {}
    for p, t, tw, c, r in zip(J["path"], J["times"], J["t_won"], J["conv"], J["rev"]):
        if not c: continue
        w = wts(len(p), [(tw - x).total_seconds() / 86400 for x in t], method, hl)
        for a, x in zip(p, w):
            cr[a] = cr.get(a, 0.0) + x
            if pd.notna(r): rv[a] = rv.get(a, 0.0) + x * r
    return pd.Series(cr, dtype=float), pd.Series(rv, dtype=float)

def markov(J):
    nconv = int(J["conv"].sum())
    if len(J) < 200 or nconv < 30: return None, None
    states = sorted({s for p in J["path"] for s in p}); ix = {s: i for i, s in enumerate(states)}; k = len(states); st = np.zeros(k); T = np.zeros((k, k)); cv = np.zeros(k); nu = np.zeros(k)
    for p, c in zip(J["path"], J["conv"]):
        st[ix[p[0]]] += 1
        for a, b in zip(p[:-1], p[1:]): T[ix[a], ix[b]] += 1
        (cv if c else nu)[ix[p[-1]]] += 1
    out = T.sum(1) + cv + nu; out[out == 0] = 1; Q = T / out[:, None]; R = cv / out; s0 = st / st.sum()
    def pconv(Q_, R_, s_):
        try: v = np.linalg.solve(np.eye(k) - Q_, R_)
        except np.linalg.LinAlgError: v = np.linalg.lstsq(np.eye(k) - Q_, R_, rcond=None)[0]
        return float(s_ @ v)
    base = pconv(Q, R, s0); re_ = {}
    for s, i in ix.items():
        Q2 = Q.copy(); Q2[i, :] = 0; Q2[:, i] = 0; R2 = R.copy(); R2[i] = 0; s2 = s0.copy(); s2[i] = 0; re_[s] = float(min(1, max(0, 1 - pconv(Q2, R2, s2) / base))) if base > 0 else 0.0
    tot = sum(re_.values()); share = pd.Series({s: (v / tot if tot > 0 else 0) * nconv for s, v in re_.items()}); return share, re_

def render_attr(dim, hl):
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data"), empty("No data")
    if V["mode"] != "event": return note("<b>Event-level path analysis is disabled.</b> The loaded data is aggregated, so individual journeys and attribution paths cannot be reconstructed. Upload event-level data (ID, event name, timestamp, channel/campaign) to enable this tab.", "warn"), empty("Event data required"), empty("Event data required")
    dim = "campaign" if dim == "Campaign" else "channel"; J = build_journeys(V["E"], V["U"], dim); nconv = int(J["conv"].sum())
    if nconv < 5: return note("Fewer than 5 conversions with journeys in view; attribution is not meaningful.", "warn"), empty("Too few conversions"), empty("Too few conversions")
    conv = J[J["conv"]]; avg_t = float(np.mean([len(p) for p in conv["path"]])); h = ""
    if avg_t <= 1.05: h += note("Converting customers have about one tracked touch each: only last-touch-like data exists, so <b>multi-touch attribution cannot be reconstructed</b> and all methods give similar credit.", "warn")
    meths = [("first", "First touch"), ("last", "Last touch"), ("linear", "Linear"), ("position", "Position-based (40/20/40)"), ("time_decay", "Time decay (half-life %gd)" % hl)]
    cr = pd.DataFrame(); rv = pd.DataFrame()
    for k, lab in meths: a, b = attribute(J, k, float(hl)); cr[lab] = a; rv[lab] = b
    mk, re_ = markov(J)
    if mk is not None:
        cr["Markov removal effect"] = mk; ts_ = conv["rev"].sum(min_count=1); rv["Markov removal effect"] = mk / nconv * ts_ if pd.notna(ts_) else np.nan
    cr = cr.fillna(0.0); rv = rv.reindex(cr.index); c2 = cr.reset_index().rename(columns={"index": dim})
    if abs(cr.iloc[:, 0].sum() - nconv) > 1e-6: h += note("Credit reconciliation warning.", "err")
    h += note("<b>Credit reconciles:</b> each method distributes exactly %s conversions (%s revenue). Attribution credit is a <b>rule-based allocation, not causal impact</b>." % (n0(nconv), money(conv["rev"].sum(min_count=1))), "ok")
    h += "<div class='sec'>Credited conversions by attribution method</div>" + tbl(c2.sort_values(c2.columns[2], ascending=False), [(c, c) for c in c2.columns], {c: (lambda v: fx(v, 1)) for c in c2.columns[1:]})
    r2 = rv.fillna(0).reset_index().rename(columns={"index": dim}); h += "<div class='sec'>Credited revenue by method</div>" + tbl(r2.sort_values(r2.columns[2], ascending=False), [(c, c) for c in r2.columns], {c: money for c in r2.columns[1:]})
    h += note("Markov removal effect: " + ("journeys form a first-order chain START → touches → CONVERSION/NULL. A channel's removal effect is the relative drop in modeled conversion probability when its transitions are redirected to NULL; credits are removal effects normalised to total conversions. Needs ≥200 journeys and ≥30 conversions; first-order assumption ignores longer memory." if mk is not None else "not run (needs ≥200 journeys and ≥30 conversions)."), "info")
    J["pstr"] = [" → ".join([p[0]] + [b for a, b in zip(p[:-1], p[1:]) if a != b]) for p in J["path"]]
    tc = J[J["conv"]].groupby("pstr").size().sort_values(ascending=False).head(10); ta = J.groupby("pstr").size()
    pt = pd.DataFrame(dict(path=tc.index, conv=tc.values, total=[ta[p] for p in tc.index])); pt["rate"] = pt["conv"] / pt["total"]; G["paths"] = pt
    h += "<div class='sec'>Most frequent conversion paths</div>" + tbl(pt, [("path", "Path"), ("conv", "Conversions"), ("total", "Journeys with this path"), ("rate", "Conversion rate")], dict(conv=n0, total=n0, rate=pc1))
    nn = J[~J["conv"]].groupby("pstr").size().sort_values(ascending=False).head(8); h += "<div class='sec'>Most common non-converting paths</div>" + tbl(pd.DataFrame(dict(path=nn.index, n=nn.values)), [("path", "Path"), ("n", "Journeys")], dict(n=n0))
    days = [(tw - t[0]).total_seconds() / 86400 for t, tw in zip(conv["times"], conv["t_won"]) if len(t) >= 1]
    h += note("Average touchpoints before conversion: <b>%.1f</b> · median days from first tracked touch to closed-won: <b>%.0f</b> · journeys analysed: %s (%s converted)." % (avg_t, float(np.median(days)) if days else np.nan, n0(len(J)), n0(nconv)), "info")
    pos = {}
    for p in conv["path"]:
        n = len(p)
        for i, a in enumerate(p): pos.setdefault(a, dict(first=0, middle=0, last=0, only=0, assist=0)); pos[a]["only" if n == 1 else ("first" if i == 0 else ("last" if i == n - 1 else "middle"))] += 1; pos[a]["assist"] += 1 if (n > 1 and i < n - 1) else 0
    pp = pd.DataFrame(pos).T.reset_index().rename(columns={"index": dim}); h += "<div class='sec'>Channel contribution by journey position and assisted conversions</div>" + tbl(pp, [(c, c) for c in pp.columns], {c: n0 for c in pp.columns[1:]})
    top = c2.sort_values(c2.columns[2], ascending=False).head(10); f1 = go.Figure([go.Bar(x=top[dim], y=top[c], name=c) for c in top.columns[1:]]); f1.update_layout(barmode="group"); f1 = sty(f1, 400, "Credited conversions: attribution methods compared")
    f2 = sty(go.Figure(go.Bar(x=pt["rate"], y=pt["path"], orientation="h", marker_color=C["orc"])), 380, "Conversion rate of the most frequent converting paths"); f2.update_yaxes(autorange="reversed"); f2.update_xaxes(tickformat=".0%")
    return h, f1, f2

# ------------------------------ lead quality ------------------------------
def lead_frame(V):
    U, E, M = V["U"], V["E"], V["M"]; L = U[U["t_lead"].notna()].copy()
    ev = E[(E["stage"] == "visit") & E["uid"].isin(L.index)].merge(L[["t_lead"]], left_on="uid", right_index=True); ev = ev[ev["ts"] <= ev["t_lead"]]
    g = ev.groupby("uid").agg(pre_visits=("ts", "size"), pre_channels=("channel", "nunique"), first_ts=("ts", "min")); L = L.join(g); L["pre_visits"] = L["pre_visits"].fillna(0); L["pre_channels"] = L["pre_channels"].fillna(0)
    L["days_to_lead"] = (L["t_lead"] - L["first_ts"]).dt.total_seconds() / 86400
    for s in ["mql", "won"]: L[s + "_f"] = M[s].reindex(L.index).astype(int) if s in M else np.nan
    L["mature"] = L["t_lead"] <= V["cutoff"] - pd.Timedelta(days=max(V["win"], 30)); return L

def run_lead_model(target):
    V = G.get("V")
    if not V or V["mode"] != "event": return note("The model needs event-level data. Run the analysis on event data first.", "warn")
    L = lead_frame(V); col = "won_f" if target and "customer" in target else "mql_f"; Lm = L[L["mature"] & L[col].notna()]; y = Lm[col].astype(int).values
    if len(Lm) < 200 or y.sum() < 30 or (len(y) - y.sum()) < 30: return note("Not enough matured, labelled leads for a reliable model (need ≥200 leads with ≥30 positives and ≥30 negatives; found %d leads, %d positives). Use the descriptive tables above; collect more lead outcomes or extend the observation window." % (len(Lm), int(y.sum())), "warn")
    nums = [c for c in ["lead_score", "pre_visits", "pre_channels", "days_to_lead"] if Lm[c].notna().mean() > .4]; cats = ["channel", "segment", "csize"]
    X = Lm[nums + cats].copy()
    for c in cats: X[c] = X[c].astype(str)
    tr, te = train_test_split(np.arange(len(Lm)), test_size=.25, stratify=y, random_state=42)
    def pre(sc): return ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median"))] + ([("s", StandardScaler())] if sc else [])), nums), ("c", OneHotEncoder(handle_unknown="ignore"), cats)])
    rows = []
    for nm, mdl, sc in [("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced"), True), ("Gradient Boosting", HistGradientBoostingClassifier(max_depth=3, learning_rate=.06, max_iter=120, random_state=42), False)]:
        p = Pipeline([("p", pre(sc)), ("m", mdl)]).fit(X.iloc[tr], y[tr]); pr = p.predict_proba(X.iloc[te])[:, 1]; k = max(1, int(.2 * len(te))); top = np.argsort(-pr)[:k]
        rows.append(dict(m=nm, roc=roc_auc_score(y[te], pr), pr=average_precision_score(y[te], pr), cap=y[te][top].sum() / max(1, y[te].sum()), lift=(y[te][top].mean() / max(y[te].mean(), 1e-9))))
    h = note("Predictive model for <b>%s</b> using only information available at lead creation (lead score, pre-lead visits and channels, days from first touch to lead, source channel, industry, company size). Matured leads only (lead older than %d days before data end); stratified 75/25 split; preprocessing fitted on training data only. Test set: %d leads (%.1f%% positive)." % ("becoming a customer" if col == "won_f" else "becoming a marketing-qualified lead", max(V["win"], 30), len(te), 100 * y[te].mean()), "info")
    h += tbl(pd.DataFrame(rows), [("m", "Model"), ("roc", "Test ROC-AUC"), ("pr", "Test PR-AUC"), ("cap", "Positives captured in top 20% of scores"), ("lift", "Lift in top 20%")], dict(roc=lambda v: fx(v, 3), pr=lambda v: fx(v, 3), cap=pc0, lift=lambda v: fx(v, 2) + "x"))
    G["lead_model"] = rows; return h + note("These are <b>predictive results on %s</b>, not causal findings and not evidence of real-world accuracy%s. Future events (MQL/SQL/opportunity stages) are never used as features." % ("synthetic data" if G["B"].get("demo") else "your data", ""), "warn")

def render_leadq():
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data")
    CT = V["CTm"]; h = note("Qualification rate = MQL ÷ leads; sales acceptance = SQL ÷ MQL; lead-to-customer = wins ÷ leads. Campaigns with fewer than 30 leads are labelled insufficient data. Tags compare each campaign with the overall lead-to-customer rate (±20%) and the median lead volume.", "info")
    d = CT.sort_values("leads", ascending=False)
    h += tbl(d, [("campaign", "Campaign"), ("leads", "Leads"), ("mql_rate", "MQL rate"), ("sql_rate", "SQL rate"), ("win_rate", "Win rate"), ("l2c", "Lead→customer"), ("rpl", "Revenue / lead"), ("tag", "Quality tag"), ("flags", "Flags")],
             dict(leads=n0, mql_rate=cond(pc1, sd(V["tot"]["mql"], V["tot"]["leads"])), sql_rate=pc1, win_rate=cond(pc1, sd(V["tot"]["wins"], V["tot"]["opps"])), l2c=cond(pc1, sd(V["tot"]["wins"], V["tot"]["leads"])), rpl=money, tag=tagf))
    fig = empty("Lead-level data required")
    if V["mode"] != "event": return h + note("Lead-level analysis (scores, attributes, models) needs lead/event-level data; aggregated data supports only the campaign table above.", "warn"), fig
    L = lead_frame(V); Lm = L[L["mature"]]; h += note("Descriptive patterns use matured leads only (%s of %s leads); recent leads have not had time to qualify." % (n0(len(Lm)), n0(len(L))), "info")
    if Lm["lead_score"].notna().sum() >= 50:
        Lm = Lm.assign(band=pd.cut(Lm["lead_score"], [0, 20, 40, 60, 80, 100]).astype(str)); g = Lm.groupby("band").agg(n=("mql_f", "size"), mql=("mql_f", "mean"), won=("won_f", "mean")).reset_index()
        h += "<div class='sec'>Qualification and conversion by lead-score band</div>" + tbl(g, [("band", "Lead score"), ("n", "Leads"), ("mql", "MQL rate"), ("won", "Customer rate")], dict(n=n0, mql=pc1, won=pc1))
        fig = sty(go.Figure([go.Bar(x=g["band"], y=g["mql"], name="MQL rate", marker_color=C["vi"]), go.Bar(x=g["band"], y=g["won"], name="Customer rate", marker_color=C["orc"])]), 360, "Lead-score bands: downstream progression"); fig.update_layout(barmode="group"); fig.update_yaxes(tickformat=".0%")
    for c, nm in [("segment", "industry"), ("csize", "company size")]:
        g = Lm.groupby(c).agg(n=("mql_f", "size"), mql=("mql_f", "mean"), won=("won_f", "mean")).reset_index(); g = g[g["n"] >= 20]
        if len(g): h += "<div class='sec'>Progression by %s</div>" % nm + tbl(g, [(c, nm.capitalize()), ("n", "Leads"), ("mql", "MQL rate"), ("won", "Customer rate")], dict(n=n0, mql=pc1, won=pc1))
    return h + note("Descriptive patterns show association only. Press the button below to train an optional predictive model.", "info"), fig

# ------------------------------ executive, funnel, channels ------------------------------
def render_exec():
    V = G.get("V")
    if not V: return ("", empty("No data"), empty("No data"), empty("No data"), "")
    t = V["tot"]; f = V["fun"]; CT = V["CTm"]; mg = V["margin"]; ft = f[f["key"] != "exposure"]; tr_ = ft.iloc[1:] if len(ft) > 1 else ft
    big = tr_.sort_values("drop", ascending=False).iloc[0] if len(tr_) and tr_["drop"].notna().any() else None; bigv = tr_.sort_values("vs", ascending=False).iloc[0] if len(tr_) and tr_["vs"].notna().any() else None
    ok = CT[(CT["leads"].fillna(0) >= 30) & CT["rpl"].notna()].sort_values("rpl"); s = scen_core(t, dict(mql=5), 0); opp = (s["rev1"] - s["rev0"]) if s and not nul(s["rev1"]) else np.nan
    cards = [card("💸", "Campaign spend", money(t["spend"]), "from cost data" if not nul(t["spend"]) else "no cost data provided"), card("👁️", "Visits" if not nul(t["visits"]) else "Impressions", n0(t["visits"] if not nul(t["visits"]) else t["impressions"]), "unique visitors/sessions in view", C["la"]),
             card("🧲", "Leads", n0(t["leads"]), "lead-generation events", C["orc"]), card("✅", "Qualified leads (MQL / SQL)", "%s / %s" % (n0(t["mql"]), n0(t["sql"])), "marketing / sales qualified", C["la"]),
             card("🧭", "Opportunities", n0(t["opps"]), "pipeline created", C["vi"]), card("🏆", "Closed-won customers", n0(t["wins"]), "%s funnel" % ("closed" if V["closed"] else "open"), C["ok"]),
             card("💰", "Recognised revenue", money(t["net_rev"]), "net of refunds", C["ok"]), card("🔁", "End-to-end conversion", pc1(sd(t["wins"], t["leads"])), "closed-won ÷ leads", C["am"]),
             card("🕳️", "Largest drop-off (volume)", esc(big["stage"]) if big is not None else "—", ("%s lost · %.0f%% drop" % (n0(big["drop"]), 100 * big["dpct"])) if big is not None else "", C["co"]),
             card("💎", "Largest value drop-off", esc(bigv["stage"]) if bigv is not None else "—", ("%s at stake (upper bound)" % money(bigv["vs"])) if bigv is not None else "needs revenue and wins", C["co"]),
             card("🥇", "Strongest campaign (revenue/lead)", esc(ok.iloc[-1]["campaign"]) if len(ok) else "—", money(ok.iloc[-1]["rpl"]) + " per lead" if len(ok) else "", C["ok"]), card("⚠️", "Weakest campaign (revenue/lead)", esc(ok.iloc[0]["campaign"]) if len(ok) else "—", money(ok.iloc[0]["rpl"]) + " per lead" if len(ok) else "", C["co"]),
             card("🔮", "Modeled opportunity (+5pp qualification)", money(opp), "assumption-based, editable in tab H", C["orc"])]
    k = ("<div class='kg'>" + "".join(cards) + "</div>")
    if G["B"].get("demo"): k = note("<b>ILLUSTRATIVE SYNTHETIC DATA</b>: fictional marketing data for demonstration. Nothing here describes a real company.", "warn") + k
    k += "".join(note(esc(w), "warn") for w in (V["W"] + G["B"]["W"])[:6]) + note("Funnel drop-off shows where progression stops being <b>observed</b>; it does not prove why. Attributed revenue is not proof of incrementality.", "info")
    f1 = go.Figure(go.Funnel(y=f["stage"], x=f["count"], textinfo="value+percent previous", marker=dict(color=[C["vi"], C["la"], C["orc"], C["co"]] * 3))); f1 = sty(f1, 400, "Customer funnel")
    tr = V["tr"]; f2 = empty("No dates available for a trend")
    if len(tr) > 1: f2 = sty(go.Figure([go.Bar(x=tr["m"], y=tr["leads"], name="Leads", marker_color=C["la"]), go.Scatter(x=tr["m"], y=dv(tr["wins"], tr["leads"]), name="Lead→customer rate", yaxis="y2", line=dict(color=C["orc"], width=3))]), 400, "Monthly leads and lead-to-customer rate (by first-seen cohort)"); f2.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
    d = CT[CT["leads"].fillna(0) > 0]; f3 = go.Figure()
    for tg, col in TAGC.items():
        x = d[d["tag"] == tg]
        if len(x): f3.add_trace(go.Scatter(x=x["leads"], y=x["l2c"], mode="markers+text", text=x["campaign"], textposition="top center", name=tg, marker=dict(size=12 + 30 * x["net_rev"].fillna(0) / max(d["net_rev"].max(), 1), color=col, opacity=.85)))
    f3 = sty(f3, 440, "Campaigns: lead volume vs lead-to-customer rate (bubble = revenue)"); f3.update_xaxes(title="Leads"); f3.update_yaxes(title="Lead → customer", tickformat=".0%")
    r = recs(); ins = "<div class='sec'>AI insight panel: recommended actions</div>" + "".join(rec_html(x) for x in r[:3]); return k, f1, f2, f3, ins

def render_funnel():
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data"), empty("No data")
    f = V["fun"]; h = note("<b>Closed funnel:</b> a customer counts at a stage only if they completed every earlier enabled stage, in order, within the time window. <b>Open funnel:</b> a customer counts at any stage where the event occurred, regardless of earlier stages (shows tracking gaps and offline sales). "
                           "Current: <b>%s</b>%s. Event data counts unique users; aggregated data uses the supplied counts only." % ("closed" if V["closed"] else "open", (", %d-day window" % V["win"]) if V["win"] > 0 and V["closed"] else ""), "info")
    if V["mode"] == "agg": h += note("Aggregated data: stage conversion uses supplied totals. Individual sequences cannot be reconstructed, and 'avg days to next stage' is unavailable.", "warn")
    if f["key"].eq("exposure").any(): h += note("Exposure → visit compares impressions with unique visitors (different units): shown for context, not valued.", "info")
    h += tbl(f, [("stage", "Stage"), ("count", "Records"), ("conv", "Conv. from previous"), ("cum", "Cumulative conv."), ("drop", "Drop-off count"), ("dpct", "Drop-off %"), ("value", "Downstream revenue of stage entrants"), ("rev_per", "Revenue per record"), ("days", "Avg days to next stage"), ("vs", "Value at stake (upper bound)")],
             dict(count=n0, conv=pc1, cum=pc1, drop=n0, dpct=pc1, value=money, rev_per=money, days=lambda v: fx(v, 1), vs=money))
    h += note("<b>Value at stake</b> = dropped records × win rate of those who progressed × average deal value: an upper bound that assumes dropped records would convert like survivors. The largest % drop is not automatically the biggest opportunity: compare with the value column.", "warn")
    t = f.iloc[1:]; fig = go.Figure([go.Bar(x=t["stage"], y=t["vs"], name="Value at stake ($)", marker_color=C["vi"]), go.Scatter(x=t["stage"], y=t["dpct"], name="Drop-off %", yaxis="y2", mode="markers+lines", marker=dict(size=10, color=C["co"]))])
    fig = sty(fig, 400, "Drop-off: % lost vs value at stake by transition"); fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False)); fig.update_xaxes(tickangle=-20)
    ff = go.Figure(go.Funnel(y=f["stage"], x=f["count"], textinfo="value+percent initial+percent previous", marker=dict(color=[C["vi"], C["la"], C["orc"], C["co"]] * 3))); return h, sty(ff, 420, "Funnel visualisation"), fig

def render_channels():
    V = G.get("V")
    if not V: return note("Run the analysis first.", "info"), empty("No data")
    t = V["tot"]; ol = sd(t["wins"], t["leads"]); mg = V["margin"]
    cols = [("leads", "Leads"), ("cpl", "Cost / lead"), ("mql_rate", "MQL rate"), ("sql_rate", "SQL rate"), ("opps", "Opps"), ("win_rate", "Win rate"), ("wins", "Wins"), ("l2c", "Lead→cust"), ("cac", "CAC"), ("adv", "Avg deal"), ("net_rev", "Revenue"), ("rpl", "Rev / lead"), ("roas", "ROAS"), ("roi", "ROI @ %.0f%% margin" % (mg * 100)), ("cycle", "Cycle (days)")]
    fm = dict(leads=n0, cpl=money, mql_rate=cond(pc1, sd(t["mql"], t["leads"])), sql_rate=pc1, opps=n0, win_rate=cond(pc1, sd(t["wins"], t["opps"])), wins=n0, l2c=cond(pc1, ol), cac=money, adv=money, net_rev=money, rpl=money, roas=x2, roi=pc0, cycle=lambda v: fx(v, 0))
    h = note("<b>Formulas.</b> CTR = clicks ÷ impressions (needs both) · CPC = spend ÷ clicks · CPL = spend ÷ leads · MQL rate = MQL ÷ leads · SQL rate = SQL ÷ MQL · lead→customer = wins ÷ leads · CAC = spend ÷ wins · revenue/lead = net revenue ÷ leads · ROAS = net revenue ÷ spend · "
             "profit ROI = (net revenue × <b>assumed margin</b> − spend) ÷ spend. Metrics that need unavailable inputs show “—”; nothing is labelled profitable without spend and revenue.", "info")
    ch = V["CH"].sort_values("net_rev", ascending=False, na_position="last"); h += "<div class='sec'>Channel comparison (downstream outcomes, not just leads)</div>" + tbl(ch, [("channel", "Channel")] + cols, fm)
    cp = V["CTm"].sort_values("net_rev", ascending=False, na_position="last"); h += "<div class='sec'>Campaign comparison</div>" + tbl(cp, [("channel", "Channel"), ("campaign", "Campaign")] + cols + [("tag", "Quality tag"), ("flags", "Flags")], dict(fm, tag=tagf))
    if nul(t["spend"]): h += note("No cost data: cost-based metrics are unavailable. Provide a cost file (campaign, campaign_cost, impressions, clicks).", "warn")
    d = V["CH"]; fig = go.Figure([go.Bar(x=d["channel"], y=d["leads"], name="Leads", marker_color=C["la"]), go.Bar(x=d["channel"], y=d["wins"], name="Closed-won", marker_color=C["ok"])]); fig = sty(fig, 380, "Leads vs closed-won customers by channel"); fig.update_layout(barmode="group"); return h, fig

# ------------------------------ AI strategist ------------------------------
ACT = dict(mql="Align lead-qualification criteria between marketing and sales, add qualifying fields to forms, and send low-score leads to nurture instead of sales.", sql="Agree a sales follow-up SLA (e.g. one business day) and capture structured rejection reasons for MQLs.",
           opp="Review the SQL-to-opportunity handoff: discovery-call quality, demo scheduling speed and pricing blockers.", won="Improve late-stage conversion with pipeline reviews, win/loss interviews and objection handling.")
KPIS = dict(mql="MQL rate by campaign", sql="MQL→SQL acceptance and time-to-first-contact", opp="SQL→opportunity rate", won="Opportunity win rate and average deal value")
def conf_of(n): return "Low (only %s wins)" % n0(n) if nul(n) or n < 30 else ("Moderate (%s wins)" % n0(n) if n < 100 else "Moderate–high (%s wins)" % n0(n))
def recs():
    V = G.get("V"); out = []
    if not V: return out
    t = V["tot"]; rows = leak_rows(t, dict(mql=5, sql=5, opp=5, won=5), 0, V["margin"]); rows = [r for r in rows if not nul(r["vs"])]
    if rows:
        r = sorted(rows, key=lambda x: -x["vs"])[0]; k = next(k for k in ACT if r["point"].startswith({"mql": "Failed qual", "sql": "Failed sales", "opp": "No opportunity", "won": "Opportunities"}[k]))
        out.append(dict(f="Fix first: " + r["point"], e=r["ev"] + "; value at stake (upper bound) " + money(r["vs"]), i="Modeled +5pp improvement: %s extra wins, %s revenue" % (fx(r["wins"], 1), money(r["rev"])), a=ACT[k], s="Dropped records would convert like survivors; +5pp improvement is an assumption; attribution not causal", c=conf_of(t["wins"]), k=KPIS[k]))
    CT = V["CTm"]; hl = CT[CT["tag"] == "High volume · low quality"]
    if len(hl):
        n = hl["leads"].sum(); out.append(dict(f="High-volume, low-quality campaigns: " + ", ".join(hl["campaign"]), e="%s leads (%s of all) convert at %s vs %s overall" % (n0(n), pc0(sd(n, t["leads"])), pc1(sd(hl["wins"].sum(), n)), pc1(sd(t["wins"], t["leads"]))),
                                               i="Spend %s on these campaigns" % money(hl["spend"].sum(min_count=1)), a="Tighten targeting and form qualification, shift budget to higher-quality sources, or test lower-volume/higher-intent offers.", s="Campaign attribution is by lead source; quality may also depend on sales follow-up", c=conf_of(hl["wins"].sum()), k="Lead→customer rate and revenue per lead by campaign"))
    hq = CT[CT["tag"].isin(["Low volume · high quality", "High volume · high quality"])].sort_values("rpl", ascending=False).head(3)
    if len(hq): out.append(dict(f="High-quality sources to protect or scale: " + ", ".join(hq["campaign"]), e="; ".join("%s: %s per lead, lead→customer %s" % (r["campaign"], money(r["rpl"]), pc1(r["l2c"])) for _, r in hq.iterrows()), i="Highest downstream revenue efficiency", a="Test incremental budget in steps with a holdout; watch for diminishing returns.", s="Past efficiency may not hold at higher spend; not causal", c=conf_of(hq["wins"].sum()), k="Marginal CAC and revenue per lead"))
    if V["mode"] == "event":
        gp = G["B"]["dq"]
        if gp["miss_cp"] > .03 or V["offline_wins"] > 0: out.append(dict(f="Tracking gaps hide where value comes from", e="%.1f%% of events lack campaign data; %s closed-won customers (%s) have no lead event" % (gp["miss_cp"] * 100, n0(V["offline_wins"]), money(V["offline_rev"])), i="Revenue that cannot be tied to a campaign", a="Enforce UTM capture, sync CRM lead source, and match offline deals to marketing contacts.", s="Untracked events may be organic, direct or offline", c="Moderate", k="% of leads/wins with campaign attribution"))
        if V["n_imm"] and V["n_imm"] / V["n_all"] > .1: out.append(dict(f="Part of the funnel has not matured yet", e="%s customers are still inside the %d-day window" % (n0(V["n_imm"]), V["win"]), i="Pending revenue modeled in tab F", a="Judge recent cohorts only after the typical sales cycle has elapsed.", s="Window length is an assumption", c="Moderate", k="Conversion by cohort age"))
    return out
def rec_html(x):
    rows = [("Finding", x["f"]), ("Evidence", x["e"]), ("Potential impact", x["i"]), ("Recommended action", x["a"]), ("Assumptions", x["s"]), ("Confidence", x["c"]), ("KPI to monitor", x["k"])]
    return "<div class='tw'><table>" + "".join("<tr><th style='width:150px'>%s</th><td>%s</td></tr>" % (a, esc(b)) for a, b in rows) + "</table></div>"
def render_strat():
    if "V" not in G: return note("Run the analysis first.", "info")
    r = recs(); return "".join(rec_html(x) for x in r) or note("Not enough data for recommendations.", "warn")

def t_summary():
    V = G["V"]; t = V["tot"]; return "Spend %s; visits %s; leads %s; MQL %s; SQL %s; opportunities %s; closed-won %s; net revenue %s; lead-to-customer %s." % (money(t["spend"]), n0(t["visits"]), n0(t["leads"]), n0(t["mql"]), n0(t["sql"]), n0(t["opps"]), n0(t["wins"]), money(t["net_rev"]), pc1(sd(t["wins"], t["leads"])))
def t_drop():
    f = G["V"]["fun"].iloc[1:]; return "Transitions: " + "; ".join("%s: %s conversion, %s lost, value at stake %s" % (r["stage"], pc1(r["conv"]), n0(r["drop"]), money(r["vs"])) for _, r in f.iterrows())
def t_camp():
    CT = G["V"]["CTm"]; m = CT[CT["leads"].fillna(0) >= 30].sort_values("rpl", ascending=False); return "Revenue per lead (leads≥30): " + "; ".join("%s %s [%s]" % (r["campaign"], money(r["rpl"]), r["tag"]) for _, r in m.iterrows())
def t_leadq():
    CT = G["V"]["CTm"]; w = CT[CT["flags"].str.contains("weak qualification", na=False)]; return "Campaigns with weak qualification (<80%% of overall MQL rate): %s." % (", ".join("%s (MQL rate %s)" % (r["campaign"], pc1(r["mql_rate"])) for _, r in w.iterrows()) or "none")
def t_track():
    V = G["V"]; CT = V["CTm"]; u = CT[CT["flags"].str.contains("incomplete tracking", na=False)]; s = "Untracked rows: " + (", ".join("%s (%s leads, %s wins)" % (r["campaign"], n0(r["leads"]), n0(r["wins"])) for _, r in u.iterrows()) or "none") + "."
    return s + (" %.1f%% of events lack campaign data; %s wins have no lead event." % (G["B"]["dq"]["miss_cp"] * 100, n0(V["offline_wins"])) if V["mode"] == "event" else "")
def t_paths():
    if G["V"]["mode"] != "event": return "Journey-path analysis is unavailable for aggregated data."
    if "paths" not in G: render_attr("Channel", 7)
    p = G.get("paths"); return "Top converting paths: " + "; ".join("%s (%s conv, %s)" % (r["path"], n0(r["conv"]), pc1(r["rate"])) for _, r in p.head(5).iterrows()) if p is not None else "No paths."
def t_attr():
    V = G["V"]
    if V["mode"] != "event": return "Attribution is unavailable for aggregated data."
    J = build_journeys(V["E"], V["U"], "channel"); a = attribute(J, "first")[0]; b = attribute(J, "last")[0]; c = attribute(J, "linear")[0]
    return "Credited conversions by channel (first / last / linear): " + "; ".join("%s %.0f / %.0f / %.0f" % (k, a.get(k, 0), b.get(k, 0), c.get(k, 0)) for k in sorted(set(a.index) | set(b.index), key=lambda k: -c.get(k, 0))[:6])
def t_opp(pp=5.0):
    V = G["V"]; s = scen_core(V["tot"], dict(mql=pp), 0)
    return "If qualification (lead→MQL) improved by %g percentage points with downstream rates unchanged: %s extra wins and %s revenue (modeled, not a forecast)." % (pp, fx(s["c1"][-1] - s["c0"][-1], 1), money(s["rev1"] - s["rev0"])) if s else "Scenario unavailable (needs leads and wins)."
def t_why():
    CT = G["V"]["CTm"]; w = CT[CT["tag"].str.contains("low quality", na=False)]
    return "Hypotheses (not established causes): " + ("; ".join("%s: %s, MQL rate %s, win rate %s" % (r["campaign"], r["tag"], pc1(r["mql_rate"]), pc1(r["win_rate"])) for _, r in w.iterrows()) or "no clearly low-quality campaigns") + ". Possible reasons: weak targeting, form fields that attract unqualified leads, slow sales follow-up, long sales cycles. Test with experiments."

def check_numbers(ans, ctx):
    cn = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ctx.replace(",", ""))]; bad = []
    for s in re.findall(r"\d+(?:\.\d+)?", ans.replace(",", "")):
        v = float(s)
        if v <= 10 or 1990 <= v <= 2100: continue
        if not any(abs(v - c) <= max(.06, .015 * abs(c)) for c in cn): bad.append(s)
    return sorted(set(bad))

def ask(q, key, model):
    if "V" not in G: return "Run the analysis first."
    q = (q or "").strip() or "Where are we losing the most value?"; ql = q.lower(); parts = []
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percentage|pp|points|%)", ql)
    if re.search(r"lose|leak|drop|fix first|stage|value", ql): parts += [("stage_dropoff", t_drop()), ("revenue_opportunity", t_opp())]
    if re.search(r"qualif|rarely|quality", ql): parts.append(("lead_quality", t_leadq()))
    if re.search(r"channel|campaign|strongest|revenue", ql): parts.append(("campaign_comparison", t_camp()))
    if re.search(r"why|fail", ql): parts.append(("hypotheses", t_why()))
    if re.search(r"path|journey", ql): parts.append(("journey_paths", t_paths()))
    if "attribution" in ql: parts.append(("attribution_comparison", t_attr()))
    if re.search(r"track", ql): parts.append(("tracking_gaps", t_track()))
    if re.search(r"what would happen|improv|percentage|scenario", ql): parts.append(("scenario_simulation", t_opp(float(m.group(1)) if m else 5.0)))
    if "report" in ql: parts.append(("report", "Use 'Generate executive report' in tab I."))
    if not parts: parts = [("funnel_summary", t_summary()), ("stage_dropoff", t_drop()), ("revenue_opportunity", t_opp())]
    ctx = "\n".join("[%s] %s" % (n, t) for n, t in parts); base = "### Answer (deterministic, computed by tools)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Tools used: " + ", ".join(n for n, _ in parts) + ". Every number is computed from the current data.*"
    k = (key or "").strip() or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        try:
            from google.colab import userdata; k = userdata.get("ANTHROPIC_API_KEY")
        except Exception: k = None
    if not k: return base + "\n\n*No API key set: rule-based agent used. Add a key (session only) for natural-language synthesis.*"
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": (model or "claude-sonnet-5-5").strip(), "max_tokens": 900, "system": "You are a marketing funnel analyst. Answer ONLY from the TOOL RESULTS. Never invent figures, campaigns or causes. Label explanations as hypotheses unless the data establishes them. Be concise.",
                                "messages": [{"role": "user", "content": "TOOL RESULTS:\n" + ctx + "\n\nQUESTION: " + q}]}, timeout=60)
        if r.status_code != 200: return base + "\n\n*LLM call failed (HTTP %d); showing deterministic answer.*" % r.status_code
        ans = "".join(b.get("text", "") for b in r.json().get("content", [])); bad = check_numbers(ans, ctx)
        return "### LLM-written answer (grounded in tool results)\n" + ans + (("\n\n⚠️ Numbers not found in tool results: " + ", ".join(bad) + ". Treat as unverified.") if bad else "\n\n✅ Numeric check: all figures match tool results.") + "\n\n---\n" + base
    except Exception as e: return base + "\n\n*LLM call failed (%s); showing deterministic answer.*" % type(e).__name__

# ------------------------------ data quality, tests, export ------------------------------
def render_dq():
    V = G.get("V"); B = G.get("B")
    if not V: return note("Run the analysis first.", "info"), ""
    dq = B["dq"]; h = note("Dataset: <b>%s</b> mode, %s rows read. " % (B["mode"], n0(B["raw_n"])) + ("<b>ILLUSTRATIVE SYNTHETIC DATA</b>: not real company results." if B.get("demo") else "Uploaded data; never written into the project ZIP."), "ok")
    h += "".join(note(esc(w), "warn") for w in B["W"] + V["W"]) or note("No data-quality issues detected.", "ok")
    rows = [("Duplicate rows/events removed", n0(dq.get("dups"))), ("Unparseable timestamps / IDs removed", n0(dq.get("bad_ts"))), ("Unrecognised events ignored", n0(dq.get("unmapped"))), ("Events missing channel/campaign", pc1(dq.get("miss_cp")))]
    if B["mode"] == "event":
        rows += [("Unique customers/visitors", n0(dq["users"])), ("Customers closed-won without a lead event", n0(dq["wo_lead"])), ("Average tracked touches per visitor", fx(dq["avg_touch"], 1)), ("Observation window", "%s to %s" % (dq["ts_min"].date(), dq["ts_max"].date())), ("Customers inside the sales-cycle window", "%s of %s" % (n0(V["n_imm"]), n0(V["n_all"])))]
    rows += [("Revenue / deal value available", "yes" if not nul(V["tot"]["net_rev"]) else "no"), ("Cost data available", "yes" if not nul(V["tot"]["spend"]) else "no")]
    h += "<div class='sec'>Coverage and quality</div>" + tbl(pd.DataFrame(rows, columns=["a", "b"]), [("a", "Check"), ("b", "Result")])
    h += tbl(pd.DataFrame([("Event-level", "ID + event name + timestamp (+ channel, campaign, deal_value, lead_score, industry, company_size)", "funnel, journeys, attribution, lead quality, cohorts"), ("Aggregated", "channel/campaign + stage counts (+ spend, revenue, date)", "funnel, channel and campaign comparison, leakage, scenarios"), ("Cost file (optional)", "campaign, campaign_cost, impressions, clicks", "CPC, CPL, CAC, ROAS, ROI")], columns=["a", "b", "c"]), [("a", "Data type"), ("b", "Required / optional fields"), ("c", "Enables")])
    th = "<div class='sec'>Self-test results (executed at start-up)</div>" + "".join(note(("✅ " if ok else "❌ ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS); return h, th

def _mini_E():
    ts = lambda d: pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
    rows = [("a", -3, "visit", "Social", "P1", np.nan), ("a", 0, "visit", "Search", "S1", np.nan), ("a", 1, "lead", "Search", "S1", np.nan), ("a", 5, "mql", "Search", "S1", np.nan), ("a", 9, "won", "Search", "S1", 100.0),
            ("b", 0, "visit", "Social", "P1", np.nan), ("b", 2, "lead", "Social", "P1", np.nan), ("b", 6, "mql", "Social", "P1", np.nan), ("c", 0, "visit", "Social", "P1", np.nan), ("c", 3, "lead", "Social", "P1", np.nan),
            ("d", 0, "visit", "Email", "E1", np.nan), ("o", 4, "won", None, None, 50.0)]
    E = pd.DataFrame(rows, columns=["uid", "d", "stage", "channel", "campaign", "value"]); E["ts"] = E["d"].map(ts)
    for c in ["refunds", "lead_score", "segment", "csize"]: E[c] = np.nan
    return E

def run_selftests(verbose=True):
    T = []
    def chk(g, name, fn):
        try: ok, det = fn(); T.append((g, name, bool(ok), det))
        except Exception as e: T.append((g, name, False, type(e).__name__ + ": " + str(e)[:80]))
    E0 = _mini_E(); U0 = build_users(E0); av = {s: True for s in CHAIN}; st = ["visit", "lead", "mql", "won"]
    def t_counts():
        M = reach(U0, st, True, 0); c = [int(M[s].sum()) for s in st]; f = funnel_df(st, {}, c, [np.nan] * 4, [np.nan] * 4, 1, 100.0, 100.0)
        return c == [4, 3, 2, 1] and abs(f["conv"].iloc[1] - .75) < 1e-9 and f["drop"].iloc[1:].tolist() == [1, 1, 1], "counts %s, conv %s, drop %s" % (c, [round(x, 3) for x in f["conv"].iloc[1:]], f["drop"].iloc[1:].tolist())
    def t_closed():
        o = int(reach(U0, st, False, 0)["won"].sum()); c = int(reach(U0, st, True, 0)["won"].sum()); return o == 2 and c == 1, "open wins %d vs closed wins %d (offline sale only in open)" % (o, c)
    def t_window():
        M = reach(U0, st, True, 7); c = [int(M[s].sum()) for s in st]; return c == [4, 3, 1, 0], "7-day window counts %s" % c
    def t_dups():
        raw = pd.DataFrame(dict(timestamp=["2026-01-01", "2026-01-01", "2026-01-02", "bad", "2026-01-03"], visitor_id=["u1", "u1", "u1", "u2", "u3"], event_name=["visit", "visit", "lead", "visit", "closed_won"], deal_value=[np.nan, np.nan, np.nan, np.nan, -5]))
        B = ingest(raw); return B["dq"]["dups"] == 1 and B["dq"]["bad_ts"] == 1 and B["E"]["value"].notna().sum() == 0, "duplicates removed %d, bad timestamps %d, negative value set to missing" % (B["dq"]["dups"], B["dq"]["bad_ts"])
    def t_rev():
        M = reach(U0, st, False, 0); g = ct_event(U0, M, av, None); return abs(g["net_rev"].sum() - 150) < 1e-9 and g["wins"].sum() == 2, "net revenue %.0f, wins %d" % (g["net_rev"].sum(), g["wins"].sum())
    def t_missing():
        E1 = E0.copy(); E1["value"] = np.nan; U1 = build_users(E1); g = metrics(ct_event(U1, reach(U1, st, False, 0), av, None), .7); return pd.isna(g["net_rev"].sum(min_count=1)) and g["roas"].isna().all() and g["cac"].isna().all(), "missing revenue/cost stays missing; ROAS and CAC unavailable"
    def t_csv():
        f = funnel_df(st, {}, [4, 3, 2, 1], [np.nan] * 4, [np.nan] * 4, 1, 100.0, 100.0); f.to_csv("_t.csv", index=False); g = pd.read_csv("_t.csv"); os.remove("_t.csv"); return g["count"].tolist() == f["count"].tolist(), "exported funnel CSV matches in-memory counts"
    def t_nodbl():
        M = reach(U0, st, True, 0); c = [int(M[s].sum()) for s in st]; return all(c[i] >= c[i + 1] for i in range(3)) and c[0] <= len(U0), "closed funnel is monotone and counts each user once"
    def t_wts():
        bad = [(m, n) for m in ["first", "last", "linear", "position", "time_decay"] for n in (1, 2, 3, 5) if abs(sum(wts(n, list(range(n, 0, -1)), m)) - 1) > 1e-9]; return not bad, "credit weights sum to 1 for 5 methods and journey lengths 1-5"
    def t_order():
        J = build_journeys(E0, U0, "channel"); ok = J.loc["a", "path"] == ["Social", "Search"]; cr = {m: attribute(J, m) for m in ["first", "last", "linear", "position", "time_decay"]}
        rec = all(abs(c[0].sum() - 2) < 1e-9 and abs(c[1].sum() - 150) < 1e-9 for c in cr.values()); return ok and rec, "journey ordered %s; credits reconcile to 2 conversions and 150 revenue" % J.loc["a", "path"]
    def t_markov():
        r = np.random.default_rng(1); rows = []
        for _ in range(400):
            p = list(r.choice(["A", "B", "C"], int(r.integers(1, 4)))); rows.append((p, bool(r.random() < (.5 if "A" in p else .1))))
        J = pd.DataFrame(rows, columns=["path", "conv"]); sh, re_ = markov(J); return abs(sh.sum() - J["conv"].sum()) < 1e-6 and all(0 <= v <= 1 for v in re_.values()), "Markov credits sum to %d conversions; removal effects in [0,1]" % int(J["conv"].sum())
    tot = dict(leads=1000.0, mql=300.0, sql=150.0, opps=60.0, wins=30.0, net_rev=300000.0)
    def t_s0(): s = scen_core(tot, {}, 0); return abs(s["c1"][-1] - 30) < 1e-9 and abs(s["rev1"] - 300000) < 1e-6, "zero change reproduces baseline"
    def t_s1(): s = scen_core(tot, dict(mql=5), 0); return abs(s["c1"][-1] - 35) < 1e-9 and abs(s["rev1"] - 350000) < 1e-6, "+5pp qualification → 35 wins, $350,000 (propagated)"
    def t_s2(): s = scen_core(tot, dict(won=80), 0); return s["c1"][-1] <= s["c1"][-2] + 1e-9, "rates clipped at 100%: wins never exceed opportunities"
    def t_leak(): rows = leak_rows(tot, dict(mql=5), 0, .7); return len(rows) == 4 and all(r["vs"] >= 0 for r in rows) and abs(rows[0]["wins"] - 1000 * .05 * (30 / 300)) < 1e-9, "value at stake ≥ 0; modeled wins use downstream win rate"
    def t_nofin(): s = scen_core(dict(tot, net_rev=np.nan), dict(mql=5), 0); return s is not None and pd.isna(s["rev1"]) and s["c1"][-1] > 30, "missing revenue: counts modeled, revenue not invented"
    for g, n, f in [("funnel", "stage counts and conversion/drop-off", t_counts), ("funnel", "closed vs open funnel", t_closed), ("funnel", "time window", t_window), ("funnel", "duplicates and invalid values", t_dups), ("funnel", "revenue aggregation", t_rev),
                    ("funnel", "missing financial inputs", t_missing), ("funnel", "CSV export consistency", t_csv), ("funnel", "no double counting", t_nodbl), ("attribution", "credit weights", t_wts), ("attribution", "journey ordering and reconciliation", t_order),
                    ("attribution", "Markov removal effect", t_markov), ("scenarios", "baseline reproduced", t_s0), ("scenarios", "propagated improvement", t_s1), ("scenarios", "rate clipping", t_s2), ("scenarios", "leakage formulas", t_leak), ("scenarios", "missing revenue", t_nofin)]: chk(g, n, f)
    if verbose:
        for g, n, ok, d in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + d)
    return T

METH = """### Methodology
- **Funnel:** Exposure → Visit → Lead → MQL → SQL → Opportunity → Closed-won → revenue. Event data counts unique users using each user's first event per stage. **Closed** = sequential within the window; **open** = any occurrence. Aggregated data uses supplied counts only.
- **Drop-off** shows where progression stops being *observed*, not why. Recent cohorts inside the sales-cycle window are flagged or excluded so unfinished journeys are not treated as lost.
- **Value at stake** = drop × (wins ÷ next-stage count) × average deal value (upper bound). **Modeled opportunity** = stage volume × assumed pp improvement × downstream win rate × average deal value. Not additive across rows; the combined scenario propagates once.
- **Metrics:** CTR, CPC, CPL, qualification rates, CAC = spend ÷ wins, ROAS = net revenue ÷ spend, ROI = (net revenue × assumed margin − spend) ÷ spend. Missing inputs give “—”, never zero.
- **Attribution:** first, last, linear, position (40/20/40), time-decay (half-life), and Markov removal effect (≥200 journeys, ≥30 conversions). Credit reconciles to total conversions and revenue. It is an allocation rule, **not causal impact**.
- **Lead model (optional):** features known at lead creation only; matured leads; stratified split; test-set ROC-AUC/PR-AUC/top-20% capture. Predictive, not causal.
- **Evidence levels:** observed counts · calculated metrics · model predictions · attribution estimates · scenario assumptions · causal conclusions (need controlled experiments such as holdouts or geo tests).
- **Privacy:** pseudonymous IDs only; keys stay in the session; only aggregated tool results go to an optional LLM.
"""
REQS = "gradio>=4.44,<6\nplotly\nscikit-learn>=1.2\nrequests\npandas\nnumpy\n"
LICENSE = "MIT License\n\nCopyright (c) 2026 JOURNEY / AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# JOURNEY / AI
Find the leaks. Fix the journey. Grow revenue.

## Overview
An interactive analytics app that traces the customer journey from campaign exposure through leads, qualification, opportunities, conversion and revenue, and shows where the funnel loses value. Bundled data is **fictional synthetic data**.

## Business problem and objectives
Management sees high lead volume but weak revenue growth. The app answers: which channels give leads and paying customers, where the biggest drop-off is, which campaigns attract low-quality leads, how much modeled revenue is associated with leakage, what to fix first, and how to measure improvement.

## Key features
Configurable funnel (closed/open, windows, stage rename/reorder/disable), channel and campaign intelligence, lead quality and optional model, journey and attribution (first, last, linear, position, time-decay, Markov), revenue leakage detector, AI strategist with deterministic fallback, scenario simulator, data quality page, HTML report and CSV exports, built-in tests.

## Screenshots
Add your own screenshots to `screenshots/` (see `screenshots/README.md`). None are included.

## Architecture
@@@mermaid
flowchart LR
  A[CSV upload / synthetic data] --> B[Schema mapping + validation]
  B --> C{Event-level?}
  C -->|yes| D[User table + journeys]
  C -->|no| E[Aggregated stage counts]
  D --> F[Funnel engine]
  E --> F
  D --> G[Attribution + lead quality]
  F --> H[Channel / campaign metrics]
  F --> I[Leakage detector + scenarios]
  H --> J[AI strategist tools]
  I --> J
  G --> J
  J --> K[Gradio UI / reports / exports]
@@@

## Technology stack
Python, Gradio, Plotly, pandas, NumPy, scikit-learn.

## Installation and execution
@@@
pip install -r requirements.txt
python app.py
@@@
In Google Colab, paste `app.py` into one cell and run it.

## Sample dataset schema
`sample_data.csv` (events): timestamp, visitor_id, event_name, channel, campaign, deal_value, refunds, lead_score, industry, company_size. `sample_costs.csv`: campaign, channel, campaign_cost, clicks, impressions. Aggregated files need channel/campaign plus stage counts (visits, leads, mql, sql, opportunities, closed_won, revenue, spend).

## Funnel definitions and formulas
Conversion = stage count / previous stage; cumulative = stage / first stage; drop-off = previous - stage. Closed funnel = sequential within the window; open = any occurrence. Value at stake = drop x downstream win rate x average deal value (upper bound).

## Attribution methodology
First, last, linear, position-based (40/20/40), time-decay (half-life) and Markov removal effect. Credits reconcile to total conversions and revenue. Attribution is not causal impact.

## Scenario calculation methodology
Improve a stage conversion by x percentage points; downstream conditional rates stay at baseline; wins = leads x product of rates; revenue = wins x average deal value x (1 + change); contribution = revenue x assumed margin. Modeled, not a forecast.

## Testing
`pip install pytest && pytest tests` runs the same checks the app shows on start-up.

## Deployment
Run locally, in Colab, or deploy `app.py` to a Hugging Face Space (SDK: Gradio). Publishing is manual and needs your own credentials.

## Data privacy
Use pseudonymous IDs. Never commit real customer data or API keys. Keys entered in the app stay in the session.

## Limitations and future enhancements
Attribution is not causal; scenarios need experiments to validate; aggregated data cannot show journeys. Future: incrementality tests, CRM connectors, time-to-convert curves.
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] == '%s']\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "JOURNEY / AI — Customer Funnel" in s and "def main" in s: return s
    except Exception: pass
    return None

def norm_E(): return ingest(SAMPLE_E)["E"]
def agg_sample():
    U = build_users(norm_E()); M = reach(U, CHAIN, False, 0); D = U[["channel", "campaign"]].copy(); D["date"] = U["t_first"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    for s in CHAIN: D[ACOL[s]] = M[s].astype(float)
    D["revenue"] = (U["value"]).where(M["won"]); D["refunds"] = U["refunds"].where(M["won"]); g = D.groupby(["date", "channel", "campaign"], as_index=False).sum(min_count=1)
    for c in ["spend", "impressions", "clicks"]:
        tot = g.groupby("campaign")["visits"].transform("sum"); cost = SAMPLE_C.set_index("campaign")[{"spend": "campaign_cost"}.get(c, c)]; g[c] = (g["visits"] / tot * g["campaign"].map(cost)).round(2)
    return g.rename(columns={"spend": "campaign_cost", "wins": "closed_won", "opps": "opportunities", "mql": "marketing_qualified_leads", "sql": "sales_qualified_leads"})

def build_project():
    root = "journey-ai"
    for d in [root + "/tests", root + "/screenshots"]: os.makedirs(d, exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    SAMPLE_E.to_csv(root + "/sample_data.csv", index=False); SAMPLE_C.to_csv(root + "/sample_costs.csv", index=False)
    for fn, tx in [("requirements.txt", REQS), ("README.md", README), ("LICENSE", LICENSE), (".gitignore", ".env\n*.key\nsecrets*\n__pycache__/\n.ipynb_checkpoints/\n*.zip\njourney_ai_*.csv\njourney_ai_*.html\n"),
                   ("screenshots/README.md", "# Screenshots\nAdd your own screenshots of the running app here (e.g. overview.png) and reference them from the main README. No screenshots are included.\n"),
                   ("tests/test_funnel.py", TFILE % ("funnel", "funnel")), ("tests/test_attribution.py", TFILE % ("attribution", "attribution")), ("tests/test_scenarios.py", TFILE % ("scenarios", "scenarios"))]:
        open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "journey_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

def make_report():
    V = G.get("V")
    if not V: return None, note("Run the analysis first.", "info")
    B = G["B"]; h = lambda t: "<h2>" + t + "</h2>"; p = []
    p.append("<h1>JOURNEY / AI — Executive Report</h1><p class='m'>Find the leaks. Fix the journey. Grow revenue.</p>" + ("<p class='w'>Illustrative synthetic data: fictional; not real company results.</p>" if B.get("demo") else ""))
    p.append(h("Executive summary") + "<p>%s</p><p>%s</p><p>%s</p>" % (esc(t_summary()), esc(t_drop()), esc(t_opp())))
    p.append(h("Dataset coverage and quality") + "<ul>" + "".join("<li>%s</li>" % esc(w) for w in B["W"] + V["W"]) + "</ul>")
    p.append(h("Funnel-stage analysis") + tbl(V["fun"], [("stage", "Stage"), ("count", "Records"), ("conv", "Conv. from previous"), ("cum", "Cumulative"), ("drop", "Drop-off"), ("vs", "Value at stake (upper bound)")], dict(count=n0, conv=pc1, cum=pc1, drop=n0, vs=money)))
    c = V["CTm"].sort_values("net_rev", ascending=False, na_position="last")
    p.append(h("Campaign and channel comparison") + tbl(c, [("channel", "Channel"), ("campaign", "Campaign"), ("leads", "Leads"), ("l2c", "Lead→customer"), ("net_rev", "Revenue"), ("rpl", "Rev/lead"), ("roas", "ROAS"), ("tag", "Tag")], dict(leads=n0, l2c=pc1, net_rev=money, rpl=money, roas=x2)))
    p.append(h("Lead-quality findings") + "<p>%s</p>" % esc(t_leadq()))
    if V["mode"] == "event": p.append(h("Journey and attribution findings") + "<p>%s</p><p>%s</p>" % (esc(t_paths()), esc(t_attr())))
    rows = leak_rows(V["tot"], dict(mql=5, sql=5, opp=5, won=5), 0, V["margin"])
    if rows: p.append(h("Potential revenue leakage (modeled, +5pp per transition, not additive)") + tbl(pd.DataFrame(rows), [("point", "Leak point"), ("vol", "Records lost"), ("vs", "Value at stake"), ("rev", "Modeled opportunity")], dict(vol=n0, vs=money, rev=money)))
    p.append(h("Recommended actions") + "".join(rec_html(x) for x in recs()))
    p.append(h("Limitations and next steps") + "<ul><li>Drop-off shows where progression stops being observed, not why.</li><li>Attribution credit and scenarios are modeled, not causal or forecasts.</li><li>Validate improvements with controlled experiments (holdouts, A/B or geo tests) and monitor the KPIs listed above.</li></ul>")
    f = go.Figure(go.Funnel(y=V["fun"]["stage"], x=V["fun"]["count"])); f.update_layout(height=360, title="Customer funnel"); p.append(f.to_html(full_html=False, include_plotlyjs="cdn"))
    css = "body{font-family:Inter,Arial,sans-serif;background:#111522;color:#F5F3FA;max-width:1000px;margin:30px auto;padding:0 18px}h1{color:#B9A5FF}h2{color:#E3A5F0;border-left:4px solid #9879FF;padding-left:10px;margin-top:28px}.m{color:#A9ADC4}.w{background:#F2C48D22;border-left:4px solid #F2C48D;padding:8px 12px}table{border-collapse:collapse;width:100%;background:#1B1E2E;margin:8px 0}th{background:#242840;color:#B9A5FF;text-align:left;padding:8px}td{padding:7px 8px;border-top:1px solid #ffffff14}.tw{overflow-x:auto}"
    path = "journey_ai_report.html"; open(path, "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>JOURNEY / AI Report</title><style>" + css + "</style></head><body>" + "".join(p) + "</body></html>")
    return path, note("Report generated (contains aggregated figures only).", "ok")

def export_csvs():
    V = G.get("V")
    if not V: return None, note("Run the analysis first.", "info")
    if V["mode"] == "event" and "paths" not in G: render_attr("Channel", 7)
    if "scn_out" not in G: render_scn("All data", 5, 0, 0, 0, 0)
    out = []
    for nm, d in [("funnel_metrics", V["fun"]), ("campaign_summary", V["CTm"]), ("journey_paths", G.get("paths")), ("scenario_output", G.get("scn_out"))]:
        if d is not None and len(d): p = "journey_ai_%s.csv" % nm; d.to_csv(p, index=False); out.append(p)
    return out, note("Exported %d CSV files." % len(out), "ok")

def load_data(file, cfile, mapping):
    G.clear()
    try:
        demo = file is None
        B = ingest(SAMPLE_E, "", SAMPLE_C) if demo else ingest(read_csv_any(file), mapping, read_csv_any(cfile) if cfile else None); B["demo"] = demo; G["B"] = B
        return note(("Fictional demo data loaded. " if demo else "Your data loaded. ") + "Mode: <b>%s</b>. Computing analytics…" % ("event-level journeys" if B["mode"] == "event" else "aggregated stage counts"), "ok")
    except Exception as e:
        return note("❌ " + esc(e), "err")

def refresh():
    if "B" not in G: return tuple(gr.update() for _ in range(4))
    B = G["B"]; d = B["U"] if B["mode"] == "event" else B["A"]; chs = sorted(d["channel"].dropna().unique().tolist()); cps = sorted(d["campaign"].dropna().unique().tolist()); sg = sorted(d["segment"].dropna().unique().tolist()) if B["mode"] == "event" else []
    return gr.update(choices=chs, value=[]), gr.update(choices=cps, value=[]), gr.update(choices=sg, value=[]), gr.update(choices=["All data"] + ["Channel: " + c for c in chs] + ["Campaign: " + c for c in cps], value="All data")

CSS = """
.gradio-container{background:#111522!important;color:#F5F3FA!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1320px!important}
.hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;padding:26px 30px;border-radius:22px;background:linear-gradient(135deg,#1B1E2E,#2b2358 70%,#9879FF55);border:1px solid #9879FF44;margin-bottom:12px;box-shadow:0 10px 30px #0006}
.brand{display:flex;gap:16px;align-items:center}.logo{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#9879FF,#E3A5F0);color:#111522;font-weight:800;font-size:26px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:34px;letter-spacing:2px;color:#F5F3FA}.hdr h1 span{color:#B9A5FF}.hdr p{margin:4px 0 0;color:#E3A5F0;font-size:16px}
.badges span{display:inline-block;margin:3px;padding:5px 12px;border-radius:99px;background:#ffffff12;border:1px solid #ffffff1f;color:#B9A5FF;font-size:12px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:10px 0}
.kc{background:linear-gradient(180deg,#232640,#1B1E2E);border:1px solid #ffffff14;border-radius:16px;padding:16px;box-shadow:0 6px 18px #0004;animation:fi .5s ease both}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.ki{font-size:22px}.kl{color:#A9ADC4;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-top:4px}.kv{font-size:25px;font-weight:700;color:#F5F3FA;margin:4px 0}.kd{color:#A9ADC4;font-size:12px}
.note{background:#1B1E2E;padding:10px 14px;border-radius:10px;margin:8px 0;color:#F5F3FA;font-size:14px;line-height:1.5}
.sec{font-size:18px;font-weight:700;color:#E3A5F0;margin:18px 0 6px;padding-left:10px;border-left:4px solid #9879FF}
.tw{overflow-x:auto;border-radius:14px;border:1px solid #ffffff14;margin:8px 0}table{border-collapse:collapse;width:100%;background:#1B1E2E;font-size:13px}
th{background:#242840;color:#B9A5FF;text-align:left;padding:10px;white-space:nowrap}td{padding:9px 10px;border-top:1px solid #ffffff10;color:#F5F3FA}tr:hover td{background:#9879FF14}
.pill{padding:3px 10px;border-radius:99px;font-weight:600;white-space:nowrap;font-size:12px}
button.primary{background:linear-gradient(90deg,#9879FF,#E3A5F0)!important;color:#111522!important;border:none!important;font-weight:700!important}
@media(max-width:640px){.hdr h1{font-size:24px}.kv{font-size:20px}.hdr{padding:18px}}
"""
HEADER = ("<div class='hdr'><div class='brand'><div class='logo'>J/</div><div><h1>JOURNEY <span>/ AI</span></h1><p>Find the leaks. Fix the journey. Grow revenue.</p></div></div>"
          "<div class='badges'><span>Drop-off ≠ cause</span><span>Attribution ≠ causal impact</span><span>Scenarios are modeled</span><span>Core analytics need no API key</span></div></div>")

def build_ui(zip_path, files):
    with gr.Blocks(css=CSS, title="JOURNEY / AI", theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("📂 Data (demo data is FICTIONAL) and schema mapping", open=True):
            with gr.Row():
                up = gr.File(label="Upload event-level or aggregated CSV (leave empty for the demo)", file_types=[".csv"]); upc = gr.File(label="Optional cost CSV (campaign, campaign_cost, impressions, clicks)", file_types=[".csv"])
                with gr.Column():
                    mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="uid=lead_id, event=stage_name, channel=utm_source, leads=form_fills", lines=2)
                    ld_btn = gr.Button("📥 Load data", variant="primary")
            gr.File(value=files, label="⬇ Sample files: events, costs, aggregated", file_count="multiple", interactive=False); status = gr.HTML(note("Loading the fictional demo data…", "info"))
        with gr.Accordion("🎛️ Filters and funnel settings (apply to every tab)", open=False):
            with gr.Row():
                f_ch = gr.Dropdown([], multiselect=True, label="Channel"); f_cp = gr.Dropdown([], multiselect=True, label="Campaign"); f_seg = gr.Dropdown([], multiselect=True, label="Customer segment (industry)")
            with gr.Row():
                f_d1 = gr.Textbox(label="From date (YYYY-MM-DD)"); f_d2 = gr.Textbox(label="To date (YYYY-MM-DD)"); f_type = gr.Radio(["Closed funnel (sequential, within window)", "Open funnel (any occurrence)"], value="Closed funnel (sequential, within window)", label="Funnel definition")
            with gr.Row():
                f_win = gr.Slider(0, 365, 120, step=5, label="Conversion window (days; 0 = none)"); f_mat = gr.Checkbox(True, label="Exclude cohorts still inside the window"); f_mar = gr.Slider(.1, .95, .7, step=.05, label="Assumed gross margin (for ROI / contribution)")
            f_stg = gr.CheckboxGroup(choices=[(LBL[s], s) for s in STG], value=STG, label="Enabled stages")
            with gr.Row():
                f_ord = gr.Textbox(label="Stage order (keys, comma-separated)", value=",".join(STG)); f_ren = gr.Textbox(label="Rename stages (key=Label, ...)", placeholder="lead=Form lead, won=Customer")
            ap_btn = gr.Button("⚡ Apply filters and settings", variant="primary"); status2 = gr.HTML()
        with gr.Tabs():
            with gr.Tab("A · Executive Overview"):
                kpi = gr.HTML()
                with gr.Row():
                    x1 = gr.Plot(); x2_ = gr.Plot()
                x3 = gr.Plot(); x_ins = gr.HTML()
            with gr.Tab("B · Customer Funnel Explorer"):
                fn_html = gr.HTML()
                with gr.Row():
                    fn_fig = gr.Plot(); fn_fig2 = gr.Plot()
            with gr.Tab("C · Channel & Campaign Intelligence"):
                ch_html = gr.HTML(); ch_fig = gr.Plot()
            with gr.Tab("D · Lead Quality & Qualification"):
                lq_html = gr.HTML(); lq_fig = gr.Plot()
                with gr.Row():
                    lq_t = gr.Radio(["Becomes MQL", "Becomes customer"], value="Becomes MQL", label="Optional model target"); lq_btn = gr.Button("Train lead model", variant="primary")
                lq_m = gr.HTML()
            with gr.Tab("E · Customer Journey & Attribution"):
                with gr.Row():
                    at_dim = gr.Radio(["Channel", "Campaign"], value="Channel", label="Attribute to"); at_hl = gr.Slider(1, 30, 7, step=1, label="Time-decay half-life (days)"); at_btn = gr.Button("Run attribution", variant="primary")
                at_html = gr.HTML()
                with gr.Row():
                    at_f1 = gr.Plot(); at_f2 = gr.Plot()
            with gr.Tab("F · Revenue Leakage Detector"):
                gr.Markdown("Editable assumptions: improvement in conversion into each stage (percentage points). Outputs are **modeled opportunities**, not actual lost revenue.")
                with gr.Row():
                    lk_mql = gr.Slider(0, 30, 5, step=1, label="Lead → MQL (+pp)"); lk_sql = gr.Slider(0, 30, 5, step=1, label="MQL → SQL (+pp)"); lk_opp = gr.Slider(0, 30, 5, step=1, label="SQL → opportunity (+pp)")
                    lk_won = gr.Slider(0, 30, 5, step=1, label="Opportunity → won (+pp)"); lk_adv = gr.Slider(0, 50, 0, step=1, label="Deal value change (%)")
                lk_btn = gr.Button("Recalculate leakage", variant="primary"); lk_html = gr.HTML(); lk_fig = gr.Plot()
            with gr.Tab("G · AI Funnel Strategist"):
                st_recs = gr.HTML()
                ag_q = gr.Textbox(label="Ask a question", placeholder="Where are we losing the most value? / What if qualification improved by 5 points? / Which campaigns have incomplete tracking?")
                with gr.Row():
                    ag_key = gr.Textbox(label="Anthropic API key (optional, session only, never stored)", type="password"); ag_model = gr.Textbox(label="Model", value="claude-sonnet-5-5")
                ag_btn = gr.Button("Ask strategist", variant="primary"); ag_out = gr.Markdown()
            with gr.Tab("H · Scenario Simulator"):
                sc_scope = gr.Dropdown(["All data"], value="All data", label="Scope (all data, channel or campaign)")
                with gr.Row():
                    sc_mql = gr.Slider(-20, 30, 5, step=1, label="Lead → MQL (pp change)"); sc_sql = gr.Slider(-20, 30, 0, step=1, label="MQL → SQL (pp)"); sc_opp = gr.Slider(-20, 30, 0, step=1, label="SQL → opportunity (pp)")
                    sc_won = gr.Slider(-20, 30, 0, step=1, label="Opportunity → won (pp)"); sc_adv = gr.Slider(-30, 50, 0, step=1, label="Average deal value change (%)")
                with gr.Row():
                    sc_btn = gr.Button("▶ Simulate", variant="primary"); sc_rst = gr.Button("↺ Reset scenario")
                sc_html = gr.HTML()
                with gr.Row():
                    sc_f1 = gr.Plot(); sc_f2 = gr.Plot()
            with gr.Tab("I · Data Quality & Methodology"):
                dq_html = gr.HTML(); tests_html = gr.HTML(); gr.Markdown(METH)
                gr.Markdown("### Export & publish\nDownload the ZIP, unzip it, create an empty GitHub repository and upload the contents (Add file → Upload files). Never upload real customer data or API keys. Deployment (e.g. a Hugging Face Space) is manual and uses your own credentials.")
                gr.File(value=zip_path, label="⬇ Download GitHub-ready project (ZIP)", interactive=False)
                with gr.Row():
                    rep_btn = gr.Button("📄 Generate executive report", variant="primary"); exp_btn = gr.Button("📊 Export CSVs", variant="primary")
                rep_msg = gr.HTML(); rep_file = gr.File(label="⬇ Report (HTML)", interactive=False); exp_msg = gr.HTML(); exp_files = gr.File(label="⬇ CSV exports", file_count="multiple", interactive=False)
        CIN = [f_ch, f_cp, f_seg, f_d1, f_d2, f_stg, f_ord, f_ren, f_type, f_win, f_mat, f_mar]; AT_IN = [at_dim, at_hl]; AT_OUT = [at_html, at_f1, at_f2]; LK_IN = [lk_mql, lk_sql, lk_opp, lk_won, lk_adv]; LK_OUT = [lk_html, lk_fig]
        SC_IN = [sc_scope, sc_mql, sc_sql, sc_opp, sc_won, sc_adv]; SC_OUT = [sc_html, sc_f1, sc_f2]
        def tabs(e):
            e = e.then(render_exec, None, [kpi, x1, x2_, x3, x_ins]).then(render_funnel, None, [fn_html, fn_fig, fn_fig2]).then(render_channels, None, [ch_html, ch_fig]).then(render_leadq, None, [lq_html, lq_fig])
            e = e.then(render_attr, AT_IN, AT_OUT).then(render_leak, LK_IN, LK_OUT).then(render_strat, None, [st_recs]).then(render_scn, SC_IN, SC_OUT)
            return e.then(render_dq, None, [dq_html, tests_html])
        def full(ev): return tabs(ev.then(refresh, None, [f_ch, f_cp, f_seg, sc_scope]).then(do_compute, CIN, [status2]))
        full(ld_btn.click(load_data, [up, upc, mp], [status], show_progress="full")); full(demo.load(load_data, [up, upc, mp], [status])); tabs(ap_btn.click(do_compute, CIN, [status2], show_progress="full"))
        lq_btn.click(run_lead_model, lq_t, lq_m, show_progress="full"); at_btn.click(render_attr, AT_IN, AT_OUT, show_progress="full"); lk_btn.click(render_leak, LK_IN, LK_OUT); sc_btn.click(render_scn, SC_IN, SC_OUT)
        sc_rst.click(lambda: (0, 0, 0, 0, 0), None, [sc_mql, sc_sql, sc_opp, sc_won, sc_adv]).then(render_scn, SC_IN, SC_OUT)
        ag_btn.click(ask, [ag_q, ag_key, ag_model], ag_out, show_progress="full"); ag_q.submit(ask, [ag_q, ag_key, ag_model], ag_out)
        rep_btn.click(make_report, None, [rep_file, rep_msg]); exp_btn.click(export_csvs, None, [exp_files, exp_msg])
    return demo

def main():
    zip_path, saved = build_project(); g = agg_sample(); g.to_csv("journey_ai_sample_aggregated.csv", index=False); SAMPLE_E.to_csv("journey_ai_sample_events.csv", index=False); SAMPLE_C.to_csv("journey_ai_sample_costs.csv", index=False)
    print("=" * 78); print("JOURNEY / AI starting. Demo data is FICTIONAL. Core analytics need no API key."); print("Running self-tests:"); TESTS[:] = run_selftests(verbose=True)
    demo = build_ui(zip_path, ["journey_ai_sample_events.csv", "journey_ai_sample_costs.csv", "journey_ai_sample_aggregated.csv"])
    print("Project ZIP:", zip_path, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("The demo loads automatically; the tabs fill in after a few seconds. Keys: use Colab Secrets (ANTHROPIC_API_KEY). The public Gradio link is open to anyone: don't share it with real data loaded.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)

main()