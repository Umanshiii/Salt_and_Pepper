import re, time, html, joblib, numpy as np

MODEL_FILE = "svm_model.joblib"
M = joblib.load(MODEL_FILE)
CLASSES = M["svm"].classes_
DEV = re.compile(r"[\u0900-\u097F]")
LAT = re.compile(r"[A-Za-z]")

CSS = """
.gradio-container {max-width: 880px !important; margin: 0 auto !important;}
.ic-wrap {font-family: Inter, 'Noto Sans Devanagari', system-ui, sans-serif; color:#1A1A1A;}
.ic-card {background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:16px 18px;}
.ic-grid {display:grid; grid-template-columns: 1.3fr 1fr; gap:12px; margin-bottom:12px;}
.ic-label {font-size:12px; color:#6B7280; margin:0;}
.ic-big {font-size:34px; font-weight:600; margin:4px 0 8px; color:#1A1A1A;}
.ic-pill {display:inline-block; font-size:12px; padding:3px 10px; border-radius:8px; margin:0 6px 4px 0;}
.ic-green {background:#DCFCE7; color:#15803D;}
.ic-amber {background:#FEF3C7; color:#B45309;}
.ic-red {background:#FEE2E2; color:#B91C1C;}
.ic-gray {background:#F3F4F6; color:#4B5563;}
.ic-fallback {background:#FEF3C7; border:1px solid #F5D78A;}
.ic-fallback .ic-label, .ic-fallback .ic-big, .ic-fallback p {color:#92400E;}
.ic-title {font-size:13px; font-weight:600; margin:0 0 10px;}
.ic-row {display:grid; grid-template-columns: 70px 1fr 52px; gap:10px; align-items:center; font-size:13px; margin-bottom:8px;}
.ic-track {height:8px; background:#F3F4F6; border-radius:4px;}
.ic-bar {height:8px; border-radius:4px; background:#9CA3AF;}
.ic-bar.top {background:#2563EB;}
.ic-bar.top.fb {background:#B45309;}
.ic-hist {display:flex; justify-content:space-between; gap:10px; font-size:13px; padding:6px 0; border-bottom:1px solid #F3F4F6;}
.ic-hist span:first-child {color:#4B5563;}
.ic-foot {font-size:12px; color:#6B7280; margin-top:10px;}
.ic-hint {text-align:center; color:#6B7280; font-size:14px; padding:26px 0;}
.ic-err {color:#B91C1C; font-size:13px; padding:10px 0;}
@media (max-width: 640px) {.ic-grid {grid-template-columns: 1fr;}}
"""


def softmax(z, t=0.2):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def script_of(text):
    d, l = bool(DEV.search(text)), bool(LAT.search(text))
    if d and l:
        return "Mixed"
    return "Devanagari" if d else "Roman (Hinglish)"


def classify(text):
    t0 = time.perf_counter()
    d = M["svm"].decision_function(M["feats"].transform([text.lower().strip()]))
    p = softmax(d)[0]
    top = np.argsort(-p)[:3]
    ms = (time.perf_counter() - t0) * 1000
    return {
        "predicted_class": int(CLASSES[top[0]]),
        "is_fallback": int(CLASSES[top[0]]) == 55,
        "top3": [(int(CLASSES[i]), float(p[i])) for i in top],
        "latency_ms": ms,
        "script": script_of(text),
    }


def pct(x):
    return f"{x*100:.0f}%" if x >= 0.1 else f"{x*100:.1f}%"


def confidence_pill(top3):
    p1, p2 = top3[0][1], top3[1][1]
    if p1 >= 0.70:
        return '<span class="ic-pill ic-green">High confidence</span>', ""
    if p1 >= 0.40:
        return '<span class="ic-pill ic-amber">Medium confidence</span>', ""
    return '<span class="ic-pill ic-red">Low confidence</span>', '<p class="ic-label">The top classes are close.</p>'


def render_result(r):
    fb = r["is_fallback"]
    pill, note = confidence_pill(r["top3"])
    ok = r["latency_ms"] < 50
    lat_line = ('<p class="ic-label" style="color:#15803D">Under the 50 ms target</p>' if ok
                else '<p class="ic-label" style="color:#B91C1C">Over the 50 ms target</p>')
    left_cls = "ic-card ic-fallback" if fb else "ic-card"
    title = f"Class {r['predicted_class']}: fallback" if fb else f"Class {r['predicted_class']}"
    sub = ('<p style="font-size:13px;margin:0 0 6px;">The assistant would say it can\'t do this request.</p>' if fb else "")
    rows = ""
    for i, (c, p) in enumerate(r["top3"]):
        cls = "ic-bar top fb" if (i == 0 and fb) else ("ic-bar top" if i == 0 else "ic-bar")
        rows += (f'<div class="ic-row"><span>Class {c}</span>'
                 f'<div class="ic-track"><div class="{cls}" style="width:{max(p*100,1):.0f}%"></div></div>'
                 f'<span style="text-align:right">{pct(p)}</span></div>')
    return f"""<div class="ic-wrap">
<div class="ic-grid">
  <div class="{left_cls}">
    <p class="ic-label">Predicted intent</p>
    <p class="ic-big">{title}</p>{sub}
    {pill}<span class="ic-pill ic-gray">Script: {r['script']}</span>{note}
  </div>
  <div class="ic-card">
    <p class="ic-label">Model response time</p>
    <p class="ic-big">{r['latency_ms']:.1f} ms</p>
    {lat_line}
  </div>
</div>
<div class="ic-card">
  <p class="ic-title">Top 3 predictions</p>{rows}
  <p class="ic-foot">Scores are relative model scores that add up to 100%. They are not calibrated probabilities.</p>
</div></div>"""


def render_history(hist):
    if not hist:
        return '<div class="ic-wrap"><div class="ic-card"><p class="ic-title">Recent queries</p><p class="ic-label">Nothing yet.</p></div></div>'
    rows = "".join(f'<div class="ic-hist"><span>{html.escape(q)}</span><span>Class {c}</span></div>' for q, c in hist)
    return f'<div class="ic-wrap"><div class="ic-card"><p class="ic-title">Recent queries</p>{rows}</div></div>'


HINT = '<div class="ic-wrap"><div class="ic-card"><p class="ic-hint">Try a sample below or type your own query.</p></div></div>'
FOOT = ('<div class="ic-wrap"><p class="ic-foot">TF-IDF + linear SVM &nbsp;|&nbsp; 16,936 training queries, plus augmented copies'
        ' &nbsp;|&nbsp; Public leaderboard Macro-F1: 0.842</p></div>')


def run(text, hist):
    hist = hist or []
    if not text or not text.strip():
        return '<div class="ic-wrap"><p class="ic-err">Enter a query first.</p></div>', render_history(hist), hist
    try:
        r = classify(text)
    except Exception:
        return '<div class="ic-wrap"><p class="ic-err">Couldn\'t reach the model. Try again.</p></div>', render_history(hist), hist
    hist = ([(text.strip(), r["predicted_class"])] + hist)[:5]
    return render_result(r), render_history(hist), hist


classify("warm up")  # first call is slower, so run one before the demo starts


def build_ui():
    import gradio as gr
    with gr.Blocks(css=CSS, title="Hindi and Hinglish intent classifier") as demo:
        gr.HTML('<div class="ic-wrap"><h2 style="margin:6px 0 2px;font-size:22px;font-weight:600;">Hindi and Hinglish intent classifier</h2>'
                '<p class="ic-label" style="font-size:13px;">56 intent classes. Class 55 is fallback for requests the assistant can\'t handle.</p></div>')
        state = gr.State([])
        with gr.Row():
            box = gr.Textbox(label="Type a Hindi or Hinglish query", placeholder="mjhe garm kofee chaahie", scale=5, lines=1)
            btn = gr.Button("Classify", variant="primary", scale=1)
        out = gr.HTML(HINT)
        gr.Examples(
            examples=[["speekar ka volyoom bilkul kam karo"], ["मेरे कैलेंडर से यह घटना हटा दो"], ["mjhe garm kofee chaahie"]],
            inputs=box, label="Samples")
        hist_out = gr.HTML(render_history([]))
        gr.HTML(FOOT)
        btn.click(run, [box, state], [out, hist_out, state])
        box.submit(run, [box, state], [out, hist_out, state])
    return demo


if __name__ == "__main__":
    build_ui().launch()
