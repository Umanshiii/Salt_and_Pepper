import re
import streamlit as st
from app import CSS, HINT, FOOT, classify, render_result, render_history

st.set_page_config(page_title="Hindi and Hinglish intent classifier", layout="centered")


def md(s):
    st.markdown(re.sub(r"\n\s*", "", s), unsafe_allow_html=True)


md(f"<style>{CSS}</style>")
md('<div class="ic-wrap"><h2 style="margin:6px 0 2px;font-size:22px;font-weight:600;">Hindi and Hinglish intent classifier</h2>'
   '<p class="ic-label" style="font-size:13px;">56 intent classes. Class 55 is fallback for requests the assistant can\'t handle.</p></div>')

if "hist" not in st.session_state:
    st.session_state["hist"] = []
if "text" not in st.session_state:
    st.session_state["text"] = ""

samples = ["speekar ka volyoom bilkul kam karo", "मेरे कैलेंडर से यह घटना हटा दो", "mjhe garm kofee chaahie"]
run_now = False
cols = st.columns(len(samples))
for c, s in zip(cols, samples):
    if c.button(s, key="sample_" + s):
        st.session_state["text"] = s
        run_now = True

with st.form("query_form"):
    st.text_input("Type a Hindi or Hinglish query", key="text", placeholder="mjhe garm kofee chaahie")
    go = st.form_submit_button("Classify")

if go or run_now:
    q = st.session_state["text"]
    if not q.strip():
        md('<div class="ic-wrap"><p class="ic-err">Enter a query first.</p></div>')
    else:
        r = classify(q)
        st.session_state["hist"] = ([(q.strip(), r["predicted_class"])] + st.session_state["hist"])[:5]
        md(render_result(r))
else:
    md(HINT)

md(render_history(st.session_state["hist"]))
md(FOOT)
