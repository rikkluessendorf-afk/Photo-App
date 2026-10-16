"""Photo Critic: Streamlit front end (deploys free on Streamlit Community Cloud)."""

import hashlib
import hmac
import os
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Photo Critic", page_icon="📷", layout="centered")


def secret(name, default=None):
    """Read from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# Copy settings into the environment so critic_core can read them.
for _k in (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "TEMPERATURE",
    "MAX_BATCH",
    "MIN_SECONDS_BETWEEN_CALLS",
    "MAX_IMAGE_EDGE",
    "MAX_ANCHORS",
):
    _v = secret(_k)
    if _v is not None:
        os.environ[_k] = _v

import critic_core as core  # noqa: E402


# --------------------------------------------------------------------------
# Password gate
# --------------------------------------------------------------------------
@st.cache_resource
def _failed_logins():
    return []  # timestamps of recent failed sign-ins, shared across sessions


def require_login():
    password = secret("APP_PASSWORD")
    if not password:
        st.title("📷 Photo Critic")
        st.error(
            "APP_PASSWORD isn't set, so the app won't open. Add it in the app's "
            "Settings → Secrets."
        )
        st.stop()
    if st.session_state.get("authed"):
        return

    st.title("📷 Photo Critic")
    fails = _failed_logins()
    now = time.time()
    fails[:] = [t for t in fails if now - t < 600]
    locked = len(fails) >= 8

    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", disabled=locked)

    if locked:
        st.error("Too many failed attempts. Please wait a few minutes and try again.")
    elif submitted:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["authed"] = True
            st.rerun()
        else:
            fails.append(now)
            time.sleep(1.5)
            st.error("That password isn't right.")
    st.stop()


require_login()

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("results", {})  # photo id -> result dict (insertion order = upload order)
ss.setdefault("chats", {})  # photo id -> list of {"role","content"}
ss.setdefault("uploader_key", 0)


def photo_id(uploaded) -> str:
    return hashlib.sha1(uploaded.getvalue()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Header + upload
# --------------------------------------------------------------------------
st.title("📷 Photo Critic")
st.markdown(
    "Upload one or many photos. Each is judged fresh against the same rubric: six categories scored 1-100, "
    "with **Overall Execution counting for 50%** of the composite.\n\n"
    "**Tiers:** S = 85+ · A = 75-84.9 · B = 65-74.9 · Field = below 65"
)

files = st.file_uploader(
    "Photos: tap to choose from your library or take a new one",
    type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
    accept_multiple_files=True,
    key=f"uploader_{ss['uploader_key']}",
)
with st.expander("Or take a photo with the in-page camera"):
    st.caption(
        "Handy if choosing from your library keeps resetting the page, because it never leaves "
        "this screen. Quality is lower than your phone's own camera app."
    )
    shot = st.camera_input("Camera", key=f"camera_{ss['uploader_key']}", label_visibility="collapsed")

all_files = list(files or []) + ([shot] if shot else [])
limit = core.max_batch()
st.caption(f"Up to {limit} new photos per run. Keep this screen open while it works.")

go = st.button("Critique my photos", type="primary", disabled=not all_files)

if go and all_files:
    new, dupes = [], 0
    for f in all_files:
        pid = photo_id(f)
        if pid in ss["results"]:
            dupes += 1
        else:
            new.append((pid, f))
    over = max(0, len(new) - limit)
    new = new[:limit]

    problems = []
    if new:
        bar = st.progress(0.0, text="Starting...")
        note = st.empty()
        for i, (pid, f) in enumerate(new):
            bar.progress(
                i / len(new),
                text=f"Critiquing {f.name} ({i + 1} of {len(new)}). Usually 10-40 seconds.",
            )
            note.empty()
            try:
                ss["results"][pid] = core.critique_upload(
                    f.name, f.getvalue(), on_status=lambda m: note.info(m)
                )
            except core.DailyQuotaError as e:
                problems.append(str(e))
                break
            except Exception as e:  # noqa: BLE001
                problems.append(f"{f.name}: {str(e)[:200]}")
        bar.empty()
        note.empty()
    if dupes:
        st.info(f"{dupes} photo(s) were already critiqued this session and were skipped.")
    if over:
        st.info(f"{over} photo(s) were left for the next run (batch limit is {limit}).")
    for p in problems:
        st.warning(p)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
results = ss["results"]
if results:
    order = sorted(results, key=lambda k: results[k]["composite"], reverse=True)

    st.subheader("Results")
    df = pd.DataFrame(
        [
            {
                "Photo": results[k]["name"],
                "Score": results[k]["composite"],
                "Tier": results[k]["tier_short"],
                **{short: results[k]["cats"][key]["score"] for key, _, short, _ in core.CATEGORIES},
                "Secs": results[k].get("seconds"),
            }
            for k in order
        ]
    )
    st.dataframe(df, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download results (CSV)",
        data=core.csv_text(list(results.values())),
        file_name="photo_critiques.csv",
        mime="text/csv",
        width="stretch",
    )
    if c2.button("Clear all results", width="stretch"):
        ss["results"] = {}
        ss["chats"] = {}
        ss["uploader_key"] += 1
        st.rerun()

    st.subheader("Look closer")
    pid = st.selectbox(
        "Choose a photo",
        order,
        format_func=lambda k: f"{results[k]['tier_short']} {results[k]['composite']:.1f} · {results[k]['name']}",
    )
    r = results[pid]

    st.image(r["jpeg"], width="stretch")
    m1, m2 = st.columns(2)
    m1.metric("Composite", f"{r['composite']:.1f}")
    m2.metric("Tier", r["tier_label"])
    meta = f"{r['photo_type']} · {r.get('model', '?')} · {r.get('seconds', '?')}s"
    if r.get("retries"):
        meta += f" · {r['retries']} retr{'y' if r['retries'] == 1 else 'ies'}"
    st.caption(meta)

    for key, label, _, weight in core.CATEGORIES:
        c = r["cats"][key]
        st.markdown(f"**{label}** · {c['score']} _(weight {int(weight * 100)}%)_")
        st.progress(c["score"] / 100)
        st.caption(c["note"])

    st.markdown("#### What's working")
    st.markdown(r["what_works"])
    st.markdown("#### What isn't")
    st.markdown(r["what_doesnt"])
    st.markdown("#### How to fix it")
    st.markdown(r["repairs"])
    st.markdown(f"> {r['closing']}")

    # ----------------------------------------------------------------------
    # Discussion (per photo)
    # ----------------------------------------------------------------------
    st.subheader("Talk it over with your instructor")
    st.caption("Push back, ask why, ask how to fix it. Discussion doesn't change the saved scores.")
    history = ss["chats"].setdefault(pid, [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("e.g. I framed it tight on purpose. Does that change anything?")
    if prompt:
        prior = list(history)
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = core.discussion_reply(r, prior, prompt)
            st.markdown(reply)
        history.append({"role": "assistant", "content": reply})
