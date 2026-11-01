import json
import time
from string import Template

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
st.set_page_config(page_title="Student Support Bot", page_icon="🎓")

MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]
PROMPT_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6"]
KB_FILES = {"DUET": "data/faq_duet.json", "Sample": "data/faq.json"}

PALETTES = {
    False: dict(
        bg1="#f6f0ff", bg2="#ffffff", text="#2d2347", muted="#7d729c",
        accent="#7c5cd6", accent2="#b79cf5", card="#ffffff",
        border="#e2d8fa", user="#ece3ff", shadow="rgba(124, 92, 214, 0.14)",
    ),
    True: dict(
        bg1="#0a0712", bg2="#1b1033", text="#efe9ff", muted="#a99bd1",
        accent="#a78bfa", accent2="#7c3aed", card="#171026",
        border="#3a2a63", user="#2b1d4f", shadow="rgba(0, 0, 0, 0.45)",
    ),
}

CSS = Template(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, $bg1 0%, $bg2 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 780px; padding-top: 2.5rem; }

html, .stApp p, .stApp h1, .stApp label, .stApp textarea, .stApp button {
    font-family: 'Poppins', sans-serif !important;
}
.stApp p, .stApp label, .stApp li, .stApp span { color: $text; }

.stApp h1 {
    font-weight: 600;
    background: linear-gradient(90deg, $accent, $accent2);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stCaptionContainer"] * { color: $muted !important; }

[data-testid="stAlert"] {
    background: $card !important;
    border: 1px solid $border;
    border-radius: 14px;
}
[data-testid="stAlert"] * { color: $text !important; }

[data-testid="stChatMessage"] {
    background: $card;
    border: 1px solid $border;
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 4px 18px $shadow;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: $user;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, $accent, $accent2) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, $accent2, $accent) !important;
}

[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] {
    background: $card !important;
    border: 1px solid $border !important;
    border-radius: 28px !important;
    box-shadow: 0 6px 24px $shadow;
}
[data-testid="stChatInput"] * { background-color: transparent !important; }
[data-testid="stChatInput"] textarea {
    color: $text !important;
    -webkit-text-fill-color: $text !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: $muted !important;
    -webkit-text-fill-color: $muted !important;
}
[data-testid="stChatInputSubmitButton"] {
    background-color: $accent !important;
    border-radius: 50% !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

[data-testid="stRadio"] label p { font-size: 0.8rem; }
[data-testid="stCaptionContainer"] { opacity: 1 !important; }
[data-testid="stCaptionContainer"] * { color: $text !important; opacity: 0.8; }

[data-testid="stAlert"], [data-testid="stAlertContainer"] {
    background: $card !important;
    border: 1px solid $border !important;
    border-radius: 14px !important;
}

[data-baseweb="radio"] > div:first-child {
    background-color: $card !important;
    border-color: $accent !important;
}

[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
[data-testid="stToggle"] [data-testid="stMarkdownContainer"],
[data-testid="stToggle"] [data-testid="stMarkdownContainer"] p {
    background: transparent !important;
}
</style>
"""
)


@st.cache_resource
def get_client():
    return genai.Client()


client = get_client()


class Reply(BaseModel):
    answer: str
    in_faq: bool
    language: str


# Header with the light / dark switch
head_l, head_r = st.columns([5, 1])
head_l.title("Student Support Bot")
dark = head_r.toggle("Dark", key="dark_mode")
st.markdown(CSS.substitute(PALETTES[dark]), unsafe_allow_html=True)

# Controls sit just above the chat box
bottom = getattr(st, "_bottom", None)
controls = bottom if bottom is not None else st.container()
with controls:
    c1, c2 = st.columns([3, 2])
    version = c1.radio("Prompt version", PROMPT_VERSIONS, index=5, horizontal=True)
    kb = c2.radio("Knowledge base", list(KB_FILES), index=0, horizontal=True)

FAQ = open(KB_FILES[kb], encoding="utf-8").read()
SYSTEM = open(f"prompts/{version}.txt", encoding="utf-8").read()


def ask(history):
    use_json = version in ("v4", "v5", "v6")
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM + "\n\nFAQ:\n" + FAQ,
        temperature=0,
        response_mime_type="application/json" if use_json else None,
        response_schema=Reply if use_json else None,
    )
    for model in MODELS:
        for attempt in range(2):
            start = time.time()
            try:
                r = client.models.generate_content(
                    model=model, contents=history, config=config
                )
                print("MODEL:", model, "| API time:", round(time.time() - start, 1), "sec")
                if use_json:
                    data = json.loads(r.text)
                    return data["answer"], data["in_faq"]
                return r.text, None
            except Exception as e:
                print("ERROR:", str(e)[:120])
                time.sleep(2)
    return "Sorry, the AI service is busy right now. Please try again in a minute.", None


def show(m):
    box = st.chat_message(m["role"])
    box.write(m["content"])
    if m.get("in_faq") is False:
        box.caption("Not found in FAQ")


if kb == "DUET":
    st.info(
        "Answers come from public DUET web pages (some may be dated). Please confirm on the DUET Portal or with the admin office."
    )
else:
    st.warning("Demo with sample data. This is not official university information.")
st.caption(
    "I answer general questions about fees, exams, transcripts and attendance from the university FAQ. I cannot see your personal account data. (Main aap ka personal account data nahi dekh sakta.)"
)

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    show(m)

if q := st.chat_input("Ask about fees, exams, attendance..."):
    user_msg = {"role": "user", "content": q}
    st.session_state.msgs.append(user_msg)
    show(user_msg)

    history = "\n".join(f'{m["role"]}: {m["content"]}' for m in st.session_state.msgs)
    with st.spinner("Thinking..."):
        t0 = time.time()
        answer, in_faq = ask(history)
        print("TOTAL time:", round(time.time() - t0, 1), "sec")

    bot_msg = {"role": "assistant", "content": answer, "in_faq": in_faq}
    st.session_state.msgs.append(bot_msg)
    show(bot_msg)