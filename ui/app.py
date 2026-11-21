import os
import re
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None! ")

    logfire.configure(token=token, send_to_logfire="if-token-present")

    # logfire.instrument_requests()   # Disabled due to OpenTelemetry bug on windows
    LOGFIRE_STATUS = "Connected & Tracing"

except Exception as e:
    print(f" Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"


# --- Page CONFIG --
st.set_page_config(page_title="Agentic RAG", page_icon="", layout="wide")

# -- Avatars --
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- Session Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f" New User Session Created: {st.session_state}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SideBar ---
with st.sidebar:
    st.title(" Agent OS")
    st.markdown("----")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(
            f"Memory Wipe Triggered for session: {st.session_state.session_id}"
        )
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# --- MAIN CHAT ---
st.title(" Agentic Assistant")


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Render metadata badges and sources for assistant messages
        if message["role"] == "assistant":
            meta = message.get("meta", {})
            if meta.get("security") == "BLOCKED":
                st.error(" Security Guardrails Triggered! Prompt Blocked.")
            elif meta.get("cache") == "HIT":
                st.caption(
                    f" **Instant Cache Hit** • {meta.get('duration', 0)*1000:.1f}ms • 0 LLM Tokens"
                )
            elif meta.get("duration"):
                st.caption(
                    f" **Hybrid RAG Retrieval** • {meta.get('duration', 0):.2f}s • Dense + BM25 RRF"
                )
            sources = message.get("sources", [])
            if sources:
                with st.expander(" Verified Sources", expanded=False):
                    for src in sources:
                        st.markdown(f"- 📄 `{src}`")

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span(
        " User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response with Real-Time SSE Token Streaming & Logfire Spans
        with st.chat_message("assistant", avatar=AI_AVATAR):
            base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            url = f"{base_url}/query/stream"
            payload = {
                "q": prompt,
                "thread_id": st.session_state.session_id,
            }

            # Telemetry metadata captured from backend headers and timing
            meta = {
                "cache": "MISS",
                "security": "PASSED",
                "duration": 0.0,
            }

            def stream_response_with_telemetry():
                # START LOGFIRE SPAN for the streaming lifecycle
                with logfire.span(
                    "Real-Time Stream Consumption",
                    session_id=st.session_state.session_id,
                    endpoint="/query/stream",
                ):
                    start_time = time.perf_counter()
                    first_token_received = False
                    total_chunks = 0
                    total_chars = 0

                    try:
                        with requests.post(
                            url, json=payload, stream=True, timeout=60
                        ) as resp:
                            # 1. Read custom headers for telemetry
                            meta["cache"] = resp.headers.get("X-Cache", "MISS")
                            meta["security"] = resp.headers.get("X-Security", "PASSED")

                            if resp.status_code != 200:
                                err_msg = f"Backend returned HTTP {resp.status_code}"
                                logfire.error(f"Streaming HTTP Error: {err_msg}")
                                yield f" **Error:** {err_msg}"
                                return

                            # Read stream chunk by chunk as generated
                            for chunk in resp.iter_content(
                                chunk_size=None, decode_unicode=True
                            ):
                                if chunk:
                                    if not first_token_received:
                                        first_token_received = True
                                        ttft = time.perf_counter() - start_time
                                        # Record Time-To-First-Token in Logfire telemetry!
                                        logfire.info(
                                            f" Time-to-First-Token (TTFT): {ttft:.3f}s",
                                            ttft_seconds=ttft,
                                        )

                                    total_chunks += 1
                                    total_chars += len(chunk)
                                    yield chunk

                        # Log stream summary upon successful completion (Record final duration)
                        meta["duration"] = time.perf_counter() - start_time
                        logfire.info(
                            "Stream completed successfully",
                            total_duration_sec=meta["duration"],
                            total_chunks=total_chunks,
                            total_characters=total_chars,
                        )

                    except Exception as e:
                        logfire.error(f"Stream interrupted or failed: {e}")
                        yield f"\n\n **Streaming Connection Interrupted:** {e}"

            # 1. Stream answer live to the screen
            full_answer = st.write_stream(stream_response_with_telemetry())

            # 2. Render Security Alert or Speed Badge
            if meta["security"] == "BLOCKED":
                st.error(
                    " **Security Alert:** Request blocked by enterprise guardrails (Adversarial Prompt Injection detected)."
                )
            elif meta["cache"] == "HIT":
                st.caption(
                    f" **Instant Cache Hit** • {meta['duration']*1000:.1f}ms • 0 LLM Tokens"
                )
            else:
                st.caption(
                    f" **Hybrid RAG Retrieval** • {meta['duration']:.2f}s • Dense + BM25 RRF"
                )

            # 3. Parse and render Verified Source Citations
            raw_sources = re.findall(
                r"\[([a-zA-Z0-9_\-\.\s]+\.(?:docx|pptx|html|pdf|txt|json|yaml|yml))\]|【([a-zA-Z0-9_\-\.\s]+\.(?:docx|pptx|html|pdf|txt|json|yaml|yml))】",
                full_answer or "",
                re.IGNORECASE,
            )
            sources = []
            for m in raw_sources:
                doc_name = m[0] or m[1]
                if doc_name and doc_name not in sources:
                    sources.append(doc_name)
            if sources:
                with st.expander(" Verified Sources", expanded=False):
                    for src in sources:
                        st.markdown(f"- 📄 `{src}`")

            # 4. Save to history with telemetry for persistent display
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "meta": meta,
                    "sources": sources,
                }
            )
