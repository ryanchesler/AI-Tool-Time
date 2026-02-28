"""
AI Tool Jam Hackathon App - Gradio interface for exploring tools, rating, and analytics.
"""

# Monkey-patch gradio_client to fix TypeError when schema is bool (older gradio_client versions)
def _patch_gradio_client():
    try:
        import gradio_client.utils as gu
        _orig_get_type = gu.get_type

        def _patched_get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return _orig_get_type(schema)

        gu.get_type = _patched_get_type
    except Exception:
        pass  # Patch may already exist or gradio_client structure changed


_patch_gradio_client()

import json
import logging
import os
import random
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
DATA_FILE = Path(__file__).parent / "data.json"
IP_TO_NAME_FILE = Path(__file__).parent / "ip_to_name.json"
UPLOADS_DIR = Path(__file__).parent / "uploads"
_data_lock = threading.Lock()

# --- Tool Data (name, url, description) ---
TOOLS_BY_CATEGORY = {
    "Chat & Multimodal Assistants": [
        ("ChatGPT (text + image tools)", "https://chatgpt.com/", "OpenAI's conversational AI with image understanding and generation."),
        ("Gemini", "https://gemini.google.com/", "Google's multimodal AI assistant for chat, code, and analysis."),
        ("Claude", "https://claude.ai/", "Anthropic's AI assistant for writing, analysis, and coding."),
        ("Perplexity", "https://www.perplexity.ai/", "AI-powered search and answer engine with citations."),
        ("NotebookLM", "https://notebooklm.google.com/", "Google's AI for summarizing, querying, and discussing your documents."),
    ],

    "Autonomous Personal Agents & Work": [
        ("Manus", "https://manus.im/", "General-purpose AI agent that can complete multi-step tasks more autonomously."),
        ("Genspark", "https://www.genspark.ai/", "AI agent and search/workflow tool for researching and carrying out tasks."),
        ("OpenClaw", "https://openclaw.ai/", "Autonomous personal AI assistant focused on taking actions across tools like inbox, calendar, and chat."),
    ],

    "Ambient AI & Personal Briefing": [
        ("Huxe", "https://www.huxe.com/", "AI-powered personalized audio briefings built from your inbox, schedule, and chosen information sources."),
    ],

    "World Models & Interactive Simulations": [
        ("Genie 3 / Project Genie", "https://deepmind.google/models/genie/", "Interactive AI-generated worlds and real-time environments from Google DeepMind."),
        ("Marble", "https://marble.worldlabs.ai/", "Multimodal world model from World Labs for creating, editing, and sharing high-fidelity persistent 3D worlds."),
    ],

    "Image Generation & Design": [
        ("ChatGPT Images", "https://openai.com/index/new-chatgpt-images-is-here/", "Generate and edit images directly in ChatGPT."),
        ("Ideogram", "https://ideogram.ai/", "AI image generator with particularly strong text-in-image results."),
        ("Krea", "https://www.krea.ai/", "Real-time AI image generation and enhancement."),
        ("Canva Magic Studio", "https://www.canva.com/magic/", "AI design tools inside Canva for graphics and creative content."),
        ("Adobe Firefly", "https://firefly.adobe.com/", "Adobe's generative AI for images, vectors, and design workflows."),
        ("FLUX (Black Forest Labs)", "https://bfl.ai/", "Popular high-quality image generation model family from Black Forest Labs."),
        ("Photoroom", "https://www.photoroom.com/", "AI background removal and product photo editing."),
    ],

    "Video Generation & Editing": [
        ("OpenAI Sora", "https://openai.com/sora/", "Text-to-video generation from OpenAI."),
        ("Veo (Google DeepMind)", "https://deepmind.google/technologies/veo/", "Google's high-quality video generation model."),
        ("Runway", "https://runwayml.com/", "AI video editing, generation, and visual effects."),
        ("Luma Dream Machine", "https://lumalabs.ai/dream-machine", "Text-and-image-to-video generation from Luma."),
        ("Pika", "https://pika.art/", "AI video creation and editing platform."),
        ("HeyGen (avatars)", "https://www.heygen.com/", "AI avatar and talking-head video generation."),
    ],

    "Music, Voice & Audio": [
        ("Suno", "https://suno.com/", "AI music generation from text prompts."),
        ("Udio", "https://www.udio.com/", "AI music creation and remixing."),
        ("ElevenLabs", "https://elevenlabs.io/", "AI voice synthesis and voice cloning."),
        ("Adobe Podcast (Enhance Speech)", "https://podcast.adobe.com/", "AI speech enhancement for podcasts and recordings."),
        ("Hume", "https://www.hume.ai/", "Voice-first AI focused on natural conversational interaction and expressive speech."),
        ("Whispr Flow", "https://wisprflow.ai/", "Voice dictation and speech-to-text tool for faster computer interaction."),
    ],

    "Coding & Developer Tools": [
        ("Cursor", "https://www.cursor.com/", "AI-powered code editor built on VS Code."),
        ("Windsurf", "https://windsurf.com/", "AI coding assistant and IDE."),
        ("Claude Code", "https://claude.com/product/claude-code", "Claude for coding with terminal and IDE integration."),
        ("GitHub Copilot", "https://github.com/features/copilot", "AI pair programmer integrated into developer workflows."),
        ("Replit Agent", "https://replit.com/products/agent", "AI agent for building and deploying inside Replit."),
        ("Lovable", "https://lovable.dev/", "AI-powered app builder from natural language."),
        ("Bolt", "https://bolt.new/", "AI-assisted full-stack app development in the browser."),
        ("v0 (Vercel)", "https://v0.app/", "AI-generated UI components and app interfaces."),
    ],

    "Research & Reading": [
        ("Elicit", "https://elicit.com/", "AI research assistant for literature review and evidence gathering."),
        ("Consensus", "https://consensus.app/", "AI search engine focused on scientific papers and research findings."),
    ],

    "Work Suites & Collaboration": [
        ("Google Workspace with Gemini", "https://workspace.google.com/solutions/ai/", "Gemini AI integrated across Docs, Sheets, Slides, Gmail, and more."),
        ("Microsoft 365 Copilot", "https://www.microsoft.com/en-us/microsoft-365-copilot", "AI assistant across Word, Excel, PowerPoint, and Outlook."),
        ("Notion AI", "https://www.notion.so/product/ai", "AI writing, summarization, and workspace assistance in Notion."),
        ("Figma AI", "https://www.figma.com/ai/", "AI design and prototyping tools in Figma."),
        ("Gamma", "https://gamma.app/", "AI-native tool for creating presentations, docs, and web-style decks."),
    ],

    "Meeting Notes & Transcription": [
        ("Otter", "https://otter.ai/", "AI meeting assistant for transcription and summaries."),
    ],

    "Automation, Agents & Workflows": [
        ("n8n", "https://n8n.io/", "Workflow automation platform with strong AI and agent integrations."),
        ("Gumloop", "https://www.gumloop.com/", "No-code AI workflow builder for automations, agents, and business tasks."),
        ("Granola", "https://www.granola.ai/", "AI meeting notepad that captures, organizes, and makes meetings searchable."),
    ],
    "Agentic Native Browsers": [
    ("Perplexity Comet", "https://www.perplexity.ai/comet/", "AI-native browser from Perplexity that combines contextual browsing assistance with task automation."),
    ("Dia", "https://www.diabrowser.com/", "AI browser from The Browser Company focused on chatting with tabs, writing help, and contextual browsing."),
    ("Opera Neon", "https://operaneon.com/", "Agentic browser from Opera that can interpret intent and take actions on the live web."),
    ("ChatGPT Atlas", "https://chatgpt.com/atlas/", "OpenAI's browser with ChatGPT built in for page-aware assistance, summaries, and task help."),
    ],
}

RATING_LABELS = {1: "Very Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"}

# Dummy user names for demo data
DUMMY_USERS = ["Alex", "Jordan", "Sam", "Taylor", "Casey", "Morgan", "Riley", "Quinn", "Avery", "Parker"]


def generate_dummy_data(num_submissions: int = 80) -> list:
    """Generate dummy submissions for testing the analytics plots."""
    all_tools = []
    for category, tools in get_all_tools_by_category().items():
        for name, url, _ in tools:
            all_tools.append((name, category))
    submissions = []
    base_time = datetime.now() - timedelta(days=2)
    for _ in range(num_submissions):
        tool_name, category = random.choice(all_tools)
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 25, 35, 25])[0]
        submissions.append({
            "id": str(uuid.uuid4()),
            "user": random.choice(DUMMY_USERS),
            "tool": tool_name,
            "category": category,
            "rating": rating,
            "notes": random.choice(["Great tool!", "Easy to use.", "Needs improvement.", ""]),
            "artifacts": [],
            "timestamp": (base_time + timedelta(minutes=random.randint(0, 2880))).isoformat(),
        })
    return submissions


def _artifacts_to_markdown(artifacts):
    """Convert list of (path, caption) to markdown."""
    if not artifacts:
        return "*No artifacts uploaded yet*"
    lines = []
    for path, caption in artifacts:
        name = os.path.basename(path)
        lines.append(f"- **{name}** — {caption}")
    return "\n".join(lines)


# Extensions for embeddable media in Gallery
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_VIDEO_EXTS = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v",
    ".3gp", ".3gpp", ".ogv", ".mpeg", ".mpg", ".wmv",
}  # gr.Video handles conversion for browser playback
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}


def _get_artifacts_by_tool():
    """Return {tool_name: [(path, caption), ...]} for all tools with artifacts."""
    data = load_data()
    subs = data.get("submissions", [])
    by_tool = {}
    for s in subs:
        for p in s.get("artifacts", []):
            if os.path.exists(p):
                caption = f"{s['user']} — {s['timestamp'][:19]}"
                tool = s["tool"]
                by_tool.setdefault(tool, []).append((p, caption))
    return by_tool


def _split_artifacts(artifacts):
    """Split (path, caption) list into image_items, video_items, audio_items, and other_files."""
    image_items = []
    video_items = []
    audio_items = []
    other_files = []
    for path, caption in artifacts:
        ext = (Path(path).suffix or "").lower()
        if ext in _IMAGE_EXTS:
            image_items.append((path, caption))
        elif ext in _VIDEO_EXTS:
            video_items.append((path, caption))
        elif ext in _AUDIO_EXTS:
            audio_items.append((path, caption))
        else:
            other_files.append((path, caption))
    return image_items, video_items, audio_items, other_files


def _media_type(path: str) -> str:
    """Return 'image', 'video', or 'audio' based on file extension."""
    ext = (Path(path).suffix or "").lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "other"


# --- Data Layer ---

def _ensure_data_dir():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with _data_lock:
        if DATA_FILE.exists():
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                if "custom_tools" not in data:
                    data["custom_tools"] = []
                return data
    return {"submissions": [], "custom_tools": []}


def save_data(data):
    with _data_lock:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)


def load_custom_tools():
    """Return list of (name, url, blurb) for user-added tools."""
    data = load_data()
    return [
        (t["name"], t.get("url", ""), t.get("blurb", ""))
        for t in data.get("custom_tools", [])
    ]


def add_custom_tool(name: str, url: str, blurb: str) -> Optional[str]:
    """Add a custom tool to the Other category. Returns error message or None on success."""
    name = (name or "").strip()
    if not name:
        return "Please enter a tool name."
    data = load_data()
    custom = data.get("custom_tools", [])
    if any(t["name"].strip().lower() == name.lower() for t in custom):
        return f"'{name}' already exists."
    custom.append({"name": name, "url": (url or "").strip(), "blurb": (blurb or "").strip()})
    data["custom_tools"] = custom
    save_data(data)
    return None


def get_all_tools_by_category():
    """Return {category: [(name, url, blurb), ...]} including custom tools in Other."""
    result = dict(TOOLS_BY_CATEGORY)
    result["Other"] = load_custom_tools()
    return result


def load_ip_to_name():
    """Load IP -> name mapping (persists across refreshes)."""
    with _data_lock:
        if IP_TO_NAME_FILE.exists():
            with open(IP_TO_NAME_FILE, "r") as f:
                return json.load(f)
    return {}


def save_ip_name(ip: str, name: str):
    """Save name for this IP address."""
    mapping = load_ip_to_name()
    mapping[ip] = name.strip()
    with _data_lock:
        with open(IP_TO_NAME_FILE, "w") as f:
            json.dump(mapping, f, indent=2)


_logger = logging.getLogger(__name__)


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available for video conversion."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _convert_video_to_h264(src: Path, dest: Path) -> bool:
    """Convert video to H.264 MP4 for browser playback. Returns True if successful."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                str(dest),
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )
        if dest.exists():
            _logger.info("Video converted to H.264: %s -> %s", src, dest)
            return True
        _logger.warning("ffmpeg completed but output missing: %s", dest)
        return False
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else str(e)
        _logger.warning("ffmpeg conversion failed for %s: %s", src, err)
        return False
    except FileNotFoundError:
        _logger.warning("ffmpeg not found; video saved without conversion")
        return False
    except subprocess.TimeoutExpired:
        _logger.warning("ffmpeg conversion timed out for %s", src)
        return False


def add_submission(user: str, tool: str, category: str, rating: int, notes: str, uploaded_files):
    _ensure_data_dir()
    sub_id = str(uuid.uuid4())
    artifact_paths = []
    videos_converted = 0
    videos_saved_without_conversion = 0
    files = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
    for f in files:
        if f is None:
            continue
        path = f.get("path", f) if isinstance(f, dict) else f
        src = Path(path)
        if not src.exists():
            continue
        ext = (src.suffix or "").lower()
        dest = UPLOADS_DIR / f"{sub_id}_{src.stem}{ext}"
        if ext in _VIDEO_EXTS:
            dest_h264 = UPLOADS_DIR / f"{sub_id}_{src.stem}_h264.mp4"
            if _convert_video_to_h264(src, dest_h264):
                artifact_paths.append(str(dest_h264))
                videos_converted += 1
            else:
                shutil.copy2(src, dest)
                artifact_paths.append(str(dest))
                videos_saved_without_conversion += 1
        else:
            shutil.copy2(src, dest)
            artifact_paths.append(str(dest))
    data = load_data()
    data["submissions"].append({
        "id": sub_id,
        "user": user or "Anonymous",
        "tool": tool,
        "category": category,
        "rating": int(rating),
        "notes": notes or "",
        "artifacts": artifact_paths,
        "timestamp": datetime.now().isoformat(),
    })
    save_data(data)
    return sub_id, videos_converted, videos_saved_without_conversion


DEFAULT_RATING = 3


def _coerce_rating(rating, default: int = DEFAULT_RATING) -> int:
    """Ensure rating is 1-5; use default if None, 0, or invalid."""
    if rating is None:
        return default
    try:
        r = int(float(rating))
        return r if 1 <= r <= 5 else default
    except (ValueError, TypeError):
        return default


def _format_submit_success(tool_name: str, videos_converted: int, videos_saved_without_conversion: int, has_artifacts: bool) -> str:
    """Build success message with video conversion status and Gallery refresh hint."""
    msg = f"Thanks! Your feedback for **{tool_name}** has been saved."
    if videos_converted > 0:
        msg += " Video converted for browser playback."
    if videos_saved_without_conversion > 0:
        msg += " Video saved; install ffmpeg for better compatibility."
    if has_artifacts:
        msg += " Refresh the Gallery tab to see your uploads."
    return msg


# --- Custom CSS for section hierarchy ---
CUSTOM_CSS = """
/* Category sections (e.g. Chat, Image, Video) - prominent blue styling */
.category-section {
    border: 2px solid #4a90d9 !important;
    border-radius: 8px !important;
    margin-bottom: 1.5rem !important;
    margin-top: 1rem !important;
    padding: 0.75rem 1rem !important;
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%) !important;
    box-shadow: 0 1px 4px rgba(74, 144, 217, 0.2) !important;
}
.category-section .label-wrap {
    font-weight: 700 !important;
    font-size: 1.15em !important;
    color: #2c5282 !important;
}

/* Individual tools - nested, lighter gray styling */
.tool-item {
    border: 1px solid #cbd5e0 !important;
    border-radius: 6px !important;
    margin: 0.5rem 0 0.5rem 0.5rem !important;
    padding: 0.35rem 0.75rem !important;
    background: #f8fafc !important;
}
.tool-item .label-wrap {
    font-weight: 500 !important;
    font-size: 0.95em !important;
}

/* Name section - subtle distinction */
.name-section {
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    background: #f8fafc !important;
    margin-bottom: 1rem !important;
}

/* Dark mode: adjust category and tool sections */
.dark .category-section {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%) !important;
    border-color: #63b3ed !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
}
.dark .category-section .label-wrap {
    color: #90cdf4 !important;
}

.dark .tool-item {
    background: #2d3748 !important;
    border-color: #4a5568 !important;
}

.dark .name-section {
    background: #2d3748 !important;
    border-color: #4a5568 !important;
}

/* Plotly charts: ensure readable in both modes (template handles this) */
.dark .plotly-graph-div {
    border-radius: 8px;
}
"""


def build_ui():
    with gr.Blocks(title="AI Tool Jam - Hackathon", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown("# AI Tool Jam - Hackathon\nExplore tools, try them out, and share your feedback.")

        with gr.Tabs():
            # --- Tab 1: Tool Explorer ---
            with gr.Tab("Tool Explorer"):
                user_name_state = gr.State(value="")

                with gr.Accordion("Enter your name (required once per session)", open=False, elem_classes="name-section") as name_accordion:
                    name_input = gr.Textbox(
                        label="Your name or alias",
                        placeholder="Enter your name for feedback",
                    )
                    set_name_btn = gr.Button("Save my name")
                    name_status = gr.Markdown("*Your name will be saved for this session.*")

                def save_name(name, _state, request: gr.Request):
                    if (name or "").strip():
                        if request and getattr(request, "client", None):
                            save_ip_name(request.client.host, name.strip())
                        return (
                            name.strip(),
                            f"**Welcome, {name.strip()}!** Your feedback will be attributed to you. Expand above to change your name.",
                            gr.update(open=False),
                        )
                    return _state, "Please enter a name.", gr.update()

                set_name_btn.click(
                    fn=save_name,
                    inputs=[name_input, user_name_state],
                    outputs=[user_name_state, name_status, name_accordion],
                )

                def get_stored_name(request: gr.Request):
                    """Look up name by IP so it persists across page refreshes."""
                    if request and getattr(request, "client", None):
                        ip = request.client.host
                        mapping = load_ip_to_name()
                        name = mapping.get(ip, "")
                        if name:
                            return (
                                name,
                                name,
                                f"**Welcome back, {name}!** (Name remembered from this device.)",
                                gr.update(open=False),
                            )
                    return (
                        "",
                        "",
                        "*Your name will be saved and remembered by IP for future visits (including after refresh).*",
                        gr.update(open=True),
                    )

                demo.load(
                    fn=get_stored_name,
                    outputs=[user_name_state, name_input, name_status, name_accordion],
                )

                # Build per-tool accordions with link, blurb, rating, notes, upload
                for i, (category, tools) in enumerate(get_all_tools_by_category().items()):
                    if i > 0:
                        gr.HTML('<div style="height:2px; background: linear-gradient(90deg, transparent, #4a90d9, transparent); margin: 1.5rem 0;"></div>')
                    with gr.Accordion(category, open=True, elem_classes="category-section"):
                        if category == "Other":
                            # Add-your-own-tool form
                            gr.Markdown("**Add a tool** that's not in the list above:")
                            with gr.Row():
                                other_name = gr.Textbox(label="Tool name", placeholder="e.g. My Favorite AI")
                                other_url = gr.Textbox(label="URL (optional)", placeholder="https://...")
                            other_blurb = gr.Textbox(label="Description (optional)", placeholder="What does it do?")
                            add_tool_btn = gr.Button("Add tool")
                            add_tool_status = gr.Markdown()

                            gr.Markdown("**Rate a custom tool:**")
                            other_tool_dropdown = gr.Dropdown(
                                label="Select custom tool",
                                choices=[t[0] for t in load_custom_tools()],
                                value=None,
                            )

                            def on_add_tool(name, url, blurb):
                                err = add_custom_tool(name, url, blurb)
                                if err:
                                    return gr.update(value=err), gr.update(choices=[t[0] for t in load_custom_tools()])
                                return gr.update(value=f"**{name.strip()}** added! You can rate it below."), gr.update(choices=[t[0] for t in load_custom_tools()])

                            add_tool_btn.click(
                                fn=on_add_tool,
                                inputs=[other_name, other_url, other_blurb],
                                outputs=[add_tool_status, other_tool_dropdown],
                            )
                            with gr.Row():
                                other_rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating (1=Very Poor, 5=Excellent)")
                            other_notes = gr.Textbox(label="Notes", placeholder="What did you try?", lines=3)
                            other_upload = gr.File(label="Upload artifacts", file_count="multiple")
                            other_submit = gr.Button("Submit feedback")
                            other_status = gr.Markdown()

                            def on_other_submit(user, tool_name, rating, notes, files):
                                if not (user or "").strip():
                                    return gr.update(visible=True, value="Please save your name above first.")
                                if not (tool_name or "").strip():
                                    return gr.update(visible=True, value="Please select a tool to rate.")
                                _, v_conv, v_saved = add_submission(user.strip(), tool_name.strip(), "Other", _coerce_rating(rating), notes, files)
                                has_artifacts = bool(files)
                                msg = _format_submit_success(tool_name.strip(), v_conv, v_saved, has_artifacts)
                                return gr.update(visible=True, value=msg)

                            other_submit.click(
                                fn=on_other_submit,
                                inputs=[user_name_state, other_tool_dropdown, other_rating, other_notes, other_upload],
                                outputs=[other_status],
                            )
                        else:
                            for name, url, blurb in tools:
                                with gr.Accordion(f"{name}", open=False, elem_classes="tool-item"):
                                    gr.Markdown(f"**What it is:** {blurb}")
                                    if url:
                                        gr.HTML(
                                            f'<a href="{url}" target="_blank" rel="noopener" '
                                            'style="display:inline-block;padding:10px 20px;background:#4a90d9;color:white!important;'
                                            'text-decoration:none;border-radius:6px;font-weight:600;'
                                            'box-shadow:0 2px 4px rgba(74,144,217,0.3);">'
                                            "Try it here →</a>"
                                        )
                                    with gr.Row():
                                        tool_rating = gr.Slider(
                                            minimum=1,
                                            maximum=5,
                                            step=1,
                                            value=3,
                                            label="Rating (1=Very Poor, 5=Excellent)",
                                        )
                                    tool_notes = gr.Textbox(
                                        label="Notes",
                                        placeholder="What did you try? What surprised you? Any gotchas?",
                                        lines=3,
                                    )
                                    tool_upload = gr.File(
                                        label="Upload artifacts (screenshots, creations)",
                                        file_count="multiple",
                                    )
                                    tool_submit = gr.Button("Submit feedback")
                                    tool_status = gr.Markdown()

                                    def make_submit_handler(tool_name, tool_cat):
                                        def handler(user, rating, notes, files):
                                            if not (user or "").strip():
                                                return gr.update(visible=True, value="Please save your name above first.")
                                            _, v_conv, v_saved = add_submission(user.strip(), tool_name, tool_cat, _coerce_rating(rating), notes, files)
                                            has_artifacts = bool(files)
                                            msg = _format_submit_success(tool_name, v_conv, v_saved, has_artifacts)
                                            return gr.update(visible=True, value=msg)

                                        return handler

                                    tool_submit.click(
                                        fn=make_submit_handler(name, category),
                                        inputs=[user_name_state, tool_rating, tool_notes, tool_upload],
                                        outputs=[tool_status],
                                    )

            # --- Tab 2: Analytics Dashboard ---
            with gr.Tab("Analytics"):
                refresh_btn = gr.Button("Refresh data")

                with gr.Row():
                    test_chart = gr.Plot(label="Times tested per tool")
                    dist_chart = gr.Plot(label="Rating distribution per tool")

                def get_tool_choices():
                    tools = []
                    for tools_list in get_all_tools_by_category().values():
                        for item in tools_list:
                            name = item[0]
                            tools.append(name)
                    return tools

                with gr.Accordion("Drill down: select a tool", open=True):
                    tool_dropdown = gr.Dropdown(
                        label="Select tool",
                        choices=get_tool_choices(),
                        value=None,
                    )
                    tool_rating_chart = gr.Plot(label="Rating distribution for this tool")
                    reviews_display = gr.Dataframe(label="Individual reviews", interactive=False)
                    artifact_display = gr.Markdown(label="Uploaded artifacts")

                def _make_empty_fig(msg: str):
                    fig = go.Figure()
                    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
                    fig.update_layout(
                        template="plotly",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        height=200,
                        margin=dict(t=40, b=40, l=40, r=40),
                    )
                    return fig

                def _chart_layout(fig, height=400):
                    fig.update_layout(
                        template="plotly",
                        height=height,
                        margin=dict(t=50, b=60, l=60, r=40),
                        font=dict(size=12),
                    )
                    return fig

                def refresh_analytics():
                    all_tools = []
                    for tools_list in get_all_tools_by_category().values():
                        for item in tools_list:
                            all_tools.append(item[0])
                    all_tools = sorted(set(all_tools))

                    data = load_data()
                    subs = data.get("submissions", [])
                    if not subs and not all_tools:
                        empty_fig = _make_empty_fig("No data yet — add feedback in the Tool Explorer tab")
                        return empty_fig, empty_fig, _make_empty_fig("Select a tool above"), pd.DataFrame(), "*No artifacts yet*", get_tool_choices()
                    df_subs = pd.DataFrame(subs) if subs else pd.DataFrame(columns=["tool", "rating"])
                    if not df_subs.empty:
                        rating_col = "rating" if "rating" in df_subs.columns else "Rating"
                        df_subs["rating"] = pd.to_numeric(df_subs.get(rating_col, 0), errors="coerce").fillna(0).astype(int)
                        df_subs = df_subs[df_subs["rating"].between(1, 5)]

                    # Test count per tool (include all tools, 0 if not tested)
                    test_counts = df_subs.groupby("tool").size().reindex(all_tools, fill_value=0) if not df_subs.empty else pd.Series(0, index=all_tools)
                    test_counts = test_counts.fillna(0).astype(int)
                    test_df = test_counts.reset_index()
                    test_df.columns = ["Tool", "Times tested"]
                    test_df = test_df.sort_values("Times tested", ascending=True).reset_index(drop=True)

                    # Chart 1: Times tested per tool
                    hovertexts = [f"<b>{t}</b><br>Times tested: {c}" for t, c in zip(test_df["Tool"], test_df["Times tested"])]
                    test_fig = go.Figure(
                        data=[go.Bar(
                            x=test_df["Times tested"],
                            y=test_df["Tool"],
                            orientation="h",
                            marker_color="#4a90d9",
                            hovertext=hovertexts,
                            hoverinfo="text",
                        )]
                    )
                    test_fig.update_layout(
                        title="Times tested per tool",
                        xaxis_title="Number of tests",
                        yaxis_title="",
                        showlegend=False,
                    )
                    test_fig = _chart_layout(test_fig, height=max(400, len(all_tools) * 18))

                    # Chart 2: Rating distribution per tool (stacked bar)
                    rating_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
                    rating_labels = ["1 (Very Poor)", "2 (Below Avg)", "3 (Average)", "4 (Good)", "5 (Excellent)"]
                    dist_fig = go.Figure()
                    if not df_subs.empty:
                        pivot = df_subs.groupby(["tool", "rating"]).size().unstack(fill_value=0)
                        for r in [1, 2, 3, 4, 5]:
                            if r not in pivot.columns:
                                pivot[r] = 0
                        pivot = pivot.reindex(all_tools, fill_value=0).fillna(0)
                        pivot = pivot[[c for c in [1, 2, 3, 4, 5] if c in pivot.columns]]
                        tools_ordered = pivot.sum(axis=1).sort_values(ascending=True).index.tolist()
                        for i, r in enumerate([1, 2, 3, 4, 5]):
                            counts = pivot.reindex(tools_ordered)[r].fillna(0).astype(int).tolist()
                            dist_fig.add_trace(go.Bar(
                                name=rating_labels[i],
                                x=counts,
                                y=tools_ordered,
                                orientation="h",
                                marker_color=rating_colors[i],
                                legendgroup=rating_labels[i],
                            ))
                    else:
                        for i, r in enumerate([1, 2, 3, 4, 5]):
                            dist_fig.add_trace(go.Bar(
                                name=rating_labels[i],
                                x=[0] * len(all_tools),
                                y=all_tools,
                                orientation="h",
                                marker_color=rating_colors[i],
                            ))
                    dist_fig.update_layout(
                        title="Rating distribution per tool",
                        xaxis_title="Count",
                        yaxis_title="",
                        barmode="stack",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    dist_fig = _chart_layout(dist_fig, height=max(400, len(all_tools) * 18))

                    tool_fig = _make_empty_fig("Select a tool above to see its rating distribution")
                    reviews_df = pd.DataFrame()
                    artifacts = []

                    artifact_md = _artifacts_to_markdown(artifacts)
                    return test_fig, dist_fig, tool_fig, reviews_df, artifact_md, get_tool_choices()

                def on_tool_select(tool_name):
                    if not tool_name:
                        return (
                            _make_empty_fig("Select a tool above"),
                            pd.DataFrame(),
                            [],
                        )
                    data = load_data()
                    subs = data.get("submissions", [])
                    subs = [s for s in subs if s["tool"] == tool_name]
                    if not subs:
                        return (
                            _make_empty_fig(f"No reviews yet for {tool_name}"),
                            pd.DataFrame(),
                            [],
                        )
                    df = pd.DataFrame(subs)
                    rating_col = "rating" if "rating" in df.columns else "Rating"
                    df["rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0).astype(int)
                    df = df[df["rating"].between(1, 5)]
                    rating_counts = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
                    fig = go.Figure(
                        data=[go.Bar(
                            x=["1", "2", "3", "4", "5"],
                            y=rating_counts.values,
                            text=rating_counts.values,
                            textposition="auto",
                            marker_color=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
                        )]
                    )
                    fig.update_layout(
                        title=f"Rating distribution: {tool_name}",
                        xaxis_title="Rating (1=Very Poor, 5=Excellent)",
                        yaxis_title="Count",
                        bargap=0.3,
                    )
                    fig = _chart_layout(fig)
                    reviews_df = df[["user", "rating", "notes", "timestamp"]].copy()
                    reviews_df.columns = ["User", "Rating", "Notes", "Timestamp"]
                    artifacts = []
                    for s in subs:
                        for p in s.get("artifacts", []):
                            if os.path.exists(p):
                                artifacts.append((p, f"{s['user']} - {s['timestamp'][:19]}"))
                    return fig, reviews_df, artifacts

                def full_refresh(selected_tool=None):
                    avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, choices = refresh_analytics()
                    if selected_tool:
                        tool_fig, reviews_df, artifacts = on_tool_select(selected_tool)
                        artifact_md = _artifacts_to_markdown(artifacts)
                    return avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, gr.update(choices=choices)

                refresh_btn.click(
                    fn=full_refresh,
                    inputs=[tool_dropdown],
                    outputs=[
                        test_chart,
                        dist_chart,
                        tool_rating_chart,
                        reviews_display,
                        artifact_display,
                        tool_dropdown,
                    ],
                )

                def on_tool_select_wrapper(tool_name):
                    fig, reviews_df, artifacts = on_tool_select(tool_name)
                    artifact_md = _artifacts_to_markdown(artifacts)
                    return fig, reviews_df, artifact_md

                tool_dropdown.change(
                    fn=on_tool_select_wrapper,
                    inputs=[tool_dropdown],
                    outputs=[tool_rating_chart, reviews_display, artifact_display],
                )

                def on_load():
                    result = full_refresh()
                    return result[0], result[1], result[2], result[3], result[4], result[5]

                demo.load(
                    fn=on_load,
                    outputs=[
                        test_chart,
                        dist_chart,
                        tool_rating_chart,
                        reviews_display,
                        artifact_display,
                        tool_dropdown,
                    ],
                )

            # --- Tab 3: Gallery ---
            with gr.Tab("Gallery"):
                gallery_refresh_btn = gr.Button("Refresh gallery")
                gallery_tool_dropdown = gr.Dropdown(
                    label="Select tool",
                    choices=get_tool_choices(),
                    value=None,
                )
                gallery_others_files = gr.File(label="Other files (download)", file_count="multiple")

                with gr.Accordion("Media", open=True):
                    gallery_media_dropdown = gr.Dropdown(
                        label="Select media",
                        choices=[],
                        value=None,
                    )
                    gallery_image = gr.Image(label="Image", type="filepath", visible=False)
                    gallery_video = gr.Video(label="Video", format="mp4", visible=False)
                    gallery_video_download = gr.File(label="Download if playback fails", interactive=False, visible=False)
                    gallery_audio = gr.Audio(label="Audio", type="filepath", visible=False)

                def get_gallery_content(tool_name):
                    by_tool = _get_artifacts_by_tool()
                    empty = (
                        gr.update(value=None),
                        gr.update(choices=[], value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                    )
                    if not tool_name:
                        return gr.update(value=None), *empty
                    artifacts = by_tool.get(tool_name, [])
                    if not artifacts:
                        return gr.update(value=None), *empty
                    image_items, video_items, audio_items, other_files = _split_artifacts(artifacts)
                    other_paths = [p for p, _ in other_files] if other_files else None
                    media_items = image_items + video_items + audio_items
                    media_choices = [(f"{os.path.basename(p)} — {cap}", p) for p, cap in media_items]
                    first_path = media_items[0][0] if media_items else None
                    first_type = _media_type(first_path) if first_path else None
                    media_update = gr.update(choices=media_choices, value=first_path) if media_items else gr.update(choices=[], value=None)
                    img_up = gr.update(visible=(first_type == "image"), value=first_path if first_type == "image" else None)
                    vid_up = gr.update(visible=(first_type == "video"), value=first_path if first_type == "video" else None)
                    vid_dl_up = gr.update(visible=(first_type == "video"), value=first_path if first_type == "video" else None)
                    aud_up = gr.update(visible=(first_type == "audio"), value=first_path if first_type == "audio" else None)
                    return gr.update(value=other_paths), media_update, img_up, vid_up, vid_dl_up, aud_up

                def on_media_select(selected_path):
                    if not selected_path:
                        return (
                            gr.update(visible=False, value=None),
                            gr.update(visible=False, value=None),
                            gr.update(visible=False, value=None),
                            gr.update(visible=False, value=None),
                        )
                    t = _media_type(selected_path)
                    return (
                        gr.update(visible=(t == "image"), value=selected_path if t == "image" else None),
                        gr.update(visible=(t == "video"), value=selected_path if t == "video" else None),
                        gr.update(visible=(t == "video"), value=selected_path if t == "video" else None),
                        gr.update(visible=(t == "audio"), value=selected_path if t == "audio" else None),
                    )

                def gallery_refresh():
                    return (
                        gr.update(choices=get_tool_choices()),
                        gr.update(value=None),
                        gr.update(choices=[], value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                    )

                gallery_tool_dropdown.change(
                    fn=get_gallery_content,
                    inputs=[gallery_tool_dropdown],
                    outputs=[
                        gallery_others_files,
                        gallery_media_dropdown,
                        gallery_image,
                        gallery_video,
                        gallery_video_download,
                        gallery_audio,
                    ],
                )
                gallery_media_dropdown.change(
                    fn=on_media_select,
                    inputs=[gallery_media_dropdown],
                    outputs=[gallery_image, gallery_video, gallery_video_download, gallery_audio],
                )
                gallery_refresh_btn.click(
                    fn=gallery_refresh,
                    outputs=[
                        gallery_tool_dropdown,
                        gallery_others_files,
                        gallery_media_dropdown,
                        gallery_image,
                        gallery_video,
                        gallery_video_download,
                        gallery_audio,
                    ],
                )

                # Accordion browse by category
                gr.Markdown("### Browse by category")
                for cat_idx, (category, tools) in enumerate(get_all_tools_by_category().items()):
                    if cat_idx > 0:
                        gr.HTML('<div style="height:2px; background: linear-gradient(90deg, transparent, #4a90d9, transparent); margin: 1rem 0;"></div>')
                    with gr.Accordion(category, open=False, elem_classes="category-section"):
                        with gr.Row():
                            for name, _url, _blurb in tools:
                                btn = gr.Button(name, size="sm")
                                btn.click(
                                    fn=lambda n=name: (n, *get_gallery_content(n)),
                                    inputs=[],
                                    outputs=[
                                        gallery_tool_dropdown,
                                        gallery_others_files,
                                        gallery_media_dropdown,
                                        gallery_image,
                                        gallery_video,
                                        gallery_video_download,
                                        gallery_audio,
                                    ],
                                )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not _ffmpeg_available():
        _logger.warning("ffmpeg not installed; videos will be saved without H.264 conversion (may not play in browser)")
    _ensure_data_dir()
    demo = build_ui()
    demo.launch(share=True, server_port=7863, show_api=False)
