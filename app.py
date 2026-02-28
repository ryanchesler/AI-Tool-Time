"""
AI Tool Jam Hackathon App - Gradio interface for exploring tools, rating, and analytics.
"""

import json
import os
import random
import shutil
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
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}


def _get_artifacts_by_tool():
    """Return {tool_name: [(path, caption), ...]} for all tools with artifacts."""
    data = load_data()
    subs = data.get("submissions", [])
    if not subs:
        subs = generate_dummy_data()
    by_tool = {}
    for s in subs:
        for p in s.get("artifacts", []):
            if os.path.exists(p):
                caption = f"{s['user']} — {s['timestamp'][:19]}"
                tool = s["tool"]
                by_tool.setdefault(tool, []).append((p, caption))
    return by_tool


def _split_artifacts(artifacts):
    """Split (path, caption) list into gallery_items (images+videos), audio_items, and other_files."""
    gallery_items = []
    audio_items = []
    other_files = []
    for path, caption in artifacts:
        ext = (Path(path).suffix or "").lower()
        if ext in _IMAGE_EXTS or ext in _VIDEO_EXTS:
            gallery_items.append((path, caption))
        elif ext in _AUDIO_EXTS:
            audio_items.append((path, caption))
        else:
            other_files.append((path, caption))
    return gallery_items, audio_items, other_files


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


def add_submission(user: str, tool: str, category: str, rating: int, notes: str, uploaded_files):
    _ensure_data_dir()
    sub_id = str(uuid.uuid4())
    artifact_paths = []
    files = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
    for f in files:
        if f is None:
            continue
        src = Path(f)
        if not src.exists():
            continue
        ext = src.suffix or ""
        dest = UPLOADS_DIR / f"{sub_id}_{src.stem}{ext}"
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
    return sub_id


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
                                add_submission(user.strip(), tool_name.strip(), "Other", int(rating), notes, files)
                                return gr.update(visible=True, value=f"Thanks! Your feedback for **{tool_name}** has been saved.")

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
                                            add_submission(user.strip(), tool_name, tool_cat, int(rating), notes, files)
                                            return gr.update(visible=True, value=f"Thanks! Your feedback for **{tool_name}** has been saved.")

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
                    avg_chart = gr.Plot(label="Average rating per tool")
                    dist_chart = gr.Plot(label="Distribution of all ratings (1-5)")

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
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        height=200,
                        margin=dict(t=40, b=40, l=40, r=40),
                    )
                    return fig

                def _chart_layout(fig, height=400):
                    fig.update_layout(
                        height=height,
                        margin=dict(t=50, b=60, l=60, r=40),
                        font=dict(size=12),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    return fig

                def refresh_analytics():
                    data = load_data()
                    subs = data.get("submissions", [])
                    if not subs:
                        subs = generate_dummy_data()
                    df_subs = pd.DataFrame(subs)
                    agg = df_subs.groupby(["tool", "category"]).agg(
                        Reviews=("rating", "count"),
                        Avg_Rating=("rating", "mean"),
                    ).reset_index()
                    agg.columns = ["Tool", "Category", "Reviews", "Avg Rating"]
                    agg["Avg Rating"] = agg["Avg Rating"].round(2)
                    df = agg.sort_values("Avg Rating", ascending=True).reset_index(drop=True)

                    hovertexts = [
                        f"<b>{row['Tool']}</b><br>Avg Rating: {row['Avg Rating']:.2f}<br>Reviews: {row['Reviews']}"
                        for _, row in df.iterrows()
                    ]
                    avg_fig = go.Figure(
                        data=[go.Bar(
                            x=df["Avg Rating"],
                            y=df["Tool"],
                            orientation="h",
                            marker=dict(
                                color=df["Avg Rating"].tolist(),
                                colorscale="RdYlGn",
                                cmin=1,
                                cmax=5,
                                showscale=False,
                            ),
                            hovertext=hovertexts,
                            hoverinfo="text",
                        )]
                    )
                    avg_fig.update_layout(
                        title="Average rating per tool",
                        xaxis_title="Average Rating",
                        yaxis_title="",
                        xaxis=dict(range=[0.5, 5.5], dtick=1),
                        showlegend=False,
                    )
                    avg_fig = _chart_layout(avg_fig, height=max(400, len(df) * 18))

                    rating_counts = df_subs["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
                    dist_fig = go.Figure(
                        data=[go.Bar(
                            x=["1 (Very Poor)", "2 (Below Avg)", "3 (Average)", "4 (Good)", "5 (Excellent)"],
                            y=rating_counts.values,
                            marker_color=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
                        )]
                    )
                    dist_fig.update_layout(
                        title="Distribution of all ratings",
                        xaxis_title="Rating",
                        yaxis_title="Count",
                        bargap=0.3,
                    )
                    dist_fig = _chart_layout(dist_fig)

                    tool_fig = _make_empty_fig("Select a tool above to see its rating distribution")
                    reviews_df = pd.DataFrame()
                    artifacts = []

                    artifact_md = _artifacts_to_markdown(artifacts)
                    return avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, get_tool_choices()

                def on_tool_select(tool_name):
                    if not tool_name:
                        return (
                            _make_empty_fig("Select a tool above"),
                            pd.DataFrame(),
                            [],
                        )
                    data = load_data()
                    subs = data.get("submissions", [])
                    if not subs:
                        subs = generate_dummy_data()
                    subs = [s for s in subs if s["tool"] == tool_name]
                    if not subs:
                        return (
                            _make_empty_fig(f"No reviews yet for {tool_name}"),
                            pd.DataFrame(),
                            [],
                        )
                    df = pd.DataFrame(subs)
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

                def full_refresh():
                    avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, choices = refresh_analytics()
                    return avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, gr.update(choices=choices)

                refresh_btn.click(
                    fn=full_refresh,
                    outputs=[
                        avg_chart,
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
                        avg_chart,
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
                gallery_display = gr.Gallery(
                    label="Images & videos",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                )
                gallery_others_files = gr.File(label="Other files (download)", file_count="multiple")

                with gr.Accordion("Audio", open=True):
                    gallery_audio_dropdown = gr.Dropdown(
                        label="Select audio to play",
                        choices=[],
                        value=None,
                    )
                    gallery_audio_player = gr.Audio(label="Play", type="filepath")

                def get_gallery_content(tool_name):
                    by_tool = _get_artifacts_by_tool()
                    if not tool_name:
                        return [], gr.update(value=None), gr.update(choices=[], value=None), None
                    artifacts = by_tool.get(tool_name, [])
                    if not artifacts:
                        return [], gr.update(value=None), gr.update(choices=[], value=None), None
                    gallery_items, audio_items, other_files = _split_artifacts(artifacts)
                    other_paths = [p for p, _ in other_files] if other_files else None
                    audio_choices = [(f"{os.path.basename(p)} — {cap}", p) for p, cap in audio_items]
                    audio_value = audio_items[0][0] if audio_items else None
                    audio_update = gr.update(choices=audio_choices, value=audio_value) if audio_items else gr.update(choices=[], value=None)
                    return gallery_items, gr.update(value=other_paths), audio_update, audio_value

                def on_gallery_tool_select(tool_name):
                    result = get_gallery_content(tool_name)
                    return result[0], result[1], result[2], result[3]

                def on_audio_dropdown_change(selected_path):
                    return selected_path

                def gallery_refresh():
                    return gr.update(choices=get_tool_choices()), [], gr.update(value=None), gr.update(choices=[], value=None), None

                gallery_tool_dropdown.change(
                    fn=on_gallery_tool_select,
                    inputs=[gallery_tool_dropdown],
                    outputs=[gallery_display, gallery_others_files, gallery_audio_dropdown, gallery_audio_player],
                )
                gallery_audio_dropdown.change(
                    fn=on_audio_dropdown_change,
                    inputs=[gallery_audio_dropdown],
                    outputs=[gallery_audio_player],
                )
                gallery_refresh_btn.click(
                    fn=gallery_refresh,
                    outputs=[gallery_tool_dropdown, gallery_display, gallery_others_files, gallery_audio_dropdown, gallery_audio_player],
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
                                    outputs=[gallery_tool_dropdown, gallery_display, gallery_others_files, gallery_audio_dropdown, gallery_audio_player],
                                )

    return demo


if __name__ == "__main__":
    _ensure_data_dir()
    demo = build_ui()
    demo.launch(share=False, server_port=7863)
