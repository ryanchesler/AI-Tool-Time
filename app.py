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
        ("Perplexity", "https://www.perplexity.ai/", "AI-powered search engine that cites sources."),
        ("NotebookLM", "https://notebooklm.google.com/", "Google's AI for summarizing and querying your documents."),
    ],
    "Image Generation & Design": [
        ("ChatGPT Images", "https://openai.com/index/new-chatgpt-images-is-here/", "Generate and edit images directly in ChatGPT."),
        ("Midjourney", "https://www.midjourney.com/", "Discord-based AI image generation known for artistic quality."),
        ("Ideogram", "https://ideogram.ai/", "AI image generator with strong text-in-image capabilities."),
        ("Krea", "https://www.krea.ai/", "Real-time AI image generation and enhancement."),
        ("Canva Magic Studio", "https://www.canva.com/magic/", "AI design tools built into Canva for graphics and presentations."),
        ("Adobe Firefly", "https://firefly.adobe.com/", "Adobe's generative AI for images, vectors, and design."),
        ("FLUX (Black Forest Labs)", "https://bfl.ai/", "High-quality open-source image generation models."),
        ("Stability AI (Stable Diffusion)", "https://stability.ai/", "Open-source image generation and editing tools."),
        ("Photoroom", "https://www.photoroom.com/", "AI background removal and product photo editing."),
        ("Pixelcut", "https://www.pixelcut.ai/home", "AI-powered product photography and background removal."),
    ],
    "Video Generation & Editing": [
        ("OpenAI Sora", "https://openai.com/sora/", "Text-to-video generation from OpenAI."),
        ("Veo (Google DeepMind)", "https://deepmind.google/technologies/veo/", "Google's high-quality video generation model."),
        ("Runway", "https://runwayml.com/", "AI video editing, generation, and effects."),
        ("Luma Dream Machine", "https://lumalabs.ai/dream-machine", "Text and image to video generation."),
        ("Pika", "https://pika.art/", "AI video creation and editing platform."),
        ("HeyGen (avatars)", "https://www.heygen.com/", "AI avatar and talking-head video generation."),
        ("Descript", "https://www.descript.com/", "AI-powered video and podcast editing with transcription."),
        ("VEED", "https://www.veed.io/", "Online video editor with AI subtitles and effects."),
        ("CapCut", "https://www.capcut.com/", "Free video editor with AI tools and templates."),
    ],
    "Music, Voice & Audio": [
        ("Suno", "https://suno.com/", "AI music generation from text prompts."),
        ("Udio", "https://www.udio.com/", "AI music creation and remixing."),
        ("ElevenLabs", "https://elevenlabs.io/", "AI voice synthesis and cloning."),
        ("Adobe Podcast (Enhance Speech)", "https://podcast.adobe.com/", "AI speech enhancement for podcasts."),
    ],
    "Coding & Developer Tools": [
        ("Cursor", "https://www.cursor.com/", "AI-powered code editor built on VS Code."),
        ("Windsurf", "https://windsurf.com/", "AI coding assistant and IDE."),
        ("Claude Code", "https://claude.com/product/claude-code", "Claude for coding with terminal and IDE integration."),
        ("GitHub Copilot", "https://github.com/features/copilot", "AI pair programmer in your IDE."),
        ("Devin", "https://devin.ai/", "Autonomous AI software engineer."),
        ("Replit Agent", "https://replit.com/products/agent", "AI agent for building and deploying in Replit."),
        ("Lovable", "https://lovable.dev/", "AI-powered app builder from natural language."),
        ("Bolt", "https://bolt.new/", "AI-assisted full-stack development."),
        ("v0 (Vercel)", "https://v0.app/", "AI-generated UI components and interfaces."),
    ],
    "Research & Reading": [
        ("Elicit", "https://elicit.com/", "AI research assistant for literature review and citations."),
        ("Consensus", "https://consensus.app/", "AI search across scientific papers."),
        ("SciSpace", "https://scispace.com/", "AI for reading and understanding research papers."),
    ],
    "Work Suites & Collaboration": [
        ("Google Workspace with Gemini", "https://workspace.google.com/solutions/ai/", "Gemini AI integrated across Docs, Sheets, Slides, and more."),
        ("AI for Presentations with Google Slides", "https://workspace.google.com/resources/presentation-ai/", "AI-assisted slide creation and design."),
        ("Gemini in the side panel", "https://workspaceupdates.googleblog.com/2024/06/gemini-in-side-panel-of-google-docs-sheets-slides-drive.html", "Gemini sidebar in Google Docs, Sheets, Slides, and Drive."),
        ("Microsoft 365 Copilot", "https://www.microsoft.com/en-us/microsoft-365-copilot", "AI assistant across Word, Excel, PowerPoint, and Outlook."),
        ("Slack AI", "https://slack.com/features/ai", "AI search and summarization in Slack."),
        ("Zoom AI Companion", "https://www.zoom.com/en/products/ai-assistant/", "AI meeting assistant for Zoom."),
        ("Notion AI", "https://www.notion.so/product/ai", "AI writing and summarization in Notion."),
        ("Figma AI", "https://www.figma.com/ai/", "AI design tools in Figma."),
    ],
    "Meeting Notes & Transcription": [
        ("Otter", "https://otter.ai/", "AI meeting assistant for transcription and summaries."),
        ("Fireflies", "https://fireflies.ai/", "AI meeting transcription and search."),
    ],
}

RATING_LABELS = {1: "Very Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"}

# Dummy user names for demo data
DUMMY_USERS = ["Alex", "Jordan", "Sam", "Taylor", "Casey", "Morgan", "Riley", "Quinn", "Avery", "Parker"]


def generate_dummy_data(num_submissions: int = 80) -> list:
    """Generate dummy submissions for testing the analytics plots."""
    all_tools = []
    for category, tools in TOOLS_BY_CATEGORY.items():
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


# --- Data Layer ---

def _ensure_data_dir():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with _data_lock:
        if DATA_FILE.exists():
            with open(DATA_FILE, "r") as f:
                return json.load(f)
    return {"submissions": []}


def save_data(data):
    with _data_lock:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)


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

                with gr.Accordion("Enter your name (required once per session)", open=True, elem_classes="name-section") as name_accordion:
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
                        return name.strip(), f"**Welcome, {name.strip()}!** Your feedback will be attributed to you. Expand above to change your name."
                    return _state, "Please enter a name."

                set_name_btn.click(
                    fn=save_name,
                    inputs=[name_input, user_name_state],
                    outputs=[user_name_state, name_status],
                )

                def get_stored_name(request: gr.Request):
                    """Look up name by IP so it persists across page refreshes."""
                    if request and getattr(request, "client", None):
                        ip = request.client.host
                        mapping = load_ip_to_name()
                        name = mapping.get(ip, "")
                        if name:
                            return name, name, f"**Welcome back, {name}!** (Name remembered from this device.)"
                    return "", "", "*Your name will be saved and remembered by IP for future visits (including after refresh).*"

                # Build per-tool accordions with link, blurb, rating, notes, upload
                for i, (category, tools) in enumerate(TOOLS_BY_CATEGORY.items()):
                    if i > 0:
                        gr.HTML('<div style="height:2px; background: linear-gradient(90deg, transparent, #4a90d9, transparent); margin: 1.5rem 0;"></div>')
                    with gr.Accordion(category, open=True, elem_classes="category-section"):
                        for name, url, blurb in tools:
                            with gr.Accordion(f"{name}", open=False, elem_classes="tool-item"):
                                gr.Markdown(f"**What it is:** {blurb}")
                                gr.Markdown(f"[**Try it here →**]({url})")
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

                demo.load(
                    fn=get_stored_name,
                    outputs=[user_name_state, name_input, name_status],
                )

            # --- Tab 2: Analytics Dashboard ---
            with gr.Tab("Analytics"):
                refresh_btn = gr.Button("Refresh data")

                with gr.Row():
                    summary_table = gr.Dataframe(
                        label="Summary: Tools by reviews and average rating",
                        interactive=False,
                    )
                with gr.Row():
                    avg_chart = gr.Plot(label="Average rating per tool")
                    dist_chart = gr.Plot(label="Distribution of all ratings (1-5)")

                def get_tool_choices():
                    tools = []
                    for tools_list in TOOLS_BY_CATEGORY.values():
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

                    avg_fig = px.bar(
                        df,
                        y="Tool",
                        x="Avg Rating",
                        orientation="h",
                        color="Avg Rating",
                        color_continuous_scale="RdYlGn",
                        range_color=[1, 5],
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
                    return df, avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, get_tool_choices()

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
                    df, avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, choices = refresh_analytics()
                    return df, avg_fig, dist_fig, tool_fig, reviews_df, artifact_md, gr.update(choices=choices)

                refresh_btn.click(
                    fn=full_refresh,
                    outputs=[
                        summary_table,
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
                    return result[0], result[1], result[2], result[3], result[4], result[5], result[6]

                demo.load(
                    fn=on_load,
                    outputs=[
                        summary_table,
                        avg_chart,
                        dist_chart,
                        tool_rating_chart,
                        reviews_display,
                        artifact_display,
                        tool_dropdown,
                    ],
                )

    return demo


if __name__ == "__main__":
    _ensure_data_dir()
    demo = build_ui()
    demo.launch(share=False, server_port=7863)
