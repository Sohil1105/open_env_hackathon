import gradio as gr

with open("HF_BLOG.md", "r") as f:
    content = f.read()

# Remove frontmatter for display
if content.startswith("---"):
    _, _, content = content.split("---", 2)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(content)

if __name__ == "__main__":
    demo.launch()
