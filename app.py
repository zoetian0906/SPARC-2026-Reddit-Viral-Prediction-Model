import gradio as gr
import pandas as pd

# This is a dummy function. Later, it will connect to Zoe's model
def get_viral_score(post_id):
    return f"Status: Model integration pending. Fetched ID: {post_id}"

# Generates the actual website layout
app = gr.Interface(
    fn=get_viral_score,
    inputs=gr.Textbox(label="Enter Reddit Post ID"),
    outputs=gr.Textbox(label="Predicted Virality Index"),
    title="Reddit Virality Predictor",
    description="Enter a post ID to see its predicted performance."
)

if __name__ == "__main__":
    app.launch()
    
