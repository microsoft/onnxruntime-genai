import gradio as gr
import requests
import pandas as pd
import json

SLM_ENDPOINT = "http://localhost:8080/completions"
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant. "
    "Please answer the questions very accurately. "
    "Use emojis and markdown as appropriate"
)

# Global variable to store chat history
chat_history = []
context_length = 0


def ask_slm_engine(prompt, history, max_tokens, slider_temp):
    global chat_history
    global context_length

    if not chat_history or len(chat_history) == 0:
        chat_history = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}",
            }
        ]

    chat_history.append({"role": "user", "content": prompt})

    # Format the message as required by your API
    payload = {
        "messages": chat_history,
        "temperature": slider_temp,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}

    # Send the request to your API endpoint
    response = requests.post(SLM_ENDPOINT, json=payload, headers=headers)

    # Extract the response content
    response_content = response.json()

    ai_response = response_content["choices"][0]["message"]
    chat_history.append(ai_response)

    # Print the Response - all of it
    print(json.dumps(response_content, indent=4))

    return ai_response["content"], pd.DataFrame([response_content["kpi"]])


def reset_chat():
    global chat_history
    chat_history = []  # Clear the chat history
    return "", pd.DataFrame(columns=["KPI", "Value"])  # Clear the chat and KPI grid


with gr.Blocks() as demo:
    kpi_grid = gr.Dataframe(
        headers=["KPI", "Value"], datatype=["str", "str"], render=False
    )
    gr.Markdown("<center><h1>Chat with ONNX SLM Engine</h1></center>")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Max Tokens")
                    slider = gr.Slider(100, 4096, 1200, label="Max Tokens")
                with gr.Column():
                    gr.Markdown("## Temperature")
                    slider_temp = gr.Slider(0, 1.0, 0.5, label="Temperature")
                with gr.Column():
                    gr.Markdown("## Reset Chat")
                    reset_button = gr.Button("Reset")
            chatbot = gr.Chatbot(height=200, render=False)
            user_prompt = gr.Textbox(
                placeholder="Ask me a question",
                container=False,
                scale=7,
                render=False,
            )
            gr.ChatInterface(
                ask_slm_engine,
                type="messages",
                additional_inputs=[slider, slider_temp],
                chatbot=chatbot,
                textbox=user_prompt,
                additional_outputs=[kpi_grid],
            )
            reset_button.click(reset_chat, outputs=[chatbot, kpi_grid])

    with gr.Row():
        with gr.Column():
            gr.Markdown("<left><h2>KPI Stats</h2></left>")
            kpi_grid.render()

demo.launch()
