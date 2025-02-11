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


def ask_slm_engine(prompt, history, max_tokens, slider_temp):
    print(f"\033[35;1mPrompt: {prompt}")
    print(f"History: {history}")
    print(f"Token Max: {max_tokens}")
    print(f"Temp: {slider_temp}\033[0m")

    if not history:
        history = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}",
            }
        ]

    history.append({"role": "user", "content": prompt})

    # Format the message as required by your API
    payload = {
        "messages": history,
        "temperature": slider_temp,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}

    # Send the request to your API endpoint
    response = requests.post(SLM_ENDPOINT, json=payload, headers=headers)

    # Extract the response content
    response_content = response.json()

    # Print the response
    # print(f"\033[32;1mResponse: {json.dumps(response_content, indent=4)}\033[0m")

    ai_response = response_content["choices"][0]["message"]
    history.append(ai_response)
    if len(history) > max_tokens:
        print(f"\033[31;1mResetting History: {len(history)}\033[0m")
        history = None

    return ai_response["content"], pd.DataFrame([response_content["kpi"]])


with gr.Blocks() as demo:
    kpi_grid = gr.Dataframe(
        headers=["KPI", "Value"], datatype=["str", "str"], render=False
    )
    chatbot = gr.Chatbot(height=300, render=False)
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
            chatbot = gr.Chatbot(height=200, render=False)
            user_prompt = gr.Textbox(
                placeholder="Ask me a yes or no question",
                container=False,
                scale=7,
                render=False,
            )
            chat_interface = gr.ChatInterface(
                ask_slm_engine,
                type="messages",
                additional_inputs=[slider, slider_temp],
                chatbot=chatbot,
                textbox=user_prompt,
                additional_outputs=[kpi_grid],
            )

            with gr.Column():
                gr.Markdown("## Reset")
                reset_button = gr.ClearButton([user_prompt, chatbot])
    with gr.Row():
        with gr.Column():
            gr.Markdown("<left><h2>KPI Stats</h2></left>")
            kpi_grid.render()

demo.launch()
