import torch
import gradio as gr
from transformers import AutoTokenizer, pipeline


model = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model)
falcon_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def generate(input):
    output = falcon_pipeline(
        input,
        max_length=40,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    return output[0]['generated_text']


def respond(message, chat_history):
    bot_message = generate(message)
    chat_history.append((message, bot_message))
    
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

demo.queue().launch()
