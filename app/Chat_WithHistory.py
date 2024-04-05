import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGUF/GGML models
import datetime
from langserve import RemoteRunnable
from typing import Dict, List, Optional, Sequence
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig

chain = RemoteRunnable("http://localhost:8080/chat/")
logfile = './app/chat_history.txt'
print("loading model...")
stt = datetime.datetime.now()
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")
#MODEL SETTINGS also for DISPLAY

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

with gr.Blocks(theme='ParityError/Interstellar') as demo: 
    #TITLE SECTION
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("<center>"
            + "<h1>🤖 Chat bot retrieves based on your data 🤖</h1></center>")  
            gr.Markdown("""
            **Currently Running**:  [Ollama - mistral](https://ollama.com/library/mistral) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Chat History Log File**: *chat_history.txt*  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
            - Vector Store: [Chroma](https://docs.trychroma.com/deployment) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
            - Embedding: [OllamaEmbeddings - nomic-embed-text](https://ollama.com/library/nomic-embed-text) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            """)         
        gr.Image(value='./app/langchain3.webp', height="100%", width='100%')
   # chat and parameters settings
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height = 425, show_copy_button=True,
                                 avatar_images = ["./app/user.png","./app/ai.png"])
            with gr.Row():
                with gr.Column(scale=14):
                    msg = gr.Textbox(show_label=False, 
                                     placeholder="Enter text",
                                     lines=2)
                submitBtn = gr.Button("\n💬 Send\n", size="lg", variant="primary", min_width=180)

        with gr.Column(min_width=100,scale=2):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.30,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=360,
                        step=4,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    rep_pen = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1.2,
                        step=0.05,
                        interactive=True,
                        label="Repetition Penalty",
                    )
                    gr.Markdown("""
                    ### History lenght
                    Insert num of chat rounds for conversation context
                    """)
                    mem_limit = gr.Slider(
                        minimum=5,
                        maximum=12,
                        value=8,
                        step=1,
                        interactive=True,
                        label="Chat History Lenght",
                    )

                clear = gr.Button("🗑️ Clear All Messages", variant='secondary')
    def user(user_message, history):
        writehistory(f"USER: {user_message}")
        return "", history + [[user_message, None]]

    async def bot(history,t,m,r,limit):
        chatHistory = []
        # always keep len(history) <= memory_limit
        if len(history) > limit:
            chatHistory = history[-limit:]   
            print("History above set limit")
        else:
            chatHistory = history
        # First prompt different because does not contain any context    
        # chat_request = ChatRequest(question=history[-1]["human"], chat_history=[{}])
        chatHistory = [{'human': sublist[0], 'ai': sublist[1]} for sublist in chatHistory]
        if len(history) == 1:
            chat_request = ChatRequest(question=history[-1][0], chat_history=[{}])
        # Prompt woth context
        else:
            chat_request = ChatRequest(question=history[-1][0], chat_history=chatHistory)
        # Preparing the CHATBOT reply
        history[-1][1] = ""
        async for chunk in chain.astream(chat_request, 
                                         config=RunnableConfig(temperature=t, max_new_tokens=m, repetition_penalty=r)):
            history[-1][1] += chunk
            yield history
        writehistory(f"temperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history}\n\n")
        #Log in the terminal the messages
        print(f"USER: {history[-1][0]}\n---\ntemperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history[-1][1]}\n\n")    
    # Clicking the submitBtn will call the generation with Parameters in the slides
    submitBtn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot,temperature,max_length_tokens,rep_pen,mem_limit], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()  #required to yield the streams from the text generation
demo.launch(inbrowser=True)
