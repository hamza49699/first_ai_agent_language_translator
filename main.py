import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

# ─── Load environment ─────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv("GEMINI_KEY")
if not gemini_api_key:
    st.error("GEMINI_KEY is not set. Add it to your .env file.")
    st.stop()

# ─── Set up Gemini-compatible OpenAI Client ───────────────────────
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

agent = Agent(
    name="Translator",
    instructions="You are a language translator. Always translate Urdu sentences into clear, simple English."
)

# ─── Async wrapper for Streamlit ──────────────────────────────────
async def translate(text: str):
    return await Runner.run(agent, input=text, run_config=config)

# ─── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(page_title="Urdu → English Translator", layout="centered")
st.title("Urdu → English Translator")

text = st.text_area("Enter Urdu text:", height=100)

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some Urdu text.")
    else:
        with st.spinner("Translating..."):
            try:
                result = asyncio.run(translate(text))
                st.success("Translation complete:")
                st.write(result.final_output)
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")
