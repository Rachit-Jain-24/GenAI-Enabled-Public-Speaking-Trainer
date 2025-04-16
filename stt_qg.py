import streamlit as st
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from datetime import datetime
from pydub import AudioSegment
import torchaudio
import torch
import numpy as np

# -------------------- Configure Gemini API --------------------
genai.configure(api_key="AIzaSyDreNylVmU4EeQ8xu06nff9WTIo8NmeBpk")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-2.0-flash")

# -------------------- Speak Function  --------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except RuntimeError as e:
        st.warning("⚠️ Voice output skipped due to engine error.")


# -------------------- Detect if User Input is a Question --------------------
def is_question(text):
    question_words = ["what", "why", "how", "when", "where", "who", "do", "does", "can", "should", "could", "is", "are", "will"]
    return text.strip().endswith("?") or text.lower().split()[0] in question_words

# -------------------- Listen from Microphone --------------------
def listen(timeout=30, phrase_time_limit=30):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            st.warning(" No speech detected within the time limit.")
            return None
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand. Please try again.")
    except sr.RequestError as e:
        st.error(f" Could not request results from Google; {e}")
    return None

# -------------------- Upload MP3 and Convert to WAV --------------------
def upload_audio():
    uploaded_file = st.file_uploader("Upload an MP3 audio file", type=["mp3"])
    if uploaded_file:
        with open("uploaded_audio.mp3", "wb") as f:
            f.write(uploaded_file.read())
        audio = AudioSegment.from_mp3("uploaded_audio.mp3")
        audio.export("uploaded_audio.wav", format="wav")
        return "uploaded_audio.wav"
    return None

# -------------------- Transcribe Uploaded Audio --------------------
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't transcribe the audio."
    except sr.RequestError as e:
        return f"Error with the service: {e}"

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="🎙️ Speech to Questions", page_icon="🎤", layout="wide")
st.title("🎙️ AI Speech-to-Question Assistant")
st.markdown("Speak or upload audio. Get questions based on your transcript. Or ask Gemini directly.")

# -------------------- Session State --------------------

if "script" not in st.session_state: st.session_state.script = ""
if "listening" not in st.session_state: st.session_state.listening = False
if "questions" not in st.session_state: st.session_state.questions = []
if "question_index" not in st.session_state: st.session_state.question_index = 0
if "answers" not in st.session_state: st.session_state.answers = []

# -------------------- Buttons --------------------

col1, col2 = st.columns(2)
with col1:
    if st.button("🎧 Start Listening"):
        st.session_state.listening = True
with col2:
    if st.button("⛔ Stop Listening"):
        st.session_state.listening = False

        if st.session_state.script.strip():
            filename = f"speech_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write(st.session_state.script.strip())
            st.success(f"📄 Transcript saved as `{filename}`")

            prompt = f"""
You are an intelligent assistant. Based on the following speech transcript, generate 5 insightful and thought-provoking questions to ask the speaker:

Transcript:
\"\"\" {st.session_state.script} \"\"\"

Only return the questions, one per line.
"""
            try:
                response = model.generate_content(prompt)
                questions = response.text.strip().split("\n")
                questions = [q.strip("-•1234567890. ") for q in questions if q.strip()]
                st.session_state.questions = questions
                st.session_state.question_index = 0
                st.session_state.answers = []
            except Exception as e:
                st.error(f"⚠️ Gemini API Error: {e}")

# -------------------- Voice Input Listening --------------------

if st.session_state.listening:
    user_input = listen()
    if user_input:
        st.write(f"🗣️ You said: `{user_input}`")

        if is_question(user_input):
            st.info("🤖 Sending your question to Gemini...")
            try:
                response = model.generate_content(user_input)
                st.success("💬 Gemini's Answer:")
                st.write(response.text)
                speak(response.text)
            except Exception as e:
                st.error(f"⚠️ Gemini API Error: {e}")
        else:
            st.session_state.script += user_input + ". "

# -------------------- Upload & Process Audio --------------------

audio_path = upload_audio()
if audio_path:
    st.info("🎧 Audio uploaded! Transcribing...")
    transcription = transcribe_audio(audio_path)
    st.markdown("### 📜 Transcription")
    st.write(f"**Transcription**: {transcription}")

    prompt = f"""
You are an intelligent assistant. Based on the following speech transcript, generate 5 insightful and thought-provoking questions to ask the speaker:

Transcript:
\"\"\" {transcription} \"\"\" 

Only return the questions, one per line.
"""
    try:
        response = model.generate_content(prompt)
        questions = response.text.strip().split("\n")
        questions = [q.strip("-•1234567890. ") for q in questions if q.strip()]
        st.session_state.questions = questions
        st.session_state.question_index = 0
        st.session_state.answers = []
    except Exception as e:
        st.error(f"⚠️ Gemini API Error: {e}")

# -------------------- Show Transcript --------------------
if st.session_state.script:
    st.markdown("### 📝 Transcript So Far")
    st.text(st.session_state.script.strip())
# -------------------- Ask Questions One-by-One --------------------
if st.session_state.questions:
    st.markdown("### ❓ Gemini's Question")

    idx = st.session_state.question_index
    questions = st.session_state.questions

    if idx < len(questions):
        current_q = questions[idx]
        st.write(f"**Q{idx + 1}:** {current_q}")

        if st.button("🔊 Ask Questions"):
            speak(current_q)

        if st.button("🎙️  Answer to the questions"):
            st.info("🎤 Listening for your answer...")
            answer = listen()
            if answer:
                st.success(f"🗣️ You said: {answer}")
                st.session_state.answers.append(answer)

                # --- Summarize the Answer ---
                st.info("📚 Summarizing your answer with Gemini...")
                try:
                    summary_prompt = f"Please summarize the following answer in 1-2 lines:\n\nAnswer: {answer}"
                    response = model.generate_content(summary_prompt)
                    summary = response.text.strip()
                    st.success("📝 Summary of your answer:")
                    st.write(summary)
                    speak("Here's a quick summary of your answer:")
                    speak(summary)
                except Exception as e:
                    st.error(f"❌ Gemini failed to summarize: {e}")

                # --- Ask to Move On ---
                speak("Can I move to the next question?")
                st.info("🎤 Listening for your response (yes or no)...")
                move_on_response = listen()

                if move_on_response and move_on_response.lower() in ["yes", "yeah", "sure", "okay", "ok", "yep"]:
                    st.session_state.question_index += 1
                    st.rerun()

        # 👉 This button should always be available
        if st.button("➡️ Move to the Next Question"):
            st.session_state.question_index += 1
            st.rerun()

    else:
        st.success("🎉 All questions are answered!")
        st.markdown("### 🧠 Your Answers")
        for i, (q, a) in enumerate(zip(questions, st.session_state.answers), 1):
            st.markdown(f"**Q{i}:** {q}\n\n**A{i}:** {a}")

        if st.button("🔁 Start Over"):
            for key in ["script", "listening", "questions", "answers", "question_index"]:
                st.session_state[key] = "" if key == "script" else False if key == "listening" else []
            st.session_state.question_index = 0
            st.rerun()
# -------------------- Reset Everything --------------------
if st.button("🧹 Reset Everything"):
    for key in ["script", "listening", "questions", "answers", "question_index"]:
        st.session_state[key] = "" if key == "script" else False if key == "listening" else []
    st.session_state.question_index = 0
    st.rerun()