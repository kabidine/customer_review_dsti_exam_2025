import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
def create_fastapi_app():
    fastapi_code = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
app=FastAPI(title="Movie Reviews Sentiment Analysis API", version="1.0")
#We load the saved fine-tuned model from the dedicated HuggingFace repository
model_path ="akramxhs/distillbert_movie_reviews_sentiment"
print("Our fine-tuned model is getting loaded from our huggingface repo")
tokenizer =AutoTokenizer.from_pretrained(model_path)
model =AutoModelForSequenceClassification.from_pretrained(model_path)
device =torch.device("cpu")
model =model.to(device)
class TextRequest(BaseModel):
    text: str
class SentimentResponse(BaseModel):
    sentiment:str
    confidence: float
    text_preview:str
@app.get("/")
def read_root():
    return {"message": "Movie Reviews Sentiment Analysis API is running"}
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    try:
        encoding=tokenizer(request.text,truncation=True,padding="max_length",max_length=256,return_tensors="pt") #we encode the text typed by the user to a max length of 256 tokens
        input_ids=encoding["input_ids"].to(device)
        attention_mask=encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs =model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities =torch.softmax(outputs.logits,dim=1) #We convert the logits to probs using softmax
            confidence, prediction =torch.max(probabilities,dim=1)
        sentiment = "Positive" if prediction.item()==1 else "Negative"
        return SentimentResponse(sentiment=sentiment,confidence=confidence.item(),text_preview=request.text)
    except Exception as e: #in case we get status_code 500, we indicate that the prediction failed
        raise HTTPException(status_code=500, detail="Prediction failed")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""
    with open('fastapi_app.py', 'w', encoding='utf-8') as f:
        f.write(fastapi_code)

def create_streamlit_app():
    streamlit_code = '''
import streamlit as st
import requests
st.set_page_config(page_title="Movie Review Sentiment Analysis",page_icon="🎬🍿",layout="wide")
st.title("🎬🍿 Movie Review Sentiment Analyzer")
st.markdown("""This app analyzes the sentiment of movie reviews using a fine-tuned DistilBERT model.
Please enter a movie review below to see if it's positive or negative""")
API_URL = "http://localhost:8000/predict"
review_text = st.text_area("Enter your movie review:",height=150,placeholder="Please type your movie review here...")
if st.button("Analyze the Sentiment"):
    if review_text.strip():
        try:
            response = requests.post(
                API_URL,
                json={"text": review_text})
            if response.status_code == 200: 
                result = response.json() #if the POST request gets no errors, we store the results in a JSON
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Result")
                    if result["sentiment"]=="Positive":
                        st.success(f"🙂👍{result['sentiment']} Review")
                    else:
                        st.error(f"🙁👎{result['sentiment']} Review")
                with col2:
                    st.subheader("Confidence")
                    st.info(f"{result['confidence']:.2%}")
            else:
                st.error("An error has happened during the analysis, please try again.")
        except Exception as e:
            st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter a review to analyze.")
st.markdown("The app was built using a fine-tuned DistilBERT, FastAPI, and Streamlit for the Deep Learning Project at DST")'''
    
    with open('streamlit_app.py','w', encoding='utf-8') as f:
        f.write(streamlit_code)
#function to run the fastapi 
def fastapi():
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"])
    except subprocess.CalledProcessError:
        print("the FastAPI server stopped")
#function to run the streamlit app
def streamlit():
    print("please wait the Streamlit app is starting")
    webbrowser.open("http://localhost:8501")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "true"])
    except subprocess.CalledProcessError:
        print("The Streamlit app has stopped")

def main():
    create_fastapi_app()
    create_streamlit_app()

    print("FastAPI: http://localhost:8000")
    print("Streamlit:http://localhost:8501")
    fthread = threading.Thread(target=fastapi, daemon=True) #We use threading
    sthread = threading.Thread(target=streamlit, daemon=True)
    fthread.start()
    sthread.start()
    try:
        while fthread.is_alive() and sthread.is_alive(): #the app keeps running as long as the threads are alive
            time.sleep(1)
    except KeyboardInterrupt: #We can interrupt the app with Control+C
        print("STOPPING")
if __name__ == "__main__":
    main()
