from fastapi import FastAPI, Query
from huggingface_hub import login
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

login(token="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")

MODEL_NAME = "theobalzeau/my-hate-speech-model"  
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextRequest(BaseModel):
    text: str

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = logits.argmax().item()
    
    return "Message Haineux" if prediction == 1 else "Message Non Haineux"

@app.post("/test")
async def classify_text(request: TextRequest):
    result = predict(request.text)
    print(request.text)
    return {"text": request.text, "prediction": result}

