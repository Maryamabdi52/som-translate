import os
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "qalpi_model"
model_path = os.path.join(model_dir, "tf_model.h5")
gdrive_id = "1T0mt42_0145xSAY2WULB9JE9kOqA-aCX"  # File ID-ga saxda ah

# Soo dejiso model-ka haddii uusan jirin
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print("Model-ka Google Drive ayaa la soo dejinayaa...")
    gdown.download(url, model_path, quiet=False)
    print("Soo dejintu waa dhamaatay.")

# Load model & tokenizer
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        inputs = tokenizer([request.text], return_tensors="tf")
        output = model.generate(**inputs)
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
