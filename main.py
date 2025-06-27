from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import os
import gdown

app = FastAPI()

# Load your custom model and tokenizer
model_dir = "qalpi_model"  # ama path-ka saxda ah ee model-kaaga
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Hubi in folder-ka model-ka uu jiro
os.makedirs("qalpi_model", exist_ok=True)

file_id = "1T0mt42_0145xSAY2WULB9JE9kOqA-aCX"
destination = "qalpi_model/tf_model.h5"
if not os.path.exists(destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model.generate(**inputs)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated}
