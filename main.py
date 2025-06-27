from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

app = FastAPI()

# Load your custom model and tokenizer
model_dir = "qalpi_model"  # ama path-ka saxda ah ee model-kaaga
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model.generate(**inputs)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated}
