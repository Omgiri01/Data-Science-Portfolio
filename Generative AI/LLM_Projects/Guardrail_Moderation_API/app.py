from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI(
    title="LLM Guardrail Moderation API",
    description="An API offering input and output guardrails for LLMs, checking for safety, toxic content, and PII leakage.",
    version="1.0.0"
)

class ModerationRequest(BaseModel):
    text: str

class ModerationResponse(BaseModel):
    is_safe: bool
    flagged_categories: list
    cleaned_text: str

# PII leakage patterns (Email, Phone, Credit Card)
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b"
}

# Forbidden / Harmful keyword category check
HARMFUL_KEYWORDS = {
    "malware_generation": ["hack", "exploit", "malware", "ransomware", "trojan", "sql injection"],
    "hate_speech": ["slur", "hate", "harass", "abuse"],
    "credentials_theft": ["password", "secret key", "private key", "auth token"]
}

def scan_text(text: str):
    flagged = []
    cleaned = text
    
    # 1. Check for PII and redact it
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            flagged.append(f"PII_leakage_{pii_type}")
            cleaned = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", cleaned)
            
    # 2. Check for harmful keywords (case-insensitive)
    lower_text = text.lower()
    for category, keywords in HARMFUL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower_text:
                flagged.append(category)
                break
                
    return flagged, cleaned

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_text(request: ModerationRequest):
    """
    Moderates incoming prompt inputs or outgoing LLM responses.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    flagged, cleaned = scan_text(request.text)
    is_safe = len(flagged) == 0
    
    return ModerationResponse(
        is_safe=is_safe,
        flagged_categories=flagged,
        cleaned_text=cleaned
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
