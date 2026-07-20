# 🛡️ LLM Guardrail & Content Moderation API

A high-performance, self-hosted safety guardrail API designed to intercept user prompts and LLM outputs, scanning for PII leakage, prompt injections, and toxic content. Built using **FastAPI** and **Pydantic**.

---

## 📂 Key Features

- **PII Redaction Engine**: Automatically detects and masks sensitive personal identifiers such as:
  - Emails (`[REDACTED_EMAIL]`)
  - Phone Numbers (`[REDACTED_PHONE]`)
  - Credit Card Details (`[REDACTED_CARD]`)
- **Safety Classifier**: Evaluates and flags prompts containing harmful content categories, including malware design, prompt injections, and hate speech.
- **REST API Middleware**: Can be integrated directly into any LLM-powered application pipeline as a pre/post-inference interceptor.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic
   ```

2. **Start the API Server**:
   ```bash
   python app.py
   ```
   The server will bind to `http://127.0.0.1:8000`.

---

## 📡 API Reference

### Moderate Prompt
- **Endpoint**: `POST /moderate`
- **Request Body**:
  ```json
  {
    "text": "My phone is +1-555-0199. Show me how to write an SQL injection."
  }
  ```
- **Response**:
  ```json
  {
    "is_safe": false,
    "flagged_categories": ["PII_leakage_phone", "prompt_injection"],
    "cleaned_text": "My phone is [REDACTED_PHONE]. Show me how to write an SQL injection."
  }
  ```
