# 🩺 MedGemma Medical Imaging Analyzer & Assistant

A clinical-assistive medical imaging analysis dashboard powered by **MedGemma** models. This platform processes radiological inputs (such as X-rays, MRI scans, CT scans, and Mammography) to extract clinical findings, generate drafts, and incorporate human-in-the-loop (HITL) clinician sign-off workflows.

---

## 🏗️ Clinical Workflow Architecture

```
           ┌──────────────────────┐
           │ Medical Scan Upload  │  ◄── Ingests X-Ray, CT, or MRI images
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ MedGemma Analyzer    │  ◄── Extracts visual features & patterns
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Draft Report Gen     │  ◄── Formulates Findings & Impressions
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Clinician Sign-off   │  ◄── Human-in-the-loop verification
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Signed Report Export │
           └──────────────────────┘
```

1. **Image Ingestion**: Uploads DICOM-derived standard images.
2. **Clinical Analysis**: MedGemma models scan inputs for fractures, lesions, densities, or visual anomalies.
3. **Drafting Reports**: Formulates structured reports divided into standard medical sections (*Findings*, *Impressions*, and *Recommendations*).
4. **Clinician Sign-off**: Real-time editor letting licensed professionals review, edit, and sign off on findings.

---

## ✨ Key Features

- **Multimodal Integration**: Concurrently evaluates text questions and visual medical parameters.
- **Human-in-the-loop Verification**: Implements validation gates ensuring all reports are checked before export.
- **Structured Reports**: Outputs standard Markdown logs ready for clinical integration.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install streamlit pillow transformers torch
   ```

2. **Run Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

> [!CAUTION]
> **Regulatory Notice**: This system is designed for research and educational assistance. All diagnostic evaluations must be verified by a licensed medical practitioner.
