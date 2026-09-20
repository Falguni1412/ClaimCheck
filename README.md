
# ClaimCheck

### Claim-Level LLM Verification Engine

ClaimCheck is an AI-powered verification system that checks individual claims from LLM-generated answers against trusted evidence and classifies them as **SUPPORTED, CONTRADICTED, or UNVERIFIABLE**.

🌐 **Live Demo:** https://claimcheck-gold.vercel.app/

## Problem Statement

Large Language Models can generate convincing but incorrect or unsupported information. ClaimCheck addresses this problem by breaking an AI-generated response into individual claims and verifying each claim against provided ground-truth evidence.

## Key Features

- 🔍 Claim-level fact verification
- 🤖 NLI-based claim analysis
- ✅ SUPPORTED / ❌ CONTRADICTED / ⚠️ UNVERIFIABLE verdicts
- 📚 Evidence-grounded verification
- 🔢 Numerical consistency checking
- 🎯 Confidence and risk scoring
- 📊 Clear verification results
- 🌐 Interactive web interface

## Tech Stack

- **Frontend:** Next.js, React, TypeScript, CSS
- **Backend:** Python, FastAPI, Uvicorn
- **AI/ML:** PyTorch, Hugging Face Transformers
- **NLI Model:** `cross-encoder/nli-MiniLM2-L6-H768`
- **Database:** SQLite
- **Testing:** Pytest

## How to Run

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8000
````

Backend API:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## How to Use

1. Enter an LLM-generated answer.
2. Provide the trusted source/evidence.
3. Click **Run ClaimCheck Verification**.
4. Review each claim's verdict, evidence, confidence, and risk information.

## Team Member

**Falguni Timande**
Developer | AI/ML | Backend | Frontend

## Live Project

🚀 **[https://claimcheck-gold.vercel.app/](https://claimcheck-gold.vercel.app/)**

```

This is intentionally **short and submission-ready**, rather than filling the README with unnecessary technical details.
```

[1]: https://claimcheck-gold.vercel.app/ "ClaimCheck — Claim-Level LLM Verification Engine"
