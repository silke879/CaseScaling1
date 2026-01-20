from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
import os, json, time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()
if os.path.isdir("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("public/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse("<h1>Chat API - Use /docs</h1>")

@app.get("/health")
async def health_check():
    return {"status": "ok"}


SCORING_SYSTEM_PROMPT = """You are an impartial AI judge evaluating closed customer support tickets.
You MUST:
- Apply the scoring rubrics exactly as provided
- Return structured, objective judgments
- Do NOT invent data
- Be concise and factual

SCORING RUBRICS:
Time to Resolution Score (0–5):
≤ 2h → 5
≤ 8h → 4
≤ 24h → 3
≤ 72h → 2
> 72h → 1

Customer Satisfaction Score (0–5):
CSAT 5 → 5
CSAT 4 → 4
CSAT 3 → 3
CSAT 2 → 2
CSAT 1 → 1

Overall Score = Time Score + CSAT Score (0–10)

Output MUST be valid JSON with:
time_score, csat_score, overall_score, rationale, open field for comments
"""

@app.post("/chat", response_model=ChatOpenAI)
def build_clean_prompt(csv_row):
    return f"""
You are a strict JSON data cleaning assistant.
Return ONLY valid JSON.

Keep only:
- Ticket ID
- Subject
- Time to Resolution (hours): calculated as Time to resolution - First response time
- Customer Satisfaction Rating

Do NOT invent values.
Replace missing values with null.

CSV record:
{csv_row}

JSON:
"""

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()

def clean_row_llm(row_df, _llm = None, max_retries=3):
    csv_row = row_df
    prompt = build_clean_prompt(csv_row)
    if _llm is not None:
        resp = _llm.invoke(prompt)
    else:
        resp = llm.invoke(prompt)
    txt = strip_code_fences(resp.content) if hasattr(resp, "content") else resp

    try:
        obj = json.loads(txt)
        required = {
            "Ticket ID",
            "Subject",
            "Time to Resolution (hours)",
            "Customer Satisfaction Rating"
        }
        if not required.issubset(obj.keys()):
            raise ValueError("Missing keys")
        return obj

    except Exception as e:
        if max_retries <= 0:
            raise RuntimeError(f"LLM output invalid after retries: {txt}") from e

        return clean_row_llm(
            row_df,
            _llm =_llm,
            max_retries = max_retries - 1
        )



def judge_record_llm(cleaned, _llm = None, max_retries=3):

    user_prompt = f"""
Time to Resolution (hours): {cleaned.get("Time to Resolution (hours)")}
Customer Satisfaction Rating: {cleaned.get("Customer Satisfaction Rating")}
"""
    llm_to_use = _llm or llm


    try:
        resp = llm_to_use.invoke([
            ("system", SCORING_SYSTEM_PROMPT),
            ("user", user_prompt)
        ])
        txt = resp.content if hasattr(resp, "content") else resp
        txt = strip_code_fences(txt)
        return json.loads(txt)

    except Exception as e:
        if max_retries <= 0:
            raise RuntimeError(f"LLM output invalid after retries: {txt}") from e
        # retry
        return judge_record_llm(cleaned, _llm=_llm, max_retries=max_retries - 1)



@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    results = []
    failures = 0

    for i in range(len(df)):
        row_df = df.iloc[i:i+1]

        start = time.perf_counter()
        cleaned = clean_row_llm(row_df.to_csv(index=False))

        try:
            judgement = judge_record_llm(cleaned)
        except Exception:
            failures += 1
            judgement = {
                "time_score": -1,
                "csat_score": -1,
                "overall_score": -1,
                "rationale": "Fallback scoring",
                "comments": "LLM output invalid"
            }

        print(cleaned)
        print(judgement)
        einde = time.perf_counter()
        duur_sec = einde - start
        print(duur_sec)

        results.append({
            "cleaned": cleaned,
            "judgement": judgement
        })

    return JSONResponse({
        "records": results,
        "total": len(results),
        "fallbacks": failures
    })



