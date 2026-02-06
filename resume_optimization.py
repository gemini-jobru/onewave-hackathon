from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from google import genai
import json
import re
import asyncio  # 재시도 대기를 위해 추가

app = FastAPI()

# ⚠️ 여기에 새로 발급받은 '깨끗한' API 키를 넣으세요.
GEMINI_API_KEY = "AIzaSyCAlAqpyNTL5iiHh0LVkqtDLsLTOr54Zxs"
client = genai.Client(api_key=GEMINI_API_KEY)

class OptimizationRequest(BaseModel):
    raw_cv: Dict[str, Any]
    job_description: str

@app.post("/cv/optimize")
async def optimize_cv(request: OptimizationRequest):
    max_retries = 1  # 429 에러 시 1번 더 자동 시도
    
    for attempt in range(max_retries + 1):
        try:
            prompt = f"""
            당신은 IT 전문 커리어 코치입니다. 제공된 'raw_cv'를 'job_description'에 맞춰 최적화하세요.
            반드시 {{ 로 시작해서 }} 로 끝나는 순수한 JSON 데이터만 출력하세요. 부연 설명은 하지 마세요.

            기업 공고(JD):
            {request.job_description}

            원본 데이터(raw_cv):
            {json.dumps(request.raw_cv, ensure_ascii=False)}
            """

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            res_text = response.text.strip()
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            
            if not json_match:
                raise ValueError("AI가 유효한 JSON을 생성하지 못했습니다.")
                
            optimized_cv = json.loads(json_match.group())

            return {
                "status": "success",
                "optimized_cv": optimized_cv
            }

        except Exception as e:
            # 429 에러인 경우 잠시 대기 후 재시도
            if "429" in str(e) and attempt < max_retries:
                print(f"Quota Exceeded. Retrying in 1 second... (Attempt {attempt + 1})")
                await asyncio.sleep(1)
                continue
            
            print(f"CRITICAL ERROR: {str(e)}")
            raise HTTPException(status_code=500, detail=f"최적화 오류: {str(e)}")