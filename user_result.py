import json
import os
import numpy as np
# TF-IDF 대신 딥러닝 기반 임베딩 모델 사용
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 파일 경로 설정
# ==========================================
USER_FILE = "user.json"         # 사용자 이력서
COMPANY_FILE = "result.json"    # 기업 공고 데이터
OUTPUT_FILE = "quantification.json" # 최종 결과

# ==========================================
# 2. 데이터 로드
# ==========================================
def load_json(filename):
    if not os.path.exists(filename):
        print(f"❌ '{filename}' 파일을 찾을 수 없습니다.")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# 3. AI 분석용 텍스트 구성 (핵심 역량 위주)
# ==========================================
def create_user_profile(user):
    text = f"희망 직무는 {user.get('desired_job', '')}입니다. "
    text += f"보유 기술은 {', '.join(user.get('skills', []))} 입니다. "
    
    for proj in user.get('projects', []):
        techs = ", ".join(proj.get('tech_stack', []))
        text += f"프로젝트 {proj.get('name', '')}에서 {techs}를 활용하여 {proj.get('description', '')} 경험이 있습니다. "
        
    for exp in user.get('experiences', []):
        text += f"{exp} "
        
    return text

def create_job_requirements(job):
    if 'job' in job:
        job = job['job']
        
    if 'embedding_text' in job:
        return job['embedding_text']

    text = f"{job.get('title', '')}. "
    text += f"담당 업무는 {' '.join(job.get('responsibilities', []))} 입니다. "
    text += f"자격 요건은 {' '.join(job.get('qualifications', []))} 입니다. "
    
    return text

# ==========================================
# 4. AI 매칭 및 수치화
# ==========================================
def calculate_ai_score(user_data, company_data):
    print("🧠 AI 모델 로딩 중... (의미 기반 분석)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🔄 사용자 및 기업 데이터 분석 중...")
    
    # 1. 벡터화
    user_text = create_user_profile(user_data)
    user_vector = model.encode([user_text]) 
    
    job_texts = [create_job_requirements(job) for job in company_data]
    job_vectors = model.encode(job_texts)
    
    # 2. 유사도 계산
    similarities = cosine_similarity(user_vector, job_vectors)[0]
    
    results = []
    for i, score in enumerate(similarities):
        # [수정된 부분] float()로 감싸서 numpy.float32 -> python float 변환 필수!
        match_score = round(float(score) * 100, 1)
        
        # 원본 데이터 가져오기
        job_origin = company_data[i]
        if 'job' in job_origin:
            job_origin = job_origin['job']
            
        # 데이터 정리
        if 'embedding_vector' in job_origin:
            del job_origin['embedding_vector']
        if 'embedding_text' in job_origin:
            del job_origin['embedding_text']

        # AI 분석 코멘트
        if match_score >= 60:
            analysis = "🌟 AI 강력 추천 (직무/스택 일치도 높음)"
        elif match_score >= 40:
            analysis = "✅ 적합 (핵심 역량 일부 부합)"
        elif match_score >= 20:
            analysis = "🤔 검토 필요 (일부 연관성 있음)"
        else:
            analysis = "⚠️ 관련성 낮음 (직무 차이 큼)"

        results.append({
            "companyName": job_origin.get('companyName'),
            "title": job_origin.get('title'),
            "match_score": match_score,
            "ai_analysis": analysis,
            "job_data": job_origin
        })
        
    # 점수 높은 순 정렬
    results.sort(key=lambda x: x['match_score'], reverse=True)
    return results

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    user_data = load_json(USER_FILE)
    company_data = load_json(COMPANY_FILE)
    
    if user_data and company_data:
        quantified_jobs = calculate_ai_score(user_data, company_data)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(quantified_jobs, f, ensure_ascii=False, indent=2)
            
        print(f"\n🎉 분석 완료! 상위 추천 기업:")
        for item in quantified_jobs[:5]:
            print(f"[{item['match_score']}%] {item['companyName']} - {item['title']}")
            
        print(f"\n📂 최종 결과는 '{OUTPUT_FILE}' 파일에 저장되었습니다.")