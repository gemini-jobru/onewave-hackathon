import json
import os
from datetime import datetime, timezone
from typing import List, Dict

# 필요한 라이브러리: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 'sentence-transformers' 라이브러리가 필요합니다.")
    exit()

# ==========================================
# 1. 파일 경로 설정
# ==========================================
INPUT_FILE = "job_data.json"   # 원본 데이터
OUTPUT_FILE = "result.json"    # Supabase 업로드용 가공 데이터

# ==========================================
# 2. 데이터 로드
# ==========================================
def load_data(filename: str) -> List[Dict]:
    if not os.path.exists(filename):
        print(f"❌ '{filename}' 파일이 없습니다.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# 3. 데이터 전처리 (필터링 + 임베딩 벡터 생성)
# ==========================================
def process_job_data(jobs: List[Dict]) -> List[Dict]:
    processed_list = []
    
    # 임베딩 모델 로드 (벡터 생성을 위함)
    print("⚙️  임베딩 모델 로딩 중... (all-MiniLM-L6-v2)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 현재 시간 (필터링 기준)
    now = datetime.now(timezone.utc)
    
    print(f"🔄 데이터 가공 시작 (총 {len(jobs)}건)...")
    
    for job in jobs:
        try:
            # [Step 1] 하드 필터링: 날짜 지난 공고 제외
            # 데이터에 따라 포맷이 다를 수 있어 예외처리
            closing_at_str = job.get('closingAt', '')
            if not closing_at_str: continue # 마감일 없으면 패스
            
            closing_at = datetime.fromisoformat(closing_at_str.replace('Z', '+00:00'))
            
            if closing_at <= now:
                continue # 이미 마감된 공고는 리스트에 담지 않음 (삭제)

            # [Step 2] 임베딩용 텍스트 구성 (직무 + 자격요건)
            # 사용자가 지정한 대로 title과 qualifications만 합칩니다.
            qual_str = " ".join(job.get('qualifications', []))
            embedding_text = f"{job['title']} {qual_str}"
            
            # [Step 3] 벡터 생성 (실제 AI가 이해하는 숫자 배열)
            # Supabase의 'vector' 타입 컬럼에 들어갈 데이터입니다.
            vector = model.encode(embedding_text).tolist()
            
            # [Step 4] 데이터 구조 정리
            # 원본 데이터에 'embedding' 필드를 추가하여 저장합니다.
            job_processed = job.copy()
            
            # 나중에 DB에서 검색할 때 쓸 '텍스트'와 '벡터'를 둘 다 저장해둡니다.
            job_processed['embedding_text'] = embedding_text  # 무엇을 기준으로 임베딩했는지 기록
            job_processed['embedding_vector'] = vector        # 실제 검색에 쓰일 벡터값
            
            processed_list.append(job_processed)
            
        except Exception as e:
            print(f"⚠️ 처리 중 오류 발생 (ID: {job.get('activityId')}): {e}")
            continue
            
    return processed_list

# ==========================================
# 4. 결과 저장
# ==========================================
def save_data(data: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료: 총 {len(data)}건의 유효한 공고가 '{filename}'에 저장되었습니다.")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    raw_jobs = load_data(INPUT_FILE)
    
    if raw_jobs:
        clean_db_data = process_job_data(raw_jobs)
        save_data(clean_db_data, OUTPUT_FILE)