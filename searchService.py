import os
from serpapi import GoogleSearch
from fastapi import HTTPException
from models import CoffeeChatRequest

# [변경] cx 파라미터 삭제 (SerpApi는 api_key만 있으면 됨)
def find_coffee_chat_targets(request: CoffeeChatRequest, api_key: str):
    try:
        # 1. 검색 쿼리 생성
        skills = " ".join(request.tech_stack[:2])
        # LinkedIn 프로필 검색 쿼리
        search_query = f"site:linkedin.com/in/ {request.company_name} {request.position} {skills}"

        # 2. SerpApi 호출 설정 (google-search-results 라이브러리 사용)
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": api_key,  # SERPAPI_KEY 사용
            "num": 3,            # 결과 개수
            "gl": "kr",          # 한국 지역
            "hl": "ko"           # 한국어
        }

        # 검색 실행
        search = GoogleSearch(params)
        results = search.get_dict()

        # 3. 결과 정제
        real_contacts = []
        
        # 'organic_results'가 실제 검색 결과 목록입니다.
        if "organic_results" in results:
            for item in results["organic_results"]:
                real_contacts.append({
                    "name_and_title": item.get("title"),
                    "profile_url": item.get("link"),
                    "snippet": item.get("snippet")
                })

        if not real_contacts:
            return {
                "status": "empty",
                "query": search_query,
                "message": "검색 결과가 없습니다."
            }

        return {
            "status": "success",
            "query": search_query,
            "results": real_contacts
        }

    except Exception as e:
        print(f"Search Service Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"커피챗 검색 중 오류: {str(e)}")