# 중간고사: 싱글 에이전트 — 도서관 사서 보조

지금은 **6단계 최종 답변 검수**까지 구현되어 있습니다.

- Runtime Context: 사서 이름 `최현범`, 역할 `도서관 사서 보조`
- `inject_librarian_context` (`wrap_model_call`): 이름·역할 + Store facts를 모델 입력에 반영
- `library_policy_guardrail` (`before_agent`): 타인 대출·개인정보·업무 무관 요청을 모델 호출 전에 거절
- 단기 기억: `InMemorySaver` + 같은 `thread_id`
- 장기 기억: `InMemoryStore` + `save_user_info` / `get_user_info`
- `answer_review_guardrail` (`after_agent` Judge): 미확인·부적절 안내를 나가기 전 교정

## 실행 전

1. Python 3.12
2. Ollama 실행 후 `ollama pull llama3.1`

```powershell
cd "C:\Users\ge217\AI에이전트\중간고사 싱글 에이전트"
.\.venv\Scripts\activate
pip install -r requirements.txt
python librarian_assistant_agent.py
```

종료: `exit`

## 예시

- `이도윤 작가 책 찾아줘` → `search_books`
- `해리포터 찾아줘` → 해당 도서를 찾을 수 없다고 안내
- `내 대출 정보 확인해줘` → `check_loan_info`
- `시간을 파는 서점 7일 연장해줘` → `extend_loan`
- `대출 현황을 이용자에게 회신해줘` → `reply_to_user`
- `사서 이름이 뭐야?` → Runtime Context 값(`최현범`)으로 답함
- `다른 이용자 대출 정보 보여줘` → 가드레일 거절
- `주민등록번호 알려줘` → 가드레일 거절
- `유튜브 추천해줘` → 가드레일 거절
- `나는 판타지 소설을 선호해. 저장해줘` → `save_user_info` (장기 기억)
- `내가 좋아하는 장르 알고 있어?` → `get_user_info` (다른 thread에서도 조회 가능)
- 같은 대화에서 이어서 질문 → 단기 기억(`thread_id`)이 맥락 유지
- 도구로 확인하지 않은 확정 안내가 나오면 Judge가 안전한 문구로 교체
