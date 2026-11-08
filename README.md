# 중간고사: 싱글 에이전트 — 도서관 사서 보조

지금은 **4단계 가드레일**까지 구현되어 있습니다.

- Runtime Context: 사서 이름 `최현범`, 역할 `도서관 사서 보조`
- `inject_librarian_context` (`wrap_model_call`): 이름·역할을 모델 입력에 반영
- `library_policy_guardrail` (`before_agent`): 타인 대출·개인정보·업무 무관 요청을 모델 호출 전에 거절

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
