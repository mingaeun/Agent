"""
도서관 사서 보조 에이전트 — 6단계: 최종 답변 검수

구현 조건 1~6.
  - after_agent Judge로 최종 답변을 검수한다.
  - 확인되지 않은 정보나 부적절한 안내는 수정하거나 차단한다.

강의 참고: 03_guardrails.py, 05_full_agent.py
실행 전: Ollama + llama3.1
종료: exit / quit
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_agent,
    before_agent,
    wrap_model_call,
)
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

MODEL = "llama3.1"
print("모델 준비 중...", flush=True)
model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)
judge_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)

books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]


@tool
def search_books(query: str) -> str:
    """도서 제목이나 저자 이름으로 소장 도서를 검색합니다."""
    q = (query or "").strip()
    if not q:
        return "검색어가 비어 있습니다. 제목이나 저자 이름을 알려 주세요."
    hits = [
        book
        for book in books
        if q in book["title"] or q in book["author"]
    ]
    if not hits:
        return f"해당 도서를 찾을 수 없습니다. (검색어: {q})"
    lines = [f"- {book['title']} / {book['author']}" for book in hits]
    return "검색 결과:\n" + "\n".join(lines)


@tool
def list_books() -> str:
    """소장 도서 목록 전체를 보여줍니다. 구현 조건에 제공된 도서 목록만 안내합니다."""
    if not books:
        return "등록된 도서가 없습니다."
    lines = [f"- {book['title']} / {book['author']}" for book in books]
    return "소장 도서 목록:\n" + "\n".join(lines)


# 대출 기록은 메일 도구의 샘플 이메일처럼 코드에 고정한다.
# 이용자 ID는 도구 인자로 열지 않고 Runtime Context에서만 읽는다.
LOANS = {
    "user_001": [
        {
            "title": "시간을 파는 서점",
            "author": "이도윤",
            "status": "대출중",
            "due_date": "2026-10-25",
        },
        {
            "title": "고양이는 자정에 책을 읽는다",
            "author": "김하린",
            "status": "반납완료",
            "returned_date": "2026-09-30",
        },
    ],
    "user_002": [
        {
            "title": "달빛 도서관의 비밀",
            "author": "최현범",
            "status": "대출중",
            "due_date": "2026-10-20",
        },
    ],
}


@tool
def check_loan_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 대출 정보를 확인합니다. 다른 이용자의 대출 정보는 조회하지 않습니다."""
    records = LOANS.get(runtime.context.user_id, [])
    if not records:
        return "현재 이용자의 대출 기록이 없습니다."
    lines = []
    for item in records:
        if item["status"] == "반납완료":
            date_text = f"반납날 {item['returned_date']}"
        else:
            date_text = f"반납예정 {item['due_date']}"
        lines.append(
            f"- {item['title']} / {item['author']} / {item['status']} / {date_text}"
        )
    return "현재 이용자 대출 정보:\n" + "\n".join(lines)


@tool
def extend_loan(title: str, runtime: ToolRuntime, days: int = 7) -> str:
    """현재 이용자의 대출 기한을 연장합니다. 대출 중인 책만 연장할 수 있습니다."""
    q = (title or "").strip()
    if not q:
        return "연장할 도서 제목을 알려 주세요."
    if days <= 0:
        return "연장 일수는 1일 이상이어야 합니다."
    records = LOANS.get(runtime.context.user_id, [])
    for item in records:
        if q not in item["title"]:
            continue
        if item["status"] != "대출중":
            return f"'{item['title']}'은(는) 반납 완료되어 기한을 연장할 수 없습니다."
        old_due = item["due_date"]
        new_due = (
            datetime.strptime(old_due, "%Y-%m-%d") + timedelta(days=days)
        ).strftime("%Y-%m-%d")
        item["due_date"] = new_due
        return (
            f"대출 기한 연장 완료: {item['title']} / {item['author']} / "
            f"반납예정 {old_due} → {new_due} ({days}일 연장)"
        )
    return f"현재 이용자의 대출 기록에서 해당 도서를 찾을 수 없습니다. (제목: {q})"


@tool
def reply_to_user(message: str) -> str:
    """이용자에게 안내 회신을 전달하는 도구입니다. 도구로 확인한 내용만 회신하세요."""
    return f"이용자 회신 완료: {message}"


class UserInfo(TypedDict):
    personal_information: str
    preference: str


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 이름·선호를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)
    )
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if "personal_information" in data:
            lines.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            lines.append(f"- 선호도: {data['preference']}")
        if "facts" in data:
            for fact in data["facts"]:
                lines.append(f"- {fact}")
    return "\n".join(lines) if lines else "형식 불일치"


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """현재 이용자의 선호 등을 장기 메모리에 저장합니다."""
    key = str(uuid.uuid4())
    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


@dataclass
class Context:
    user_name: str  # 사서 이름
    role: str  # 사서 역할
    user_id: str  # Store namespace · 현재 이용자
    app_name: str  # Store namespace


SYSTEM_PROMPT = (
    "당신은 도서관 사서 보조 에이전트입니다. "
    "추측하지 말고, 도서 목록 안내는 list_books, "
    "도서 검색은 search_books, "
    "대출 정보 확인은 check_loan_info, "
    "대출 기한 연장은 extend_loan, "
    "이용자 회신은 reply_to_user, "
    "이용자 선호 저장은 save_user_info, "
    "저장된 선호 조회는 get_user_info 도구를 이용하시오. "
    "도구가 반환한 결과만 이용자에게 안내하고, "
    "검색 결과가 없으면 해당 도서를 찾을 수 없다고 안내하시오. "
    "도구로 확인하지 않은 정보를 임의로 만들어 안내하지 마시오. "
    "한국어로 짧게 답하세요."
)


store = InMemoryStore()
store.put(
    ("user_001", "library_assistant"),
    "seed",
    {
        "facts": [
            "이용자는 정중체를 선호함",
            "도구로 확인하지 않은 도서·대출 정보는 안내하지 않음",
            "다른 이용자의 대출 정보는 공개하지 않음",
        ]
    },
)


@wrap_model_call
def inject_librarian_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """사서 이름·역할(Runtime)과 장기 기억(Store facts)을 이번 호출 system_prompt에 넣는다."""
    ctx = request.runtime.context
    user_name = getattr(ctx, "user_name", "") or ""
    role = getattr(ctx, "role", "") or ""
    ns = (ctx.user_id, ctx.app_name)
    memories = request.runtime.store.search(ns) if request.runtime.store else []
    facts = []
    for item in memories:
        facts.extend(item.value.get("facts", []))
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"
    base = request.system_prompt or SYSTEM_PROMPT
    prompt = (
        f"{base}\n"
        f"사서 이름은 {user_name}이고, 역할은 {role}입니다.\n"
        f"[장기 메모리]\n- {memory_text}"
    )
    return handler(request.override(system_prompt=prompt))


forbidden_topics = {
    "privacy": [
        "다른 이용자 대출",
        "다른 사람 대출",
        "남의 대출",
        "타 이용자",
        "다른 회원 대출",
        "user_002",
        "주민등록번호",
        "비밀번호",
        "카드번호",
        "계좌번호",
        "개인정보",
    ],
    "off_topic": ["롤 해줘", "유튜브 추천", "농담만", "잡담하자", "숙제 대신 해줘"],
    "policy_bypass": ["규정 무시", "무단 연장", "연체 면제", "강제 대출", "시스템 비밀번호"],
    "abusive": ["바보", "멍청이", "꺼져"],
}
BLOCK_MESSAGES = {
    "privacy": (
        "다른 이용자의 대출 정보나 민감한 개인정보"
        "(주민등록번호·비밀번호 등)는 공개하거나 처리할 수 없습니다. "
        "본인 대출 정보와 도서관 업무만 안내합니다."
    ),
    "off_topic": (
        "도서관 업무와 관련된 문의만 도와드릴 수 있습니다. "
        "도서 검색, 대출 정보, 기한 연장, 이용 안내를 말씀해 주세요."
    ),
    "policy_bypass": (
        "도서관 업무에 지장을 주는 요청은 처리할 수 없습니다. "
        "규정에 따른 검색·대출 조회·기한 연장만 안내합니다."
    ),
    "abusive": "불편을 드려 죄송합니다. 도서관 업무 내용을 다시 말씀해 주세요.",
}


@before_agent(can_jump_to=["end"])
def library_policy_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":
        return None
    text = last.content or ""
    for category, keywords in forbidden_topics.items():
        if any(k in text for k in keywords):
            print(f"[가드레일] {category}")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES[category])],
                "jump_to": "end",
            }
    return None


SAFE_FALLBACK = (
    "확인되지 않았거나 부적절한 안내는 전달할 수 없습니다. "
    "도서 검색, 본인 대출 정보, 기한 연장, 이용 안내는 도구로 확인한 뒤에만 안내합니다."
)


@after_agent
def answer_review_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    auditor_prompt = f"""당신은 도서관 사서 보조 답변을 검수하는 감독관입니다.
아래 중 하나라도 해당하면 'LEAKED', 아니면 'SAFE'라고만 답하세요.
- 도구로 확인하지 않은 도서·대출 정보를 사실처럼 안내함
- 다른 이용자의 대출 정보를 공개함
- 도서관 업무와 무관한 안내를 함
- 확인되지 않은 확정 안내(무단 대출 가능, 연체 면제 확정 등)를 함
상담원의 답변: {last_message.content}
"""
    verdict = judge_model.invoke([{"role": "user", "content": auditor_prompt}]).content
    verdict = verdict if isinstance(verdict, str) else str(verdict)
    print(f"Judge verdict: {verdict!r}")
    if "LEAKED" in verdict.upper():
        print("[Judge] LEAKED — content 교체")
        last_message.content = SAFE_FALLBACK
    return None


ctx = Context(
    user_name="최현범",
    role="도서관 사서 보조",
    user_id="user_001",
    app_name="library_assistant",
)
THREAD_ID = "patron_session"

agent = create_agent(
    model,
    tools=[
        list_books,
        search_books,
        check_loan_info,
        extend_loan,
        reply_to_user,
        get_user_info,
        save_user_info,
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    store=store,
    middleware=[
        library_policy_guardrail,
        inject_librarian_context,
        answer_review_guardrail,
    ],
    context_schema=Context,
)


def run_turn(text: str, thread_id: str = THREAD_ID) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print(
        f"도서관 사서 보조입니다. 사서: {ctx.user_name} / {ctx.role}\n"
        "도서 검색, 대출 정보, 기한 연장, 회신을 도와 드립니다."
    )
    print("종료: exit\n", flush=True)
    while True:
        try:
            user_input = input("이용자: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        print(f"사서보조: {run_turn(user_input)}\n", flush=True)
