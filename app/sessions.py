import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class SessionState:
    """In-memory state for a single chat session."""

    history: list[BaseMessage] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=time.monotonic)
    in_use: int = 0


class SessionStore:
    """Process-local session store with LRU eviction.

    Sessions live in this process only, so the application must run with a
    single worker. The store lock only guards the dictionary; callers hold a
    session's own lock while processing a request.
    """

    def __init__(
        self, max_sessions: int = 1000, max_stored_messages: int = 50
    ) -> None:
        self.max_sessions = max_sessions
        self.max_stored_messages = max_stored_messages
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> SessionState:
        """Return the session state, creating it if it does not exist.

        The returned state is marked in use until the caller calls
        ``release`` so that LRU eviction cannot drop a session that has been
        fetched but has not yet acquired its per-session lock.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState()
                self._sessions[session_id] = state
            state.in_use += 1
            state.last_access = time.monotonic()
            self._sessions.move_to_end(session_id)
            self._evict_if_needed(current_session_id=session_id)
            return state

    async def release(self, state: SessionState) -> None:
        """Mark a state returned by ``get`` as no longer in use."""
        async with self._lock:
            if state.in_use > 0:
                state.in_use -= 1

    def append_turn(
        self, state: SessionState, user_text: str, assistant_text: str
    ) -> None:
        """Append one user/assistant turn and enforce the storage cap."""
        state.history.append(HumanMessage(content=user_text))
        state.history.append(AIMessage(content=assistant_text))
        if len(state.history) > self.max_stored_messages:
            del state.history[: -self.max_stored_messages]

    def _evict_if_needed(self, current_session_id: str) -> None:
        """Evict least-recently-used sessions that are not in use.

        A session is eviction-safe only when no caller holds a fetched
        reference (``in_use == 0``) and its lock is free. If every other
        session is in use or locked, the store is allowed to exceed
        ``max_sessions`` until a later call finds a candidate.
        """
        if len(self._sessions) <= self.max_sessions:
            return

        for session_id in list(self._sessions):
            if len(self._sessions) <= self.max_sessions:
                return
            if session_id == current_session_id:
                continue
            state = self._sessions[session_id]
            if state.in_use > 0 or state.lock.locked():
                continue
            del self._sessions[session_id]
