import asyncio
from collections import OrderedDict


class ConversationLocks:
    """Process-local locks keyed by conversation.

    Message history lives in MySQL; this registry only serializes turns that
    share a conversation. Locks that are currently held are never evicted.
    """

    def __init__(self, max_locks: int = 1000) -> None:
        self.max_locks = max_locks
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        self._locks.move_to_end(key)
        self._evict_if_needed()
        return lock

    def _evict_if_needed(self) -> None:
        if len(self._locks) <= self.max_locks:
            return
        for key in list(self._locks):
            if len(self._locks) <= self.max_locks:
                return
            if not self._locks[key].locked():
                del self._locks[key]
