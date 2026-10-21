from app.sessions import ConversationLocks


def test_same_key_returns_same_lock() -> None:
    locks = ConversationLocks()

    assert locks.lock_for("a") is locks.lock_for("a")


def test_different_keys_return_different_locks() -> None:
    locks = ConversationLocks()

    assert locks.lock_for("a") is not locks.lock_for("b")


async def test_held_lock_is_not_evicted() -> None:
    locks = ConversationLocks(max_locks=1)
    first = locks.lock_for("a")

    async with first:
        locks.lock_for("b")
        assert locks.lock_for("a") is first


def test_unheld_locks_are_evicted_lru() -> None:
    locks = ConversationLocks(max_locks=2)
    first = locks.lock_for("a")
    second = locks.lock_for("b")
    locks.lock_for("a")

    locks.lock_for("c")

    assert locks.lock_for("a") is first
    assert locks.lock_for("b") is not second


def test_lock_registry_stays_within_limit() -> None:
    locks = ConversationLocks(max_locks=2)

    locks.lock_for("a")
    locks.lock_for("b")
    locks.lock_for("c")

    assert len(locks._locks) == 2
