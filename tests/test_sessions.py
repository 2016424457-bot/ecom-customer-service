from app.sessions import SessionStore


async def test_sessions_are_isolated() -> None:
    store = SessionStore()
    a = await store.get("a")
    b = await store.get("b")

    store.append_turn(a, "hello", "hi there")

    assert [message.content for message in a.history] == ["hello", "hi there"]
    assert b.history == []

    await store.release(a)
    await store.release(b)


async def test_append_turn_caps_stored_messages() -> None:
    store = SessionStore(max_stored_messages=4)
    state = await store.get("a")

    store.append_turn(state, "u0", "a0")
    store.append_turn(state, "u1", "a1")
    store.append_turn(state, "u2", "a2")

    assert len(state.history) == 4
    assert state.history[-1].content == "a2"
    assert [message.content for message in state.history] == ["u1", "a1", "u2", "a2"]

    await store.release(state)


async def test_lru_eviction() -> None:
    store = SessionStore(max_sessions=2)
    a_state = await store.get("a")
    b_state = await store.get("b")
    await store.release(a_state)
    await store.release(b_state)
    store.append_turn(b_state, "old", "reply")

    a_state = await store.get("a")
    await store.release(a_state)
    c_state = await store.get("c")
    await store.release(c_state)

    revived_b = await store.get("b")

    assert revived_b is not b_state
    assert revived_b.history == []


async def test_active_session_not_evicted() -> None:
    store = SessionStore(max_sessions=2)
    b_state = await store.get("b")
    a_state = await store.get("a")
    await store.release(a_state)
    await store.release(b_state)

    async with b_state.lock:
        c_state = await store.get("c")
        await store.release(c_state)

    assert await store.get("b") is b_state


async def test_fetched_but_not_locked_session_not_evicted() -> None:
    store = SessionStore(max_sessions=2)
    a_state = await store.get("a")
    b_state = await store.get("b")
    c_state = await store.get("c")

    assert c_state is not a_state
    assert c_state is not b_state
    assert await store.get("a") is a_state
    assert await store.get("b") is b_state


async def test_release_allows_eviction() -> None:
    store = SessionStore(max_sessions=2)
    a_state = await store.get("a")
    b_state = await store.get("b")
    await store.release(a_state)
    await store.release(b_state)

    c_state = await store.get("c")
    await store.release(c_state)

    assert await store.get("a") is not a_state
