from app.services.chat import actions_after_handoff


def test_actions_untouched_before_handoff():
    actions = [{"type": "transfer_human"}]
    assert actions_after_handoff(actions, False) == actions


def test_handoff_keeps_existing_create_ticket():
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket", "draft": {"description": "x"}},
    ]
    assert actions_after_handoff(actions, True) == [
        {"type": "create_ticket", "draft": {"description": "x"}}
    ]


def test_handoff_adds_create_ticket_when_only_transfer():
    assert actions_after_handoff(
        [{"type": "transfer_human"}], True, "你好"
    ) == [
        {
            "type": "create_ticket",
            "draft": {"description": "你好", "ticket_type": "咨询"},
        }
    ]


def test_handoff_adds_create_ticket_when_no_actions():
    assert actions_after_handoff([], True, "在吗") == [
        {
            "type": "create_ticket",
            "draft": {"description": "在吗", "ticket_type": "咨询"},
        }
    ]


def test_handoff_keeps_other_actions():
    actions = [
        {"type": "transfer_human"},
        {"type": "refund_form", "order_id": "1001"},
    ]
    assert actions_after_handoff(actions, True) == [
        {"type": "refund_form", "order_id": "1001"},
        {
            "type": "create_ticket",
            "draft": {"description": "", "ticket_type": "咨询"},
        },
    ]