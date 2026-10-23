from datetime import date
from typing import Any

from sqlalchemy.exc import IntegrityError


async def create_ticket_record(
    repository: Any,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Any:
    """Create a ticket with a unique number; do not change conversation status."""
    ticket = None
    for _ in range(5):
        ticket_no = await repository.next_ticket_no(date.today())
        try:
            ticket = await repository.create_ticket(
                conversation_id,
                description,
                ticket_type,
                ticket_no,
            )
            break
        except IntegrityError:
            continue

    if ticket is None:
        raise RuntimeError("failed to allocate a unique ticket number")
    return ticket
