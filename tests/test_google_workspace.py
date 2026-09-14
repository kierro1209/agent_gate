from typing import Any

from connectors.google_workspace import (
    SourceRecord,
    map_calendar_events,
    map_gmail_messages,
    route_sources,
)
from demo.chat import format_sources


def test_maps_calendar_event_with_provenance() -> None:
    records = map_calendar_events(
        [
            {
                "id": "event-1",
                "summary": "Dentist",
                "start": {"dateTime": "2026-09-14T14:30:00-07:00"},
            }
        ],
        "observed",
    )
    assert records == [
        SourceRecord(
            "google-calendar",
            "event-1",
            "observed",
            "2026-09-14T14:30:00-07:00 — Dentist",
        )
    ]


def test_maps_gmail_metadata_without_body() -> None:
    records = map_gmail_messages(
        [
            {
                "id": "message-1",
                "snippet": "Short preview",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "sender@example.com"},
                        {"name": "Subject", "value": "Status"},
                        {"name": "Date", "value": "Sunday"},
                    ],
                    "body": {"data": "must-not-be-used"},
                },
            }
        ],
        "observed",
    )
    assert "Short preview" in records[0].text
    assert "must-not-be-used" not in records[0].text


def test_routes_only_explicit_source_turns() -> None:
    class Workspace:
        configured = True

        def calendar_events(self) -> list[SourceRecord]:
            return [SourceRecord("calendar", "1", "now", "event")]

        def gmail_messages(self) -> list[SourceRecord]:
            return [SourceRecord("gmail", "2", "now", "message")]

    workspace: Any = Workspace()
    assert [item.source for item in route_sources(workspace, "check calendar and email")] == [
        "calendar",
        "gmail",
    ]
    assert route_sources(workspace, "explain CAP theorem") == []


def test_source_format_includes_provenance() -> None:
    output = format_sources([SourceRecord("gmail", "id-1", "now", "message")])
    assert "source=gmail id=id-1 observed_at=now" in output
