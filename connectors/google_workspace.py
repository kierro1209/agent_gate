from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]


@dataclass(frozen=True)
class SourceRecord:
    source: str
    id: str
    observed_at: str
    text: str


class GoogleWorkspace:
    def __init__(
        self,
        credentials_path: str | Path = "google_credentials.json",
        token_path: str | Path = "google_token.json",
    ) -> None:
        self._credentials_path = Path(credentials_path)
        self._token_path = Path(token_path)
        self._credentials: Any = None

    @property
    def configured(self) -> bool:
        return self._credentials_path.exists() or self._token_path.exists()

    def _authorize(self) -> Any:
        if self._credentials is not None:
            return self._credentials
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

        credentials = None
        if self._token_path.exists():
            credentials = Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
                str(self._token_path), SCOPES
            )
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not self._credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google OAuth client file not found: {self._credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_path), SCOPES
                )
                credentials = flow.run_local_server(port=0)
            self._token_path.write_text(credentials.to_json(), encoding="utf-8")
        self._credentials = credentials
        return credentials

    def calendar_events(self, max_results: int = 10) -> list[SourceRecord]:
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        now = datetime.now(timezone.utc)
        response = (
            build("calendar", "v3", credentials=self._authorize(), cache_discovery=False)
            .events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=(now + timedelta(days=7)).isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return map_calendar_events(response.get("items", []), now.isoformat())

    def gmail_messages(self, max_results: int = 10) -> list[SourceRecord]:
        from googleapiclient.discovery import build

        service = build("gmail", "v1", credentials=self._authorize(), cache_discovery=False)
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q="newer_than:30d",
                maxResults=max_results,
            )
            .execute()
        )
        messages = [
            service.users()
            .messages()
            .get(
                userId="me",
                id=item["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
            for item in response.get("messages", [])
        ]
        return map_gmail_messages(messages, datetime.now(timezone.utc).isoformat())


def map_calendar_events(items: list[dict[str, Any]], observed_at: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for item in items:
        start = item.get("start", {})
        when = start.get("dateTime") or start.get("date") or "unknown time"
        records.append(
            SourceRecord(
                "google-calendar",
                str(item.get("id", "")),
                observed_at,
                f"{when} — {item.get('summary', '(untitled event)')}",
            )
        )
    return records


def map_gmail_messages(items: list[dict[str, Any]], observed_at: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for item in items:
        headers = {
            str(header.get("name", "")).lower(): str(header.get("value", ""))
            for header in item.get("payload", {}).get("headers", [])
        }
        records.append(
            SourceRecord(
                "gmail",
                str(item.get("id", "")),
                observed_at,
                f"From: {headers.get('from', 'unknown')}; "
                f"Date: {headers.get('date', 'unknown')}; "
                f"Subject: {headers.get('subject', '(no subject)')}; "
                f"Snippet: {item.get('snippet', '')}",
            )
        )
    return records


def route_sources(workspace: GoogleWorkspace, message: str) -> list[SourceRecord]:
    if not workspace.configured:
        return []
    lowered = message.lower()
    records: list[SourceRecord] = []
    if any(word in lowered for word in ("calendar", "schedule", "appointment", "meeting")):
        records.extend(workspace.calendar_events())
    if any(word in lowered for word in ("email", "gmail", "inbox", "mail")):
        records.extend(workspace.gmail_messages())
    return records
