"""
SessionStore - S3-backed session management for multi-turn conversations.

Provides persistent session storage using Amazon S3 with optimistic locking
via ETags to handle concurrent writes safely. Supports automatic turn history
truncation and structured error handling.

Architecture:
    - Load: Fetch session from S3 (returns empty list if not found)
    - Save: Write entire session to S3 with ETag-based optimistic locking
    - Append: Load, append turn, truncate to 6 turns, save
    - Delete: Remove session from S3 (GDPR compliance)

Error Handling:
    - Transient errors (503, timeouts) → Exponential backoff retry
    - Permanent errors (403, 404 bucket) → Fail fast
    - Concurrency conflicts (412 ETag mismatch) → Reload and retry once

Usage:
    >>> store = SessionStore(bucket="sessions", prefix="prod/")
    >>> session = store.load("session-uuid")
    >>> store.append("session-uuid", turn_dict)
    >>> store.delete("session-uuid")
"""

import json
import time
from datetime import datetime
from typing import Any, Final

import boto3
from botocore.exceptions import ClientError
from loguru import logger
from typing_extensions import TypedDict

# ============================================================================
# Type Definitions
# ============================================================================


class Turn(TypedDict, total=False):
    """
    Turn schema for session storage.

    Aligned with QueryResult structure but excludes data field to prevent
    storing PII or large result sets in session history.

    Fields:
        content: User's original query text (required)
        sql: Generated SQL statement (optional)
        success: Whether turn succeeded (required)
        row_count: Number of rows returned (optional)
        cache_hit: Whether semantic cache was used (required)
        visualization: Plotly chart schema for replay (optional)
        timestamp: ISO 8601 UTC timestamp (required)
        duration_ms: Total processing time in milliseconds (optional)
        error: Error message if turn failed (optional)
    """

    content: str
    sql: str | None
    success: bool
    row_count: int | None
    cache_hit: bool
    visualization: dict[str, Any] | None
    timestamp: str
    duration_ms: int | None
    error: str | None


class SessionBundle(TypedDict):
    """
    Complete session state stored in S3.

    Fields:
        session_id: UUIDv4 session identifier
        turns: List of turn dictionaries (max 6)
        updated_at: ISO 8601 UTC timestamp of last update
        metadata: Optional metadata dictionary
    """

    session_id: str
    turns: list[Turn]
    updated_at: str
    metadata: dict[str, Any]


# ============================================================================
# Constants
# ============================================================================

MAX_TURNS: Final[int] = 6
MAX_RETRIES: Final[int] = 3
INITIAL_RETRY_DELAY_MS: Final[int] = 100

# Transient error codes that should be retried
TRANSIENT_ERROR_CODES: Final[set[str]] = {
    "ServiceUnavailable",
    "RequestTimeout",
    "RequestTimeoutException",
    "ThrottlingException",
    "SlowDown",
}


# ============================================================================
# Helper Functions
# ============================================================================


def query_result_to_turn(result: dict[str, Any]) -> Turn:
    """
    Convert QueryResult to Turn schema for session storage.

    NOTE: Excludes 'data' field to prevent storing PII or large result sets.
    Only stores metadata required for conversation context.

    Args:
        result: QueryResult dictionary from QueryProcessor

    Returns:
        Turn dictionary suitable for session storage

    """
    return Turn(
        content=result["query"],
        sql=result.get("sql"),
        success=result["success"],
        row_count=result.get("row_count"),
        cache_hit=result.get("cache_hit", False),
        visualization=result.get("visualization"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        duration_ms=result.get("total_duration_ms"),
        error=result.get("error"),
    )


# ============================================================================
# SessionStore Class
# ============================================================================


class SessionStore:
    """
    S3-backed session storage with optimistic locking.

    Responsibilities:
        - Load/save session state from/to S3
        - Manage turn history with automatic truncation
        - Handle concurrency with ETag-based optimistic locking
        - Retry transient errors with exponential backoff
        - Distinguish transient vs permanent errors

    Does NOT Handle:
        - Session ID generation (caller's responsibility)
        - Query processing (delegates to QueryProcessor)
        - Authentication/authorization (relies on IAM)
    """

    __slots__ = ("bucket", "prefix", "_s3_client", "_etag_cache")

    def __init__(self, bucket: str, prefix: str = "sessions/"):
        """
        Initialize SessionStore with S3 configuration.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (default: "sessions/")

        Raises:
            ValueError: If bucket is empty

        """
        if not bucket or not bucket.strip():
            raise ValueError("S3 bucket name cannot be empty")

        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._s3_client = boto3.client("s3")
        self._etag_cache: dict[str, str] = {}

        logger.info(f"✅ SessionStore initialized (bucket={bucket}, prefix={prefix})")

    def load(self, session_id: str) -> SessionBundle:
        """
        Load session from S3.

        NOTE: Returns empty session if not found (404) - not an error condition.

        Args:
            session_id: UUIDv4 session identifier

        Returns:
            SessionBundle with turns list (empty if session not found)

        Raises:
            RuntimeError: If S3 operation fails after retries

        """
        key = self._build_key(session_id)

        logger.info(
            "session.load.start",
            extra={"session_id": session_id, "s3_key": f"s3://{self.bucket}/{key}"},
        )

        start_time = time.time()

        try:
            response = self._get_object_with_retry(key)

            if response is None:
                # Session not found - return empty session
                logger.info(
                    "session.load.miss",
                    extra={
                        "session_id": session_id,
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                )
                return SessionBundle(
                    session_id=session_id,
                    turns=[],
                    updated_at=datetime.utcnow().isoformat() + "Z",
                    metadata={},
                )

            # Parse session data
            body = response["Body"].read().decode("utf-8")
            session_data = json.loads(body)

            # Cache ETag for optimistic locking
            etag = response.get("ETag", "").strip('"')
            self._etag_cache[session_id] = etag

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "session.load.success",
                extra={
                    "session_id": session_id,
                    "turn_count": len(session_data.get("turns", [])),
                    "duration_ms": duration_ms,
                    "etag": etag,
                },
            )

            return SessionBundle(
                session_id=session_data.get("session_id", session_id),
                turns=session_data.get("turns", []),
                updated_at=session_data.get(
                    "updated_at", datetime.utcnow().isoformat() + "Z"
                ),
                metadata=session_data.get("metadata", {}),
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "session.load.fail",
                extra={
                    "session_id": session_id,
                    "error": str(e),
                    "duration_ms": duration_ms,
                },
            )
            raise RuntimeError(f"Failed to load session {session_id}: {e}") from e

    def save(self, session_id: str, turns: list[Turn]) -> None:
        """
        Save session to S3 with optimistic locking.

        Uses cached ETag from previous load to detect concurrent modifications.
        Automatically truncates to MAX_TURNS if needed.

        Args:
            session_id: UUIDv4 session identifier
            turns: List of turn dictionaries

        Raises:
            RuntimeError: If save fails after retries

        """
        # Truncate to max turns
        if len(turns) > MAX_TURNS:
            logger.warning(
                f"Truncating session {session_id} from {len(turns)} to {MAX_TURNS} turns"
            )
            turns = turns[-MAX_TURNS:]

        key = self._build_key(session_id)

        session_data = SessionBundle(
            session_id=session_id,
            turns=turns,
            updated_at=datetime.utcnow().isoformat() + "Z",
            metadata={},
        )

        body = json.dumps(session_data, indent=2, ensure_ascii=False)

        logger.info(
            "session.save.start",
            extra={
                "session_id": session_id,
                "turn_count": len(turns),
                "s3_key": f"s3://{self.bucket}/{key}",
            },
        )

        start_time = time.time()

        try:
            # Get cached ETag for optimistic locking
            etag = self._etag_cache.get(session_id)

            # Put object with conditional write if ETag available
            put_kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": key,
                "Body": body.encode("utf-8"),
                "ContentType": "application/json",
                "ServerSideEncryption": "AES256",
            }

            if etag:
                put_kwargs["IfMatch"] = etag

            response = self._put_object_with_retry(put_kwargs)

            # Update cached ETag
            new_etag = response.get("ETag", "").strip('"')
            self._etag_cache[session_id] = new_etag

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "session.save.success",
                extra={
                    "session_id": session_id,
                    "turn_count": len(turns),
                    "duration_ms": duration_ms,
                    "etag": new_etag,
                },
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "PreconditionFailed":
                # ETag mismatch - concurrent modification detected
                logger.warning(
                    "session.save.concurrency_conflict",
                    extra={
                        "session_id": session_id,
                        "error_code": error_code,
                        "expected_etag": etag,
                    },
                )

                # Reload session and retry once
                logger.info(f"Reloading session {session_id} after ETag conflict")
                current_session = self.load(session_id)

                # Merge new turn into reloaded session
                merged_turns = current_session["turns"] + turns
                merged_turns = merged_turns[-MAX_TURNS:]  # Truncate

                # Retry save with new ETag
                self.save(session_id, merged_turns)
            else:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error(
                    "session.save.fail",
                    extra={
                        "session_id": session_id,
                        "error_code": error_code,
                        "error": str(e),
                        "duration_ms": duration_ms,
                    },
                )
                raise RuntimeError(f"Failed to save session {session_id}: {e}") from e

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "session.save.fail",
                extra={
                    "session_id": session_id,
                    "error": str(e),
                    "duration_ms": duration_ms,
                },
            )
            raise RuntimeError(f"Failed to save session {session_id}: {e}") from e

    def append(self, session_id: str, turn: Turn) -> None:
        """
        Append turn to session and save.

        Convenience method that loads, appends, truncates, and saves.

        Args:
            session_id: UUIDv4 session identifier
            turn: Turn dictionary to append

        Raises:
            RuntimeError: If load or save fails

        """
        session = self.load(session_id)
        turns = session["turns"]
        turns.append(turn)

        # Truncate to max turns before saving
        turns = turns[-MAX_TURNS:]

        self.save(session_id, turns)

    def delete(self, session_id: str) -> None:
        """
        Delete session from S3 (GDPR compliance).

        Args:
            session_id: UUIDv4 session identifier

        Raises:
            RuntimeError: If deletion fails

        """
        key = self._build_key(session_id)

        logger.info(
            "session.delete.start",
            extra={"session_id": session_id, "s3_key": f"s3://{self.bucket}/{key}"},
        )

        try:
            self._s3_client.delete_object(Bucket=self.bucket, Key=key)

            # Remove from ETag cache
            self._etag_cache.pop(session_id, None)

            logger.info("session.delete.success", extra={"session_id": session_id})

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                "session.delete.fail",
                extra={
                    "session_id": session_id,
                    "error_code": error_code,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete session {session_id}: {e}") from e

    def _build_key(self, session_id: str) -> str:
        """Build S3 key for session."""
        return f"{self.prefix}session_{session_id}/session.json"

    def _get_object_with_retry(self, key: str) -> dict[str, Any] | None:
        """
        Get object from S3 with exponential backoff retry.

        Returns None if object not found (404), raises on other errors.

        Args:
            key: S3 object key

        Returns:
            S3 GetObject response or None if not found

        Raises:
            ClientError: If operation fails after retries

        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                return self._s3_client.get_object(Bucket=self.bucket, Key=key)

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")

                # 404 is not an error - return None for missing sessions
                if error_code == "NoSuchKey":
                    return None

                # Check if error is transient
                if error_code in TRANSIENT_ERROR_CODES:
                    last_error = e
                    delay_ms = INITIAL_RETRY_DELAY_MS * (2**attempt)

                    logger.warning(
                        "session.load.retry",
                        extra={
                            "error_code": error_code,
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "delay_ms": delay_ms,
                        },
                    )

                    time.sleep(delay_ms / 1000.0)
                    continue
                else:
                    # Permanent error - fail fast
                    raise

        # Max retries exceeded
        raise last_error if last_error else RuntimeError("Max retries exceeded")

    def _put_object_with_retry(self, put_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Put object to S3 with exponential backoff retry.

        Args:
            put_kwargs: Arguments for put_object call

        Returns:
            S3 PutObject response

        Raises:
            ClientError: If operation fails after retries

        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                return self._s3_client.put_object(**put_kwargs)

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")

                # Don't retry ETag conflicts - caller handles this
                if error_code == "PreconditionFailed":
                    raise

                # Check if error is transient
                if error_code in TRANSIENT_ERROR_CODES:
                    last_error = e
                    delay_ms = INITIAL_RETRY_DELAY_MS * (2**attempt)

                    logger.warning(
                        "session.save.retry",
                        extra={
                            "error_code": error_code,
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "delay_ms": delay_ms,
                        },
                    )

                    time.sleep(delay_ms / 1000.0)
                    continue
                else:
                    # Permanent error - fail fast
                    raise

        # Max retries exceeded
        raise last_error if last_error else RuntimeError("Max retries exceeded")
