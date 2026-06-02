"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: runner_utils.py                                                       │
│ Developed by: Davidson Gomes                                                 │
│ Creation date: May 17, 2025                                                  │
│ Contact: contato@evolution-api.com                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ @copyright © Evolution API 2025. All rights reserved.                        │
│ Licensed under the Apache License, Version 2.0                               │
│                                                                              │
│ You may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│    http://www.apache.org/licenses/LICENSE-2.0                                │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ @important                                                                   │
│ For any future changes to the code in this file, it is recommended to        │
│ include, together with the modification, the information of the developer    │
│ who changed it and the date of modification.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part, Blob
from google.adk.sessions import DatabaseSessionService
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events import Event, EventActions
import time
from src.utils.logger import setup_logger
from src.core.exceptions import AgentNotFoundError, LLMRateLimitError
from src.services.agent_service import get_agent
from src.services.adk.agent_builder import AgentBuilder
from src.utils.adk_utils import extract_state_params
from sqlalchemy.orm import Session
from typing import Optional, List, Tuple, Dict, Any, Union
import asyncio
import base64
import json
import uuid
from src.services.temp_limits_service import check_session_limit
from src.services.session_service import SessionLimitExceeded
from datetime import datetime
from fastapi import HTTPException

logger = setup_logger(__name__)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 2.0


def is_rate_limit_error(exception: Exception) -> bool:
    """Detect rate limit errors from LLM providers (OpenRouter, LiteLLM, etc.)."""
    error_str = str(exception).lower()
    rate_limit_markers = [
        "rate_limit",
        "rate limit",
        "ratelimiterror",
        "429",
        "rate-limited",
        "temporarily rate-limited",
        "provider returned error",
    ]
    return any(marker in error_str for marker in rate_limit_markers)


def is_file_unsupported_error(exception: Exception) -> bool:
    """Detect file/image type not supported errors from LLM providers.

    Triggered when a model (e.g. Xiaomi via OpenRouter) rejects inline_data
    (images, PDFs, etc.) because it does not support multimodal content.
    """
    error_str = str(exception)
    markers = [
        "file type is not supported",
        "unsupported file type",
        "image type not supported",
        "file format not supported",
    ]
    return any(marker in error_str.lower() for marker in markers)


def is_llm_response_parse_error(exception: Exception) -> bool:
    """Detect LiteLLM response parsing/deserialization errors.

    Triggered when the provider returns a valid HTTP 200 but with a response
    structure that the current LiteLLM version cannot deserialize (e.g. a new
    annotation type like 'file' that isn't yet in the Pydantic schema).
    These errors are NOT retryable — the same request will fail every time.
    """
    error_str = str(exception)
    markers = [
        "invalid response object",
        "validationerror",
        "input should be 'url_citation'",
        "annotations.0.type",
        "convert_to_model_response_object",
        "convert_dict_to_response",
        # LiteLLM tries to access e.request when wrapping a plain exception
        "has no attribute 'request'",
    ]
    return any(marker in error_str.lower() for marker in markers)


async def execute_with_retry(
    run_async_fn,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    initial_delay: float = INITIAL_RETRY_DELAY_SECONDS,
    log_context: str = "",
) -> Any:
    """Execute an async generator with retry on rate limit errors.

    Yields events from the generator. On rate limit errors, retries with
    exponential backoff up to max_attempts times. Other exceptions propagate.

    Args:
        run_async_fn: Async callable that returns an async generator of events.
        max_attempts: Maximum number of execution attempts.
        initial_delay: Initial backoff delay in seconds (doubles each attempt).
        log_context: Optional context string for log messages.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            async for event in run_async_fn():
                yield event, None
            return
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and attempt < max_attempts:
                delay = initial_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{log_context}] Rate limit detected on attempt {attempt}/{max_attempts}. "
                    f"Retrying in {delay:.1f}s... Error: {str(e)[:200]}"
                )
                await asyncio.sleep(delay)
                continue
            raise

    if last_error:
        raise LLMRateLimitError(
            f"[{log_context}] LLM rate limit exceeded after {max_attempts} attempts: {str(last_error)[:300]}"
        ) from last_error


def convert_sets(obj):
    """Convert sets to lists for JSON serialization."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(i) for i in obj]
    else:
        return obj


class RunnerUtils:
    """Utility class for common runner operations."""

    def __init__(self, db: Session):
        self.db = db
        self.agent_builder = AgentBuilder(db)

    async def get_and_build_agent(self, agent_id: str):
        """Get agent from database and build it."""
        from src.services.agent_service import get_agent, get_agent_integrations
        
        get_root_agent = await get_agent(self.db, agent_id)
        if get_root_agent is None:
            raise AgentNotFoundError(f"Agent with ID {agent_id} not found")

        # Load integrations directly from database
        integrations = await get_agent_integrations(self.db, agent_id)
        
        # Attach integrations to agent object for use in builder
        get_root_agent._integrations = integrations

        root_agent, state_params = await self.agent_builder.build_agent(get_root_agent)
        logger.debug(f"State params: {state_params}")

        return root_agent, state_params

    def create_session_id(
        self, external_id: str, agent_id: str, session_id: Optional[str] = None
    ) -> str:
        """Create or use provided session ID."""
        if session_id:
            return session_id
        else:
            return f"{external_id}_{agent_id}"

    def create_runner(
        self,
        agent,
        agent_id: str,
        session_service: DatabaseSessionService,
        artifacts_service: InMemoryArtifactService,
        memory_service: Optional[BaseMemoryService] = None,
        memory_runner: bool = False,
    ) -> Runner:
        """Create and configure Runner."""
        if memory_runner:
            return InMemoryRunner(
                app_name=agent_id,
                agent=agent,
                # session_service=session_service,
                # artifacts_service=artifacts_service,
                # memory_service=memory_service,
            )
        else:
            return Runner(
                agent=agent,
                app_name=agent_id,
                session_service=session_service,
                artifact_service=artifacts_service,
                memory_service=memory_service,
            )

    async def get_or_create_session(
        self,
        session_service: DatabaseSessionService,
        agent_id: str,
        external_id: str,
        adk_session_id: str,
    ):
        """Get existing session or create new one."""
        # First, try to get session by ID directly from database (more reliable)
        # This ensures we find sessions created via the API endpoint
        try:
            from src.services.session_service import get_session_by_id
            # Pass self.db to use the same database session
            session = await get_session_by_id(session_service, adk_session_id, db=self.db)
            if session:
                logger.debug(f"Found existing session {adk_session_id} in database")
                return session
        except HTTPException as e:
            # If session not found (404), continue to standard method
            if e.status_code == 404:
                logger.debug(f"Session {adk_session_id} not found in database, trying standard method")
            else:
                # Re-raise other HTTP exceptions
                raise
        except Exception as e:
            logger.debug(f"Could not get session by ID {adk_session_id}: {str(e)}, trying standard method")
        
        # Fallback to standard method
        session = await session_service.get_session(
            app_name=agent_id,
            user_id=external_id,
            session_id=adk_session_id,
        )

        if session is None:
            # Check session limits before creating new session
            await self._check_session_limits(external_id)

            session = await session_service.create_session(
                app_name=agent_id,
                user_id=external_id,
                session_id=adk_session_id,
            )
            logger.info(f"Created new session {adk_session_id} for agent {agent_id} and user {external_id}")

        return session

    async def _check_session_limits(self, user_id: str) -> None:
        """Check session limits before creating new sessions.

        Args:
            user_id: The user ID to check limits for

        Raises:
            SessionLimitExceeded: If session limit is exceeded
        """
        # Check session count limit
        allowed, message = check_session_limit(self.db)
        if not allowed:
            logger.warning(f"Session limit exceeded for user {user_id}: {message}")
            raise SessionLimitExceeded(message)

        logger.debug(f"Session limits check passed for user {user_id}")

    async def setup_session_state(
        self,
        session_service: DatabaseSessionService,
        session,
        message: str,
        state_params: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Setup session state with user input, state params and metadata."""
        # Store user input
        state_changes = {"user_input": message}
        actions_with_update = EventActions(state_delta=state_changes)
        user_input_event = Event(
            invocation_id=f"user_input_{int(time.time())}",
            author="system",
            actions=actions_with_update,
            timestamp=time.time(),
        )

        await session_service.append_event(session, user_input_event)
        logger.debug(f"Stored user input in session state via ADK: {message}")

        # Setup state parameters
        if state_params:
            for param in state_params:
                state_changes = {f"{param}": ""}
                actions_with_update = EventActions(state_delta=state_changes)
                new_param_event = Event(
                    invocation_id=f"param_{param}_{int(time.time())}",
                    author="system",
                    actions=actions_with_update,
                    timestamp=time.time(),
                )
                await session_service.append_event(session, new_param_event)
                logger.debug(f"Stored {param} in session state via ADK")

        # Store current time
        state_changes = {"_datetime": datetime.now().isoformat()}
        actions_with_update = EventActions(state_delta=state_changes)
        current_time_event = Event(
            invocation_id=f"current_time_{int(time.time())}",
            author="system",
            actions=actions_with_update,
            timestamp=time.time(),
        )
        await session_service.append_event(session, current_time_event)
        logger.debug(f"Stored current time in session state via ADK: {time.time()}")

        # Setup metadata
        if metadata:
            logger.info(f"[RunnerUtils] Setting up metadata with {len(metadata)} keys: {list(metadata.keys())}")
            for key, value in metadata.items():
                # Log contact data specifically
                if key == "contact" and isinstance(value, dict):
                    logger.info(f"[RunnerUtils] Storing contact data: name={value.get('name', 'N/A')}, id={value.get('id', 'N/A')}")
                elif key == "evoai_crm_data" and isinstance(value, dict) and "contact" in value:
                    contact_in_data = value.get("contact", {})
                    logger.info(f"[RunnerUtils] Storing evoai_crm_data with contact: name={contact_in_data.get('name', 'N/A') if isinstance(contact_in_data, dict) else 'N/A'}")
                
                state_changes = {f"{key}": value}
                actions_with_update = EventActions(state_delta=state_changes)
                new_metadata_event = Event(
                    invocation_id=f"metadata_{key}_{int(time.time())}",
                    author="system",
                    actions=actions_with_update,
                    timestamp=time.time(),
                )
                await session_service.append_event(session, new_metadata_event)
                logger.info(f"[RunnerUtils] ✅ Stored metadata '{key}' in session state via ADK")

    async def process_files(
        self,
        files: Optional[List],
        artifacts_service: InMemoryArtifactService,
        agent_id: str,
        external_id: str,
        adk_session_id: str,
    ) -> Tuple[List[Part], List[str]]:
        """Process uploaded files and return file parts and transcribed audio texts.

        Validates file size (max {} bytes) and base64 decoding. Files exceeding
        the limit are skipped with a warning log.
        """.format(MAX_FILE_SIZE_BYTES)
        file_parts = []
        transcribed_texts = []

        if files and len(files) > 0:
            for file_data in files:
                try:
                    filename = getattr(file_data, "filename", None) or str(file_data.get("filename", "unknown"))
                    content_type = getattr(file_data, "content_type", None) or str(file_data.get("content_type", ""))
                    data = getattr(file_data, "data", None) or file_data.get("data", "")

                    # Check if file is audio
                    is_audio = self._is_audio_file(content_type, filename)

                    logger.info(
                        f"Processing file: {filename} (type: {content_type}, is_audio: {is_audio})"
                    )

                    # Deduplicate: if this file was already attached in a previous turn
                    # of the current session, skip re-attaching the inline_data.
                    # Some providers (e.g. Amazon Bedrock) reject conversations that
                    # contain the same document name in multiple messages.
                    try:
                        existing_keys = await artifacts_service.list_artifact_keys(
                            app_name=agent_id,
                            user_id=external_id,
                            session_id=adk_session_id,
                        )
                        if filename in existing_keys:
                            logger.info(
                                f"File {filename} already exists in session artifacts. "
                                "Skipping re-attachment to avoid duplicate document errors."
                            )
                            file_parts.append(
                                Part(
                                    text=f"[O arquivo '{filename}' já foi enviado nesta conversa e está disponível no histórico da sessão.]"
                                )
                            )
                            continue
                    except Exception as _dedup_err:
                        logger.warning(
                            f"Could not check existing artifacts for deduplication ({filename}): {_dedup_err}"
                        )

                    # Validate base64 data
                    if not data:
                        logger.warning(f"Skipping file {filename}: empty data")
                        continue

                    file_bytes = base64.b64decode(data)

                    # Validate file size
                    file_size = len(file_bytes)
                    if file_size > MAX_FILE_SIZE_BYTES:
                        logger.warning(
                            f"File {filename} exceeds size limit ({file_size} > {MAX_FILE_SIZE_BYTES} bytes). "
                            f"Skipping file attachment."
                        )
                        continue

                    if file_size == 0:
                        logger.warning(f"Skipping file {filename}: decoded to 0 bytes")
                        continue

                    file_part = Part(
                        inline_data=Blob(
                            mime_type=content_type,
                            data=file_bytes,
                        )
                    )

                    # Always save to artifacts for reference
                    try:
                        await artifacts_service.save_artifact(
                            app_name=agent_id,
                            user_id=external_id,
                            session_id=adk_session_id,
                            filename=filename,
                            artifact=file_part,
                        )
                    except Exception as artifact_error:
                        logger.warning(
                            f"Could not save artifact for file {filename}: {artifact_error}"
                        )

                    # Add file to content parts for LLM processing
                    # Audio files: LLM can transcribe
                    # Image/PDF files: multimodal LLMs can process
                    file_parts.append(file_part)

                    # For non-audio files (PDF/images), add a text hint so the LLM
                    # knows it can call parse_fatura_energia without passing 'fonte'.
                    # This is needed because the model cannot re-encode inline_data
                    # back to a base64 string to pass as a tool argument.
                    if not is_audio:
                        file_parts.append(
                            Part(
                                text=(
                                    f"[ARQUIVO_RECEBIDO: nome='{filename}', tipo='{content_type}']\n"
                                    f"O arquivo '{filename}' foi salvo na sess\u00e3o. "
                                    f"Para analisar esta fatura, chame `parse_fatura_energia()` "
                                    f"SEM fornecer o par\u00e2metro 'fonte' \u2014 a ferramenta carrega\r\u00e1 "
                                    f"o arquivo automaticamente da sess\u00e3o."
                                )
                            )
                        )

                    logger.info(
                        f"Added file {filename} (type: {content_type}, size: {file_size} bytes)"
                        f" to content parts for LLM processing"
                    )

                except base64.binascii.Error as e:
                    logger.error(
                        f"Base64 decode error for file {getattr(file_data, 'filename', 'unknown')}: {str(e)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing file {getattr(file_data, 'filename', 'unknown')}: {str(e)}"
                    )

        return file_parts, transcribed_texts

    def _is_audio_file(self, content_type: str, filename: str) -> bool:
        """Check if file is an audio file based on content type and extension."""
        if not content_type and not filename:
            return False

        # Check MIME type (handle types with parameters like "audio/webm;codecs=opus")
        if content_type:
            # Extract main MIME type (before semicolon if present)
            main_mime_type = content_type.split(";")[0].strip().lower()

            audio_mime_types = [
                "audio/mpeg",  # MP3
                "audio/mp3",
                "audio/wav",  # WAV
                "audio/wave",
                "audio/x-wav",
                "audio/ogg",  # OGG
                "audio/vorbis",
                "audio/flac",  # FLAC
                "audio/x-flac",
                "audio/aac",  # AAC
                "audio/mp4",  # M4A
                "audio/x-m4a",
                "audio/webm",  # WebM Audio
                "audio/opus",  # Opus
                "audio/amr",  # AMR
                "audio/3gpp",  # 3GP Audio
                "audio/x-ms-wma",  # WMA
            ]

            if main_mime_type in audio_mime_types:
                return True

        # Check file extension as fallback
        if filename:
            audio_extensions = [
                ".mp3",
                ".wav",
                ".ogg",
                ".flac",
                ".aac",
                ".m4a",
                ".wma",
                ".opus",
                ".amr",
                ".3gp",
                ".webm",
            ]
            filename_lower = filename.lower()
            for ext in audio_extensions:
                if filename_lower.endswith(ext):
                    return True

        return False

    def create_content(self, message: str, file_parts: List[Part]) -> Optional[Content]:
        """Create content with message and file parts."""
        if not message.strip() and not file_parts:
            return None

        parts: List[Part] = []
        if message.strip():
            parts.append(Part(text=message))
        if file_parts:
            parts.extend(file_parts)
        return Content(role="user", parts=parts)

    def create_content_with_transcribed_audio(
        self, message: str, file_parts: List[Part], transcribed_texts: List[str]
    ) -> Optional[Content]:
        """Create content with message, file parts, and transcribed audio texts."""
        full_message = message
        if transcribed_texts:
            transcriptions = "\n\n".join(transcribed_texts)
            full_message += f"\n\n{transcriptions}"

        if not full_message.strip() and not file_parts:
            logger.info("Empty message and transcription detected, skipping processing")
            return None

        parts: List[Part] = []
        if full_message.strip():
            parts.append(Part(text=full_message))
        if file_parts:
            parts.extend(file_parts)
        return Content(role="user", parts=parts)

    def strip_file_parts(self, content: Content) -> Optional[Content]:
        """Return a copy of content with only text parts, removing any inline_data.

        Used as a fallback when the LLM model does not support file/image content
        (e.g. 'file type is not supported' error from OpenRouter).
        """
        if not content or not content.parts:
            return content
        text_parts = [
            p for p in content.parts
            if not (hasattr(p, "inline_data") and p.inline_data)
        ]
        if not text_parts:
            return None
        return Content(role=content.role, parts=text_parts)

    def _is_meaningful_transcription(self, transcribed_text: str) -> bool:
        """Check if transcribed text contains meaningful content."""
        # Only ignore if completely empty
        return bool(transcribed_text and transcribed_text.strip())

    async def add_session_to_memory(
        self,
        memory_service: Optional[BaseMemoryService],
        session_service: DatabaseSessionService,
        agent_id: str,
        effective_user_id: str,
        adk_session_id: str,
        root_agent: Optional[Any] = None,
    ):
        """Add completed session to memory."""
        # Skip if memory service is not provided
        if memory_service is None:
            logger.debug(f"Memory service not provided, skipping memory storage for session {adk_session_id}")
            return
            
        try:
            completed_session = await session_service.get_session(
                app_name=agent_id,
                user_id=effective_user_id,
                session_id=adk_session_id,
            )

            # Check if session was retrieved successfully
            if completed_session is None:
                logger.warning(f"Session {adk_session_id} not found, cannot add to memory")
                return

            # Extract compression parameters from agent config if available
            short_term_max_messages = None
            compression_interval = None
            memory_base_config_id = None

            # Get agent from database to extract config
            try:
                agent = await get_agent(self.db, agent_id)
                if agent:
                    # Extract compression parameters from agent config
                    if agent.config:
                        agent_config = agent.config if isinstance(agent.config, dict) else {}
                        if isinstance(agent_config, dict):
                            short_term_max_messages = agent_config.get("memory_short_term_max_messages")
                            compression_interval = agent_config.get("memory_medium_term_compression_interval")
                            memory_base_config_id = agent_config.get("memory_base_config_id")
            except Exception as e:
                logger.debug(f"Could not extract compression parameters from agent: {e}")

            # Pass database session and compression parameters if memory service supports it
            if hasattr(memory_service, "add_session_to_memory"):
                # Check if method accepts compression parameters
                import inspect
                sig = inspect.signature(memory_service.add_session_to_memory)
                params = list(sig.parameters.keys())
                
                # Build kwargs based on what the method accepts
                kwargs = {}
                if "db" in params:
                    kwargs["db"] = self.db
                if "short_term_max_messages" in params and short_term_max_messages is not None:
                    kwargs["short_term_max_messages"] = short_term_max_messages
                if "compression_interval" in params and compression_interval is not None:
                    kwargs["compression_interval"] = compression_interval
                if "memory_base_config_id" in params and memory_base_config_id is not None:
                    kwargs["memory_base_config_id"] = memory_base_config_id
                
                await memory_service.add_session_to_memory(completed_session, **kwargs)
            else:
                await memory_service.add_session_to_memory(completed_session)
            
            logger.debug(f"Successfully added session {adk_session_id} to memory")
        except Exception as e:
            # Check if it's an OpenSearch shard limit error
            error_str = str(e)
            if "maximum shards" in error_str and "validation_exception" in error_str:
                logger.warning(
                    f"OpenSearch shard limit reached for session {adk_session_id}. "
                    f"Memory service may be degraded. Consider cleaning up old indices or "
                    f"increasing shard limits in OpenSearch configuration. Error: {error_str}"
                )
            else:
                logger.error(
                    f"Failed to add session {adk_session_id} to memory service: {error_str}"
                )
            # Continue execution - memory failure shouldn't break agent functionality
