"""
Text-to-Speech service backed by Chatterbox Multilingual (Resemble AI).

Replaces the deprecated Coqui XTTS v2. Voice profile WAVs in `voice_profiles/`
remain compatible — Chatterbox accepts any WAV reference for zero-shot cloning.

Falls back to Google TTS (gTTS) if model loading or synthesis fails so the
chat pipeline degrades gracefully rather than 500ing the whole turn.

Synthesis result reports which engine produced the audio so the caller can
notify the user when voice cloning was silently dropped during fallback.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SynthResult:
    output_path: str
    engine: str  # "chatterbox" or "gtts"
    fallback: bool  # True if the caller's preferred path was not taken
    voice_cloned: bool  # True if a speaker WAV was actually applied


class TTSService:
    """Text-to-Speech service. Lazy-loads the model on first synthesis."""

    def __init__(self):
        self.provider = settings.TTS_PROVIDER
        self.model = None

    def _check_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    async def initialize(self):
        """Load the Chatterbox model (downloaded from HuggingFace on first run)."""
        if self.model is not None:
            return

        if self.provider != "chatterbox":
            raise ValueError(f"Unsupported TTS provider: {self.provider}")

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = "cuda" if self._check_cuda() else "cpu"
            logger.info(f"Loading Chatterbox multilingual TTS on {device}...")
            self.model = await asyncio.to_thread(
                ChatterboxMultilingualTTS.from_pretrained, device=device
            )
            logger.info(f"Chatterbox loaded (sr={self.model.sr}, device={device})")

        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            raise

    async def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> SynthResult:
        """
        Synthesize speech.

        Args:
            text: Text to speak.
            output_path: Destination WAV path.
            speaker_wav: Optional reference audio for voice cloning (≥10s recommended).
            language: 2-letter code from Chatterbox's 23-language set.

        Returns:
            SynthResult describing the WAV path and which engine was used.
            `fallback=True` indicates the preferred Chatterbox path failed
            and gTTS was used instead — voice cloning is lost in that case.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.model is None:
                await self.initialize()

            logger.info(f"Synthesizing (chatterbox, lang={language}): {text[:80]}...")

            if speaker_wav and not Path(speaker_wav).exists():
                logger.warning(f"Speaker WAV not found: {speaker_wav!r} — using default voice")
                speaker_wav = None

            kwargs = {"language_id": language}
            if speaker_wav:
                kwargs["audio_prompt_path"] = speaker_wav

            wav = await asyncio.to_thread(self.model.generate, text, **kwargs)
            await asyncio.to_thread(torchaudio.save, output_path, wav, self.model.sr)

            logger.info(
                f"Synthesis complete{' (cloned voice)' if speaker_wav else ''}: {output_path}"
            )
            return SynthResult(
                output_path=output_path,
                engine="chatterbox",
                fallback=False,
                voice_cloned=bool(speaker_wav),
            )

        except Exception as e:
            if speaker_wav:
                logger.warning(
                    f"Chatterbox voice-clone failed — cloned voice NOT applied, "
                    f"falling back to gTTS default. Error: {e}"
                )
            else:
                logger.warning(f"Chatterbox failed ({e}), falling back to gTTS")
            await self._gtts_fallback(text, output_path, language)
            return SynthResult(
                output_path=output_path,
                engine="gtts",
                fallback=True,
                voice_cloned=False,
            )

    async def _gtts_fallback(self, text: str, output_path: str, language: str = "en") -> str:
        """Network-only fallback using Google TTS — no GPU/local model required."""
        try:
            from gtts import gTTS
            from pydub import AudioSegment

            logger.info(f"Synthesizing (gTTS): {text[:80]}...")
            mp3_path = output_path.replace(".wav", "_gtts.mp3")

            await asyncio.to_thread(
                lambda: gTTS(text=text, lang=language, slow=False).save(mp3_path)
            )
            await asyncio.to_thread(
                lambda: AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
            )
            Path(mp3_path).unlink(missing_ok=True)

            logger.info(f"gTTS synthesis complete: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"gTTS also failed: {e}")
            raise

    async def synthesize_bytes(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> bytes:
        """Synthesize and return WAV bytes (used by REST callers)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await self.synthesize(text, tmp_path, speaker_wav, language)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Suppress unused-name warning — re-exported for type hints elsewhere
__all__ = ["TTSService", "SynthResult", "tts_service"]


# Global instance
tts_service = TTSService()
