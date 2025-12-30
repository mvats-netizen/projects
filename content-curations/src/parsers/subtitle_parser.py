"""
Subtitle Parser for SRT and VTT Files

Converts subtitle files into timestamped segments for the chunking pipeline.

Supported formats:
- SRT (SubRip)
- VTT (WebVTT)

Usage:
    >>> parser = SubtitleParser()
    >>> segments = parser.parse_file("video_subtitles.srt")
    >>> # or
    >>> segments = parser.parse_file("video_subtitles.vtt")
    >>> 
    >>> # Use with chunker
    >>> chunks = chunker.chunk_transcript(segments, item_id="video_123")
"""

import re
from pathlib import Path
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)


def _parse_srt_time(time_str: str) -> float:
    """
    Parse SRT timestamp to seconds.
    
    Format: HH:MM:SS,mmm (e.g., "00:01:23,456")
    """
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def _parse_vtt_time(time_str: str) -> float:
    """
    Parse VTT timestamp to seconds.
    
    Format: HH:MM:SS.mmm or MM:SS.mmm (e.g., "00:01:23.456" or "01:23.456")
    """
    time_str = time_str.strip()
    parts = time_str.split(":")
    
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def parse_srt(content: str) -> List[dict]:
    """
    Parse SRT content into timestamped segments.
    
    Args:
        content: Raw SRT file content
        
    Returns:
        List of dicts with 'text', 'start', 'end' keys
    """
    segments = []
    
    # SRT pattern: index, timestamp line, text lines, blank line
    # Example:
    # 1
    # 00:00:01,000 --> 00:00:04,000
    # Hello, welcome to this video.
    #
    # 2
    # 00:00:04,500 --> 00:00:07,000
    # Today we'll learn about...
    
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        
        if len(lines) < 2:
            continue
        
        # Find timestamp line (contains " --> ")
        timestamp_idx = None
        for i, line in enumerate(lines):
            if ' --> ' in line:
                timestamp_idx = i
                break
        
        if timestamp_idx is None:
            continue
        
        # Parse timestamp
        timestamp_line = lines[timestamp_idx]
        match = re.match(r'([\d:,\.]+)\s*-->\s*([\d:,\.]+)', timestamp_line)
        
        if not match:
            continue
        
        start_time = _parse_srt_time(match.group(1))
        end_time = _parse_srt_time(match.group(2))
        
        # Get text (all lines after timestamp)
        text_lines = lines[timestamp_idx + 1:]
        text = ' '.join(line.strip() for line in text_lines if line.strip())
        
        # Remove HTML-style tags often found in subtitles
        text = re.sub(r'<[^>]+>', '', text)
        
        if text:
            segments.append({
                'text': text,
                'start': start_time,
                'end': end_time,
            })
    
    logger.info(f"Parsed {len(segments)} segments from SRT content")
    return segments


def parse_vtt(content: str) -> List[dict]:
    """
    Parse VTT content into timestamped segments.
    
    Args:
        content: Raw VTT file content
        
    Returns:
        List of dicts with 'text', 'start', 'end' keys
    """
    segments = []
    
    # Remove WEBVTT header and metadata
    lines = content.strip().split('\n')
    
    # Find where actual cues start (after WEBVTT line and any header metadata)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('WEBVTT'):
            start_idx = i + 1
            break
    
    # Skip any header metadata (lines before first timestamp)
    while start_idx < len(lines):
        line = lines[start_idx].strip()
        if not line or ' --> ' in line or (line and not line.startswith('NOTE')):
            break
        start_idx += 1
    
    # Rejoin and split by blank lines
    content_after_header = '\n'.join(lines[start_idx:])
    blocks = re.split(r'\n\s*\n', content_after_header.strip())
    
    for block in blocks:
        block_lines = block.strip().split('\n')
        
        if not block_lines:
            continue
        
        # Find timestamp line
        timestamp_idx = None
        for i, line in enumerate(block_lines):
            if ' --> ' in line:
                timestamp_idx = i
                break
        
        if timestamp_idx is None:
            continue
        
        # Parse timestamp (may have positioning info after times)
        timestamp_line = block_lines[timestamp_idx]
        match = re.match(r'([\d:\.]+)\s*-->\s*([\d:\.]+)', timestamp_line)
        
        if not match:
            continue
        
        start_time = _parse_vtt_time(match.group(1))
        end_time = _parse_vtt_time(match.group(2))
        
        # Get text
        text_lines = block_lines[timestamp_idx + 1:]
        text = ' '.join(line.strip() for line in text_lines if line.strip())
        
        # Remove VTT styling tags
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\{[^}]+\}', '', text)  # Remove position markers
        
        if text:
            segments.append({
                'text': text,
                'start': start_time,
                'end': end_time,
            })
    
    logger.info(f"Parsed {len(segments)} segments from VTT content")
    return segments


class SubtitleParser:
    """
    Unified parser for subtitle files.
    
    Automatically detects format (SRT or VTT) and parses accordingly.
    
    Example:
        >>> parser = SubtitleParser()
        >>> 
        >>> # From file
        >>> segments = parser.parse_file("subtitles.srt")
        >>> 
        >>> # From string
        >>> segments = parser.parse_string(srt_content, format="srt")
        >>> 
        >>> # Auto-detect from content
        >>> segments = parser.parse_string(content)
    """
    
    def parse_file(self, filepath: str) -> List[dict]:
        """
        Parse a subtitle file.
        
        Args:
            filepath: Path to SRT or VTT file
            
        Returns:
            List of timestamped segments
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {filepath}")
        
        content = path.read_text(encoding='utf-8')
        
        # Detect format from extension
        ext = path.suffix.lower()
        
        if ext == '.srt':
            return parse_srt(content)
        elif ext in ['.vtt', '.webvtt']:
            return parse_vtt(content)
        else:
            # Try to auto-detect
            return self.parse_string(content)
    
    def parse_string(
        self,
        content: str,
        format: Optional[str] = None,
    ) -> List[dict]:
        """
        Parse subtitle content from string.
        
        Args:
            content: Subtitle file content
            format: "srt", "vtt", or None for auto-detect
            
        Returns:
            List of timestamped segments
        """
        if format:
            format = format.lower()
            if format == 'srt':
                return parse_srt(content)
            elif format in ['vtt', 'webvtt']:
                return parse_vtt(content)
            else:
                raise ValueError(f"Unknown format: {format}")
        
        # Auto-detect
        if content.strip().startswith('WEBVTT'):
            return parse_vtt(content)
        else:
            # Assume SRT (more common, also works for plain text with timestamps)
            return parse_srt(content)
    
    def merge_short_segments(
        self,
        segments: List[dict],
        min_duration: float = 2.0,
    ) -> List[dict]:
        """
        Merge very short segments together.
        
        Useful when subtitles are split at unnatural points.
        
        Args:
            segments: List of timestamped segments
            min_duration: Minimum segment duration in seconds
            
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        merged = []
        current = None
        
        for segment in segments:
            if current is None:
                current = segment.copy()
                continue
            
            duration = current['end'] - current['start']
            
            if duration < min_duration:
                # Merge with next segment
                current['text'] += ' ' + segment['text']
                current['end'] = segment['end']
            else:
                merged.append(current)
                current = segment.copy()
        
        # Don't forget the last segment
        if current:
            merged.append(current)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged
    
    def to_plain_text(self, segments: List[dict]) -> str:
        """
        Convert timestamped segments to plain text.
        
        Args:
            segments: List of timestamped segments
            
        Returns:
            Plain text transcript
        """
        return ' '.join(s['text'] for s in segments)

