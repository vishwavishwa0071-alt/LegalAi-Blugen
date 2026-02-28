"""
Semantic Chunker for Legal Documents
Uses LLM to intelligently identify ORDER boundaries and create semantic chunks
that preserve the integrity of legal structures across page boundaries.
"""

import re
import json
import logging
import httpx
from typing import List, Dict, Tuple
from chunking_prompts import (
    order_detection_prompt,
    boundary_validation_prompt,
    content_classification_prompt,
    order_completion_check_prompt
)

# Constants
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 8000  # Smaller chunks to prevent memory overflow
RULE_SPLIT_THRESHOLD = 5000  # Split by rules if ORDER exceeds this size
OLLAMA_API_URL = "http://localhost:11434/api/generate"

class SemanticChunker:
    """
    Intelligent chunker that uses LLM to identify ORDER boundaries
    and create semantically meaningful chunks.
    """
    
    def __init__(self, llm_model: str = "gemma3:4b"):
        self.llm_model = llm_model
        self.client = httpx.Client(timeout=300.0)
        logging.info(f"Initialized SemanticChunker with model: {llm_model}")
    
    def chunk_document(self, full_text: str) -> List[str]:
        """
        Main method to chunk a legal document semantically.
        
        Args:
            full_text: Complete document text from Docling
            
        Returns:
            List of semantic chunks
        """
        logging.info("Starting semantic chunking process...")
        
        # Step 1: Identify ORDER boundaries using pattern matching and LLM validation
        order_boundaries = self._identify_order_boundaries(full_text)
        
        if not order_boundaries:
            logging.warning("No ORDERs detected, falling back to paragraph chunking")
            return self._fallback_chunking(full_text)
        
        logging.info(f"Detected {len(order_boundaries)} ORDERs")
        
        # Step 2: Extract ORDER chunks
        chunks = self._extract_order_chunks(full_text, order_boundaries)
        
        # Step 3: Handle non-ORDER content (intro, sections, schedules)
        chunks = self._handle_non_order_content(full_text, order_boundaries, chunks)
        
        # Step 4: Validate and post-process chunks
        chunks = self._validate_and_merge_chunks(chunks)
        
        logging.info(f"Created {len(chunks)} semantic chunks")
        self._log_chunk_statistics(chunks)
        
        return chunks
    
    def _identify_order_boundaries(self, text: str) -> List[Dict[str, any]]:
        """
        Identify ORDER boundaries using regex pattern matching.
        Returns list of dicts with 'number', 'title', 'start_pos', 'end_pos'
        """
        # Pattern to match ORDER headings in markdown format
        # Docling exports as markdown with ## ORDER I, ## ORDER VII, etc.
        # Pattern 1: Markdown headers with ORDER
        order_pattern = r'\n#+\s*ORDER\s+([IVXLCDM]+)\s*\n'
        
        matches = list(re.finditer(order_pattern, text, re.IGNORECASE | re.MULTILINE))
        
        if not matches:
            # Try alternate pattern for plain text (no markdown)
            order_pattern = r'\n\s*ORDER\s+([IVXLCDM]+)\s*[-–—]?\s*\n'
            matches = list(re.finditer(order_pattern, text, re.IGNORECASE | re.MULTILINE))
        
        if not matches:
            # Try pattern with title on same line
            order_pattern = r'\n#+?\s*ORDER\s+([IVXLCDM]+)\s*[-–—]?\s*([A-Z][A-Z\s,\-()]+?)(?=\n)'
            matches = list(re.finditer(order_pattern, text, re.IGNORECASE | re.MULTILINE))
        
        boundaries = []
        
        for i, match in enumerate(matches):
            order_num = match.group(1)
            order_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
            start_pos = match.start()
            
            # If no title found in the match, try to extract it from the next line
            if not order_title:
                # Look for the next line after the ORDER heading
                next_line_start = match.end()
                next_line_end = text.find('\n', next_line_start)
                if next_line_end != -1:
                    potential_title = text[next_line_start:next_line_end].strip()
                    # Check if it looks like a title (markdown header or capitalized text)
                    if potential_title.startswith('##'):
                        order_title = potential_title.replace('#', '').strip()
                    elif potential_title and potential_title[0].isupper():
                        order_title = potential_title
            
            # End position is start of next ORDER or end of document
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            boundaries.append({
                'number': order_num,
                'title': order_title,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'full_heading': match.group(0).strip()
            })
            
            logging.debug(f"Found ORDER {order_num}: {order_title}")
        
        return boundaries
    
    def _extract_order_chunks(self, text: str, boundaries: List[Dict]) -> List[str]:
        """
        Extract complete ORDER content as individual chunks.
        Split by Rules if ORDER is too large.
        """
        chunks = []
        
        for boundary in boundaries:
            order_text = text[boundary['start_pos']:boundary['end_pos']].strip()
            order_num = boundary['number']
            
            # Split by Rules if ORDER exceeds threshold OR max size
            if len(order_text) > RULE_SPLIT_THRESHOLD:
                logging.info(
                    f"ORDER {order_num} size: {len(order_text)} chars, splitting by Rules"
                )
                # Split large ORDERs by Rules
                sub_chunks = self._split_large_order(order_text, order_num)
                chunks.extend(sub_chunks)
            else:
                # Small enough to keep as single chunk
                chunks.append(order_text)
                logging.debug(
                    f"Extracted ORDER {order_num}: {len(order_text)} chars"
                )
        
        return chunks
    
    def _split_large_order(self, order_text: str, order_num: str) -> List[str]:
        """
        Split a large ORDER by Rules while keeping related content together.
        Each chunk gets ORDER context for better retrieval.
        """
        # Pattern to match Rule headings
        rule_pattern = r'\n\s*(\d+\.?\s+)?Rule\s+(\d+[A-Z]?)\s*[.:\-–—]?\s*'
        matches = list(re.finditer(rule_pattern, order_text, re.IGNORECASE))
        
        # Try RULES header pattern if no rules found
        if not matches:
            rules_header_pattern = r'\n#+?\s*RULES\s*\n'
            rules_match = re.search(rules_header_pattern, order_text, re.IGNORECASE)
            if rules_match:
                numbered_pattern = r'\n\s*(\d+)\.\s+'
                numbered_matches = list(re.finditer(numbered_pattern, order_text[rules_match.end():]))
                # Create simple Match-like objects with adjusted positions
                class SimpleMatch:
                    def __init__(self, original_match, offset):
                        self._start = offset + original_match.start()
                        self._end = offset + original_match.end()
                        self._groups = original_match.groups()
                    def start(self): return self._start
                    def end(self): return self._end
                    def group(self, n): return self._groups[n-1] if n > 0 else None
                
                matches = [SimpleMatch(m, rules_match.end()) for m in numbered_matches]
        
        if not matches:
            logging.warning(f"No rules found in ORDER {order_num}, using paragraph split")
            return self._split_by_paragraphs(order_text, MAX_CHUNK_SIZE)
        
        chunks = []
        order_header = order_text[:matches[0].start()].strip()
        
        for i, match in enumerate(matches):
            rule_start = match.start()
            rule_end = matches[i + 1].start() if i < len(matches) - 1 else len(order_text)
            rule_text = order_text[rule_start:rule_end].strip()
            
            # Create chunk with ORDER context
            if i == 0:
                chunk_text = f"{order_header}\n\n{rule_text}"
            else:
                # Add ORDER context for remaining rules
                chunk_text = f"ORDER {order_num} (continued)\n\n{rule_text}"
            
            # If single rule is still too large, split by paragraphs
            if len(chunk_text) > MAX_CHUNK_SIZE:
                sub_chunks = self._split_by_paragraphs(chunk_text, MAX_CHUNK_SIZE)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        logging.info(f"Split ORDER {order_num} into {len(chunks)} chunks")
        return chunks
    
    def _handle_non_order_content(
        self, 
        full_text: str, 
        order_boundaries: List[Dict], 
        existing_chunks: List[str]
    ) -> List[str]:
        """
        Handle content that's not part of ORDERs (Sections, Schedules, Preamble, etc.)
        """
        chunks = existing_chunks.copy()
        
        # Find text before first ORDER
        if order_boundaries:
            first_order_start = order_boundaries[0]['start_pos']
            preamble_text = full_text[:first_order_start].strip()
            
            if preamble_text and len(preamble_text) > 100:
                # Chunk preamble by sections or paragraphs
                preamble_chunks = self._chunk_by_sections(preamble_text)
                logging.info(f"Added {len(preamble_chunks)} preamble chunks")
                chunks = preamble_chunks + chunks
        
        # Find text after last ORDER (Schedules, Appendices)
        if order_boundaries:
            last_order_end = order_boundaries[-1]['end_pos']
            appendix_text = full_text[last_order_end:].strip()
            
            if appendix_text and len(appendix_text) > 100:
                appendix_chunks = self._chunk_by_sections(appendix_text)
                logging.info(f"Added {len(appendix_chunks)} appendix chunks")
                chunks.extend(appendix_chunks)
        
        return chunks
    
    def _chunk_by_sections(self, text: str) -> List[str]:
        """
        Chunk text by Section headings (for non-ORDER content).
        """
        # Pattern for Section headings
        section_pattern = r'\n\s*(?:Section\s+)?(\d+[A-Z]?)\s*[.:\-–—]\s*([A-Z][A-Za-z\s,\-()]+?)(?=\n)'
        
        matches = list(re.finditer(section_pattern, text, re.MULTILINE))
        
        if not matches:
            # No sections found, use paragraph chunking
            return self._split_by_paragraphs(text, MAX_CHUNK_SIZE)
        
        chunks = []
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) > MAX_CHUNK_SIZE:
                sub_chunks = self._split_by_paragraphs(section_text, MAX_CHUNK_SIZE)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section_text)
        
        # Handle text before first section
        if matches:
            intro_text = text[:matches[0].start()].strip()
            if intro_text and len(intro_text) > 100:
                intro_chunks = self._split_by_paragraphs(intro_text, MAX_CHUNK_SIZE)
                chunks = intro_chunks + chunks
        
        return chunks
    
    def _split_by_paragraphs(self, text: str, max_size: int) -> List[str]:
        """
        Fallback: split text by paragraphs, merging to reach target size.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _validate_and_merge_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validate chunks and merge very small ones.
        """
        validated_chunks = []
        current_merge = ""
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
            
            # If chunk is too small, merge with next
            if len(chunk) < MIN_CHUNK_SIZE:
                current_merge += "\n\n" + chunk if current_merge else chunk
            else:
                # First save any pending merge
                if current_merge:
                    if len(current_merge) < MIN_CHUNK_SIZE and validated_chunks:
                        # Merge with previous chunk
                        validated_chunks[-1] += "\n\n" + current_merge
                    else:
                        validated_chunks.append(current_merge)
                    current_merge = ""
                
                validated_chunks.append(chunk)
        
        # Handle remaining merge
        if current_merge:
            if validated_chunks and len(current_merge) < MIN_CHUNK_SIZE:
                validated_chunks[-1] += "\n\n" + current_merge
            else:
                validated_chunks.append(current_merge)
        
        return validated_chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback to simple paragraph-based chunking if ORDER detection fails.
        """
        logging.warning("Using fallback paragraph chunking")
        return self._split_by_paragraphs(text, 1000)
    
    def _log_chunk_statistics(self, chunks: List[str]):
        """
        Log statistics about chunk sizes for analysis.
        """
        sizes = [len(c) for c in chunks]
        
        if not sizes:
            return
        
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        logging.info(f"Chunk Statistics:")
        logging.info(f"  Total chunks: {len(chunks)}")
        logging.info(f"  Average size: {avg_size:.0f} chars")
        logging.info(f"  Min size: {min_size} chars")
        logging.info(f"  Max size: {max_size} chars")
        
        # Size distribution
        small = sum(1 for s in sizes if s < 1000)
        medium = sum(1 for s in sizes if 1000 <= s < 5000)
        large = sum(1 for s in sizes if s >= 5000)
        
        logging.info(f"  Size distribution:")
        logging.info(f"    < 1KB: {small} chunks")
        logging.info(f"    1-5KB: {medium} chunks")
        logging.info(f"    > 5KB: {large} chunks")
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
