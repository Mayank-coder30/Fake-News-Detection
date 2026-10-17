import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

@dataclass
class EvidenceDocument:
    passage_id: str
    title: str
    text: str
    source_id: str
    source_url: Optional[str] = None

    def to_dict(self):
        return asdict(self)

class DocumentLoader:
    """Loads and validates evidence documents from disk in JSONL format."""

    @staticmethod
    def load_jsonl(file_path: Path) -> List[EvidenceDocument]:
        """Loads evidence documents from a JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {file_path}")

        documents = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = EvidenceDocument(
                        passage_id=str(data.get("passage_id", f"doc_{line_num}")),
                        title=str(data.get("title", "Untitled")),
                        text=str(data.get("text", "")).strip(),
                        source_id=str(data.get("source_id", "local_corpus")),
                        source_url=data.get("source_url")
                    )
                    if doc.text:
                        documents.append(doc)
                except Exception as e:
                    continue
        return documents
