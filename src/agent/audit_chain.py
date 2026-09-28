"""Hash-chained audit log for the intent-routed diagnostic agent.

Honest substitute for the "Hyperledger Fabric Blockchain Node" in the
original architecture diagram — a real distributed-ledger node is out of
scope for a hackathon demo. What this DOES give you, for real: each entry's
hash is computed over the previous entry's hash plus this entry's payload,
so any tampering with a past entry breaks every hash after it (the actual
tamper-evidence property a blockchain provides). What it does NOT give you:
distributed consensus, multiple independent nodes, or Byzantine fault
tolerance — it's a single-database hash chain, not a blockchain. See
MODEL_CARD.md for the full honest framing.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import AuditChainEntry

GENESIS_HASH = "0" * 64


def _compute_entry_hash(prev_hash: str, timestamp: float, kind: str, payload: Dict[str, Any]) -> str:
    body = json.dumps({"prev_hash": prev_hash, "timestamp": timestamp, "kind": kind, "payload": payload},
                       sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append_audit_entry(db: Session, kind: str, payload: Dict[str, Any]) -> AuditChainEntry:
    """Appends one entry to the hash chain and commits it. `kind` is a short
    label (e.g. 'diagnostic:thermal_stress_analysis', 'fault_transition')."""
    last = db.scalar(select(AuditChainEntry).order_by(AuditChainEntry.seq.desc()).limit(1))
    prev_hash = last.entry_hash if last else GENESIS_HASH
    seq = (last.seq + 1) if last else 1
    timestamp = time.time()

    entry_hash = _compute_entry_hash(prev_hash, timestamp, kind, payload)

    entry = AuditChainEntry(
        seq=seq,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
        kind=kind,
        payload_json=json.dumps(payload, default=str),
        timestamp=timestamp,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def verify_chain(db: Session) -> Dict[str, Any]:
    """Walks the full chain and recomputes each hash — returns whether it's
    intact, and the first broken link if not. Used for demonstration/testing,
    not called on the hot path."""
    entries = db.scalars(select(AuditChainEntry).order_by(AuditChainEntry.seq.asc())).all()
    prev_hash = GENESIS_HASH
    for e in entries:
        if e.prev_hash != prev_hash:
            return {"valid": False, "broken_at_seq": e.seq, "reason": "prev_hash mismatch"}
        recomputed = _compute_entry_hash(e.prev_hash, e.timestamp, e.kind, json.loads(e.payload_json))
        if recomputed != e.entry_hash:
            return {"valid": False, "broken_at_seq": e.seq, "reason": "entry_hash mismatch (tampered payload)"}
        prev_hash = e.entry_hash
    return {"valid": True, "entries_checked": len(entries)}
