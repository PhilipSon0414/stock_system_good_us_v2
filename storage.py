"""Atomic local writes; readers never see half-written JSON/JSONL."""
import os
import tempfile
from pathlib import Path

def write_text(path, text, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        return True
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
