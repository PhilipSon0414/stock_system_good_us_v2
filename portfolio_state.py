"""User-maintained account snapshot; no orders are submitted."""
import json
import math
from pathlib import Path
PATH = Path(__file__).parent / 'portfolio_state.json'

def load_state(session, path=PATH):
    if not path.exists():
        return None, '보유 현황 미등록: 비중 제안 보류'
    try:
        state = json.loads(path.read_text())
        if state['as_of_session'] != session:
            return None, '보유 현황 기준일 불일치: 비중 제안 보류'
        positions = state['positions']
        if not isinstance(positions, list):
            raise ValueError('positions must be a list')
        seen = set()
        for p in positions:
            if p['ticker'] in seen or not isinstance(p['ticker'], str):
                raise ValueError('duplicate ticker')
            seen.add(p['ticker'])
            if any(not math.isfinite(float(p[k])) for k in ('weight','stop_pct')):
                raise ValueError('non-finite holding')
            if not 0 < p['weight'] <= 1 or not 0 < p['stop_pct'] < 1:
                raise ValueError('invalid holding')
        if sum(p['weight'] for p in positions) > 1:
            raise ValueError('weights exceed account equity')
        return state, None
    except (KeyError, ValueError, TypeError):
        return None, '보유 현황 형식 오류: 비중 제안 보류'
