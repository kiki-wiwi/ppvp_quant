"""User-approved CP3 window adapter; frozen CP1 calendar remains byte-identical."""
from . import config

END_DATE = '2026-06-01'
LAST_MONTH = '2026-05'

def window(name, asset='BTC'):
    original = config.window(name, asset)
    if name != 'Post-IS':
        return original
    return config.Window('Post-IS', config.utc('2024-01-01T00:00Z'),
                         config.utc(END_DATE+'T00:00Z'), config.utc('2024-01-01T00:00Z'))
