import json, ssl, sys, urllib.request, concurrent.futures as cf, time
urls = json.load(open(sys.argv[1]))
import os
_ca = os.environ.get('AUDIT_CA_BUNDLE', '/root/.ccr/ca-bundle.crt')
ctx = ssl.create_default_context(cafile=_ca) if os.path.exists(_ca) else ssl.create_default_context()
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), urllib.request.ProxyHandler())
def get(u):
    for attempt in range(4):
        try:
            with opener.open(u + '.CHECKSUM', timeout=60) as r:
                return u, r.status, r.read().decode().strip()
        except urllib.error.HTTPError as e:
            return u, e.code, ''
        except Exception as e:
            time.sleep(2 ** attempt)
            err = repr(e)
    return u, -1, err
res = {}
with cf.ThreadPoolExecutor(8) as ex:
    for u, code, body in ex.map(get, urls):
        live = body.split()[0] if code == 200 and body else None
        res[u] = {'code': code, 'live_sha256': live, 'registered': urls[u], 'match': live in urls[u] if live else False, 'fname_ok': body.endswith(u.rsplit('/',1)[1]) if code==200 else None}
json.dump(res, open(sys.argv[2], 'w'), indent=0)
from collections import Counter
print(Counter((v['code'], v['match']) for v in res.values()))
print('filename mismatches', [u for u,v in res.items() if v['fname_ok'] is False][:5])
print('non-match examples', [(u, v) for u,v in res.items() if not v['match']][:5])
