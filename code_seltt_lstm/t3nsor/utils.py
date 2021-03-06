from scipy.stats import entropy
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
from itertools import cycle, islice
import torch



MODES = ['ascending', 'descending', 'mixed']
CRITERIONS = ['entropy', 'var']


def _to_list(p):
    res = []
    for k, v in p.items():
        res += [k, ] * v
    return res


def _roundup(n, k):
    return int(np.ceil(n / 10**k)) * 10**k


def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def _get_all_factors(n, d=3, mode='ascending'):
    # n의 모든 공약수를 구해줌. d는 공약수의 갯수.
    # n=28,d=3 : [2,2,7] / n=28,d=2: [(4, 7), (2, 14)]
    p = _factorint2(n)
    if len(p) < d:
        p = p + [1, ] * (d - len(p))

    if mode == 'ascending':
        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x]))
    elif mode == 'descending':
        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x], reverse=True))

    elif mode == 'mixed':
        def prepr(x):
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[:N//2], x[N//2:]
            return tuple(_roundrobin(xf, xl))

    else:
        raise ValueError('Wrong mode specified, only {} are available'.format(MODES))

    raw_factors = multiset_partitions(p, d)
    clean_factors = [prepr(f) for f in raw_factors]
    clean_factors = list(set(clean_factors))
    return clean_factors


def _factorint2(p):
    return _to_list(factorint(p))


def auto_shape(n, d=3, criterion='entropy', mode='ascending'):
    factors = _get_all_factors(n, d=d, mode=mode)
    # n=28, d=2로 들어가면 모든 공약수는 옆에처럼 나옴: factors =  [(4, 7), (2, 14)]
    # 이것들의 엔트로피는  weights=[0.6554817739013927, 0.37677016125643675]가 나옴. (4,7)이 더 값진정보란 소린데 좌우지간 둘 중에 하나 선택할라고 만든것임
    if criterion == 'entropy':
        weights = [entropy(f) for f in factors]
    elif criterion == 'var':
        weights = [-np.var(f) for f in factors]
    else:
        raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))
    # [0.6554817739013927, 0.37677016125643675] 이거 둘 중 큰 값의 인덱스반환.따라서 i=0
    # list(factors[0]) = [4,7]반환
    i = np.argmax(weights)
    return list(factors[i])


def suggest_shape(n, d=3, criterion='entropy', mode='ascending'):
    weights = []
    for i in range(len(str(n))):

        n_i = _roundup(n, i)
        if criterion == 'entropy':
            weights.append(entropy(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        elif criterion == 'var':
            weights.append(-np.var(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        else:
            raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))

    i = np.argmax(weights)
    factors = auto_shape(int(_roundup(n, i)), d=d, mode=mode, criterion=criterion)
    return factors

def svd_fix(x):
    n = x.shape[0]
    m = x.shape[1]
    
    if n > m:
        u, s, v = torch.svd(x)
        
    else:
        u, s, v = torch.svd(x.t())
        v, u = u, v
    
    return u, s, v

def ind2sub(siz, idx):
    n = len(siz)
    b = len(idx)
    subs = []
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))

    for i in range(n - 1, -1, -1):
        subs.append(torch.floor(idx.float() / k[i]).long())
        idx = torch.fmod(idx, k[i])

    return torch.stack(subs[::-1], dim=1)
