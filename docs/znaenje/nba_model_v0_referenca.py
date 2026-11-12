"""REFERENCA (v0, 23.10.2026) — стар рачен NBA модел од Claude проектот.
НЕ се користи директно (заковани патеки /home/claude, податоци во текстуални фајлови).
Служи како пример за логиката: минути -> стапка -> тимски тотал од пазарот -> DvP -> мешавина blowout -> P(НАД).
Новиот код (model/) мора да ја подобри: модел БЕЗ пазарот во проекцијата, калибрација од архивата.
"""
import math, collections, sys
from scipy.stats import norm, nbinom

LG = 115.5          # просечни поени на тим 2025-26
SHRINK_DVP = 0.5    # DvP е од минатата сезона, составите се сменети -> половина ефект
W_MODEL = 0.6       # финална веројатност = 0.6 модел + 0.4 пазар (фер)
K = {'A': 6, 'R': 10}   # negative binomial "k" за асистенции и скокови

# --- одбрана против позиција (по 48 мин, 2025-26) ---
D = {}
for x in open('/home/claude/nba/dvp.txt').read().strip().split(';'):
    p, t, pts, t3, r, a = x.split(',')
    D[(t, p)] = dict(P=float(pts), R=float(r), A=float(a))
AV = collections.defaultdict(lambda: collections.defaultdict(float))
for (t, p), v in D.items():
    for k in v: AV[p][k] += v[k] / 30
ALIAS = {'NYK': 'NY', 'SAS': 'SA', 'GSW': 'GS', 'NOP': 'NO', 'UTA': 'UTA', 'PHX': 'PHO'}
def dvp(opp, pos, m):
    o = ALIAS.get(opp, opp)
    r = D[(o, pos)][m] / AV[pos][m]
    return 1 + SHRINK_DVP * (r - 1)

PB = 0.35   # 24.10 лекција: рана сезона 11/28 натпревари решени со 15+ -> стартерите губат минути
def p_over(m, mu, line):
    """Мешавина: 65% нормален натпревар, 35% „голема разлика / фаул-проблеми“ (−20% минути)."""
    return (1 - PB) * _p(m, mu, line) + PB * _p(m, 0.8 * mu, line)

def _p(m, mu, line):
    if m == 'P':
        sd = 0.28 * mu + 2.5
        return 1 - norm.cdf(line, mu, sd)
    k = K[m]; p = k / (k + mu)
    return 1 - nbinom.cdf(math.floor(line), k, p)

games, rows = [], []
cur = None
for ln in open('/home/claude/nba/players.txt'):
    ln = ln.strip()
    if not ln or ln.startswith('#'): continue
    f = ln.split('|')
    if f[0] == 'G':
        h, a, sp, tot = f[1], f[2], float(f[3]), float(f[4])
        cur = dict(h=h, a=a, sp=sp, tot=tot, imp={h: (tot - sp) / 2, a: (tot + sp) / 2})
        games.append(cur); continue
    name, tm, opp, pos, L, O, new, blow, lines, mo = f
    Lm, Lp, Lr, La = map(float, L.split(',')); Om, Op, Or, Oa = map(float, O.split(','))
    new, blow = int(new), int(blow)
    # 1) МИНУТИ: 60% минатата сезона + 40% прв натпревар (15% ако прв беше разлика 20+)
    wm = 0.15 if blow else 0.4
    if blow and Lm >= 28: wm = 0.1
    if blow and Lm < 28 and Om > Lm: wm = 0.1
    if blow and Lm < 28 and Om < Lm: wm = 0.6
    mins = float(mo) if mo else (1 - wm) * Lm + wm * Om
    if abs(cur['sp']) >= 8: mins -= 1.5           # ризик од голема разлика -> помалку минути за стартерите
    # 2) УЛОГА (по минута)
    ws = (0.4 if new else 0.25) * (0.5 if blow else 1)
    rate = {}
    for m, l, o in (('P', Lp, Op), ('R', Lr, Or), ('A', La, Oa)):
        rl = l / Lm; ro = o / Om if Om > 0 else rl
        rate[m] = (1 - ws) * rl + ws * ro
    # 3) ТЕМПО/ТИМСКИ ТОТАЛ од пазарот  4) МАТЧАП
    tf = cur['imp'][tm] / LG
    proj = {}
    proj['P'] = rate['P'] * mins * tf * dvp(opp, pos, 'P')
    proj['A'] = rate['A'] * mins * tf * dvp(opp, pos, 'A')
    proj['R'] = rate['R'] * mins * math.sqrt(cur['tot'] / (2 * LG)) * dvp(opp, pos, 'R')
    for seg in lines.split(';'):
        m, line, ov, un = seg.split(); line, ov, un = float(line), float(ov), float(un)
        fair_o = (1 / ov) / (1 / ov + 1 / un)
        po = p_over(m, proj[m], line)
        pf_o = W_MODEL * po + (1 - W_MODEL) * fair_o
        side, p, odd = ('НАД', pf_o, ov) if pf_o >= 0.5 else ('ПОД', 1 - pf_o, un)
        gap = (proj[m] - line) / max(line, 1)
        rows.append(dict(g=f"{cur['h']}-{cur['a']}", n=name, tm=tm, m=m, line=line, proj=round(proj[m], 1),
                         mins=round(mins, 1), side=side, p=round(p * 100, 1), pm=round((po if side=='НАД' else 1-po) * 100, 1),
                         odd=odd, ev=round(p * odd, 3), gap=round(gap, 2), fair=round((fair_o if side=='НАД' else 1-fair_o)*100,1)))
# (печатење на извештај — испуштено; филтер: p>=56, EV>=1.04, |gap|<=0.25, мин>=24)
