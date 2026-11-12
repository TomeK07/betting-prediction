# Betting Prediction (NBA · Евролига · Еврокуп)

## Прв пат — подесување (Windows, еднаш)
1. Отвори го фолдерот `Documents\betting-prediction` во VS Code (File → Open Folder).
2. Терминал во VS Code (Ctrl+`) и:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (Ако `python` не се препознава → инсталирај Python 3.11+ од python.org, штиклирај „Add to PATH“.)
3. Избриши го фолдерот `_to_delete` (стар, не ти треба).
4. Отвори Claude во VS Code и напиши:
   **„Прочитај CLAUDE.md, docs/PROCES.md, docs/BUILD_PROMPT.md и docs/PLAN.md и почни со Фаза 1.“**

## Секоја вечер (кога проектот е готов)
```
venv\Scripts\activate
python main.py fetch                 # податоци + снимка од Mozzart линиите (секој пат кога пушташ = историја за CLV)
python main.py grade                 # резултати на изборите што чекаат
python main.py stats --nedela        # погодок, калибрација НАД/ПОД, тимски модел минатата недела
python main.py povredi               # отсуства НОВО/СТАРО + ефект врз соиграчите
python main.py linii                # снимка на линиите пред натпреварите (за CLV)
python main.py picks --league all    # избори НА ХАРТИЈА (ништо не се уплаќа) → reports/picks_*.md
python main.py profil               # (неделно) профили на тимови → data/timovi/
```
Извештајот е во `reports\`.

## Документи
| | |
|---|---|
| `CLAUDE.md` | правила за Claude (се чита автоматски) |
| `docs/PROCES.md` | зошто и како — целиот процес, метрики, подобрување |
| `docs/BUILD_PROMPT.md` | техничка спецификација |
| `docs/PLAN.md` | статус, одлуки, дневник на промени |
| `docs/znaenje/` | досегашно знаење (модел за играчи, лекции, профили) |
| `data/archive/` | досегашни избори (CSV) |

## Ако сакаш нешто да смениш
Кажи му на Claude што сакаш → тој го менува кодот/документот и го запишува во `docs/PLAN.md` (што, зошто, кога).
