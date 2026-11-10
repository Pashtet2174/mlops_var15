# MLOps laboratory - Variant 15

Task: predict building heating load (`Y1`) from `X1`-`X8` using the UCI Energy Efficiency dataset.

- `Y2` (Cooling Load) is excluded from model inputs.
- `X6` (Orientation) and `X8` (Glazing Area Distribution) are treated as categorical codes.
- Numeric preprocessing and categorical encoding are stored in the sklearn Pipeline.
- Experiments are compared by validation MAE; the test set is evaluated only after selecting the model.

The detailed Windows 11 PowerShell runbook is provided separately in the ChatGPT instructions.

## Project layout

- `data/raw/ENB2012_data.xlsx` - unchanged source workbook
- `src/train.py` - one MLflow training run
- `src/promote.py` - assign the `champion` alias
- `src/export_model.py` - export the immutable Registry version selected by alias
- `src/evaluate.py` - final test evaluation
- `src/app.py` - FastAPI regression API
- `src/check_api.py` - API/exported-model smoke test
- `Dockerfile`, `compose.yaml` - container build and Compose deployment

## Dependencies

Python 3.12 is used to match the lab manual. Install with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
