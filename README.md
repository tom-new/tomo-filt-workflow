# Workflow description for tomography filtering

1. `.pvtu` -> `interp.py`
2. `.pvtu` -> `LLNL_ToFi.py` -> `llnl_to_nc.py`
3. `.pvtu` -> `fd_to_s40.ipynb` -> `dofilt_ES_new` -> `s40_to_nc.py`

## Requirements

- LLNL_ToFi_3 (LLNL-G3D-JPS filtering code)
- tomofilt_ES_new (S40RTS filtering code)
