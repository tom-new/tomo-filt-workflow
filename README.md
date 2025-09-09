# Workflow description for tomography filtering

1. `.pvtu` -> `interp.py`
2. `.pvtu` -> `LLNL_ToFi.py` -> `reconstruct.py`
3. `.pvtu` -> `fd_to_s40.py` -> `dofilt_ES_new` -> `s40_to_nc.py`
