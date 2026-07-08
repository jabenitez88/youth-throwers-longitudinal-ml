# Data folder

Raw athlete spreadsheets are not included in this repository.

To rerun the analysis, place the raw files in `data/raw/`:

- `DatosGrupo1.xlsx`
- `DatosGrupo2.xlsx`
- `DatosGrupo3.xlsx`

The analysis script can also read the raw files from another location by setting:

```bash
ML_THROWERS_RAW_DIR=/path/to/raw/files
```

The script creates `data/processed/` with row-level derived cohorts. These files may contain athlete-level longitudinal information and should not be committed or shared publicly without data-owner approval.
