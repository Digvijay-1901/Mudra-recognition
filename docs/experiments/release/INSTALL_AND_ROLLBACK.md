# Install and rollback · Windows PowerShell

Run from your project root after approval. Set the source to the extracted preview directory. The gate is included. Keep a separate copy of your original upload.

```powershell
$ErrorActionPreference = 'Stop'
$src = 'C:\Downloads\mudra_release_candidate'
$backup = Join-Path $PWD ('.phase5-backup\' + [guid]::NewGuid().ToString())
function Install-Phase5File($file) {
    $dest = Join-Path $PWD $file
    $saved = Join-Path $backup $file
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $saved) | Out-Null
    if (Test-Path -LiteralPath $dest) { Copy-Item -LiteralPath $dest -Destination $saved }
    else { New-Item -ItemType File "$saved.new" | Out-Null }
    Copy-Item -LiteralPath (Join-Path $src $file) -Destination $dest -Force
}
function Undo-Phase5File($file) {
    $dest = Join-Path $PWD $file
    $saved = Join-Path $backup $file
    if (Test-Path -LiteralPath $saved) { Copy-Item -LiteralPath $saved -Destination $dest -Force }
    elseif ((Test-Path -LiteralPath "$saved.new") -and (Test-Path -LiteralPath $dest)) {
        Remove-Item -LiteralPath $dest
    }
}
Install-Phase5File '.gitignore'
Install-Phase5File 'INSTALL_AND_ROLLBACK.md'
Install-Phase5File 'README.md'
Install-Phase5File '_quiet.py'
Install-Phase5File 'brain.py'
Install-Phase5File 'detector.py'
Install-Phase5File 'examples/anjali_one.jpg'
Install-Phase5File 'examples/anjali_two.jpg'
Install-Phase5File 'examples/mayura.jpg'
Install-Phase5File 'examples/pathaka.jpg'
Install-Phase5File 'feature_extraction.py'
Install-Phase5File 'feature_extraction_double.py'
Install-Phase5File 'gate.py'
Install-Phase5File 'gate/.gate.joblib.efIggz'
Install-Phase5File 'gate/gate.joblib'
Install-Phase5File 'model_double_v3/feature_scaler.joblib'
Install-Phase5File 'model_double_v3/label_encoder.joblib'
Install-Phase5File 'model_double_v3/mlp_mudra_classifier.joblib'
Install-Phase5File 'model_double_v3_norm/feature_scaler.joblib'
Install-Phase5File 'model_double_v3_norm/label_encoder.joblib'
Install-Phase5File 'model_double_v3_norm/mlp_mudra_classifier.joblib'
Install-Phase5File 'model_v3/feature_scaler.joblib'
Install-Phase5File 'model_v3/label_encoder.joblib'
Install-Phase5File 'model_v3/mlp_mudra_classifier.joblib'
Install-Phase5File 'model_v3_norm/feature_scaler.joblib'
Install-Phase5File 'model_v3_norm/label_encoder.joblib'
Install-Phase5File 'model_v3_norm/mlp_mudra_classifier.joblib'
Install-Phase5File 'overlay.py'
Install-Phase5File 'requirements.txt'
Install-Phase5File 'streamlit_app.py'
Write-Host "Rollback backup: $backup"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

To roll back, stop Streamlit and use these lines in the same PowerShell session with the same backup path and helper. Each line restores a prior file or removes the installed file if it was new.

```powershell
Undo-Phase5File 'streamlit_app.py'
Undo-Phase5File 'requirements.txt'
Undo-Phase5File 'overlay.py'
Undo-Phase5File 'model_v3_norm/mlp_mudra_classifier.joblib'
Undo-Phase5File 'model_v3_norm/label_encoder.joblib'
Undo-Phase5File 'model_v3_norm/feature_scaler.joblib'
Undo-Phase5File 'model_v3/mlp_mudra_classifier.joblib'
Undo-Phase5File 'model_v3/label_encoder.joblib'
Undo-Phase5File 'model_v3/feature_scaler.joblib'
Undo-Phase5File 'model_double_v3_norm/mlp_mudra_classifier.joblib'
Undo-Phase5File 'model_double_v3_norm/label_encoder.joblib'
Undo-Phase5File 'model_double_v3_norm/feature_scaler.joblib'
Undo-Phase5File 'model_double_v3/mlp_mudra_classifier.joblib'
Undo-Phase5File 'model_double_v3/label_encoder.joblib'
Undo-Phase5File 'model_double_v3/feature_scaler.joblib'
Undo-Phase5File 'gate/gate.joblib'
Undo-Phase5File 'gate/.gate.joblib.efIggz'
Undo-Phase5File 'gate.py'
Undo-Phase5File 'feature_extraction_double.py'
Undo-Phase5File 'feature_extraction.py'
Undo-Phase5File 'examples/pathaka.jpg'
Undo-Phase5File 'examples/mayura.jpg'
Undo-Phase5File 'examples/anjali_two.jpg'
Undo-Phase5File 'examples/anjali_one.jpg'
Undo-Phase5File 'detector.py'
Undo-Phase5File 'brain.py'
Undo-Phase5File '_quiet.py'
Undo-Phase5File 'README.md'
Undo-Phase5File 'INSTALL_AND_ROLLBACK.md'
Undo-Phase5File '.gitignore'
```

The .gitignore applies to untracked files. Already-tracked data and old weights remain tracked until you explicitly remove them from Git’s index.
