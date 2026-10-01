$ErrorActionPreference = 'Stop'

# This script lives in formal_experiment/scripts. It finds the shared Python
# environment and the Qwen model stored under the formal experiment root.
$formalRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $formalRoot
$python = Join-Path $workspaceRoot '.venv312\Scripts\python.exe'
$sourceDir = Join-Path $formalRoot 'src'
$defaultModel = Join-Path $formalRoot 'models\qwen3-embedding-0.6b'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
    throw "Service source directory not found: $sourceDir"
}
if (-not $env:QWEN_EMBEDDING_MODEL_PATH) {
    $env:QWEN_EMBEDDING_MODEL_PATH = $defaultModel
}
if (-not (Test-Path -LiteralPath $env:QWEN_EMBEDDING_MODEL_PATH -PathType Container)) {
    throw "Qwen model directory not found: $env:QWEN_EMBEDDING_MODEL_PATH"
}

& $python -m uvicorn qwen_embedding_service:app --app-dir $sourceDir --host 127.0.0.1 --port 8000
