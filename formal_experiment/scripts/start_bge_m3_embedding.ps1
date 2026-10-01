[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

$ErrorActionPreference = 'Stop'
$formalRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $formalRoot
$python = Join-Path $workspaceRoot '.venv312\Scripts\python.exe'
$sourceDir = Join-Path $formalRoot 'src'
$defaultModel = Join-Path $formalRoot 'models\bge-m3'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not $env:BGE_M3_MODEL_PATH) {
    $env:BGE_M3_MODEL_PATH = $defaultModel
}
if (-not (Test-Path -LiteralPath $env:BGE_M3_MODEL_PATH -PathType Container)) {
    throw "BGE-M3 model directory not found: $env:BGE_M3_MODEL_PATH"
}

& $python -m uvicorn bge_m3_embedding_service:app --app-dir $sourceDir --host 127.0.0.1 --port $Port
