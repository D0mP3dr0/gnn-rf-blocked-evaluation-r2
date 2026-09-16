# The script aborts on its own if the reproduced split does not match the
# idx_sha256_global of the campaign's run JSON.
# Resumable: skips the tile if the output JSON already has a 'concluido_em' (completed) timestamp.
$ErrorActionPreference = "Stop"

$PY   = "F:\S33_dslm\s33_amb_virtual\.venv\Scripts\python.exe"
$BASE = "D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
$SCR  = Join-Path $BASE "scripts\baselines_v2_por_particao.py"
$OUTD = Join-Path $BASE "dados\baselines_v2"
$LOG  = Join-Path $BASE "treinos\fila_baselines_v2.log"

function L($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -Encoding utf8 $LOG }

function Concluido($json) {
    if (-not (Test-Path $json)) { return $false }
    try {
        $d = Get-Content $json -Raw | ConvertFrom-Json
        return ($null -ne $d.concluido_em)
    } catch { return $false }
}

L "===== fila_baselines_v2 iniciada (pid $PID) ====="

foreach ($c in @("lins", "bauru", "campinas", "sorocaba")) {
    foreach ($q in @("Q1", "Q2", "Q3", "Q4")) {
        $out = Join-Path $OUTD "baselines_v2_${c}_${q}.json"
        if (Concluido $out) { L "pula ${c}_${q} (concluido)"; continue }
        L "$ baselines_v2 ${c}_${q}"
        $t0 = Get-Date
        $eap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PY -u $SCR --cidade $c --quadrante $q *>> $LOG
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $eap
        $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($rc -ne 0) { L "  [${c}_${q}] FALHOU rc=$rc em $min min" }
        else { L "  [${c}_${q}] ok em $min min" }
    }
}

L "===== fila_baselines_v2 concluida ====="
