# Same frozen training script as the main campaign (train_gnn_c0_spatial.py),
# same graph_data_v3 lineage, same split-seed 42,
# same 10 km grid / 2 km buffer. No code change: the only
# difference is the presence/absence/order of --transfer-from.
# Labels: c0c1cfsc_* (no chaining) and c0c1cfinv_* (reversed chain);
# do not collide with c0c1cf_* or earlier campaigns.
$ErrorActionPreference = "Stop"

$PY   = "F:\S33_dslm\s33_amb_virtual\.venv\Scripts\python.exe"
$BASE = "D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
$TREI = Join-Path $BASE "treinos"
$CONG = Join-Path $BASE "dados\scripts_congelados"
$V3   = "F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3"
$LOG  = Join-Path $TREI "fila_controles_c1v2.log"

function L($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -Encoding utf8 $LOG }

function Concluido($runJson) {
    if (-not (Test-Path $runJson)) { return $false }
    try {
        $d = Get-Content $runJson -Raw | ConvertFrom-Json
        return ($null -ne $d.custo.tempo_total_s)
    } catch { return $false }
}

function Treinar($label, $c, $q, $seed, $prev, $ep, $lr) {
    $runJson = Join-Path $TREI "$label\run_$label.json"
    $ckpt = Join-Path $TREI "$label\checkpoints\checkpoint_best.pt"
    if (Concluido $runJson) { L "pula $label (concluido)"; return $ckpt }
    if (Test-Path $runJson) { L "$label tem JSON incompleto - refazendo" }

    $rf = "transfer_dataset_${c}_v19_${q}_enriched_cftudo.pt"
    $gpu = "${c}_v19_${q}_gpu.pt"
    if (-not (Test-Path (Join-Path $V3 $rf)) -or ((Get-Item (Join-Path $V3 $rf)).Length -le 20e9)) {
        L "ABORTA $label : dataset regenerado nao encontrado em graph_data_v3"; return $null
    }
    if (-not (Test-Path (Join-Path $V3 $gpu))) { L "ABORTA $label : falta $gpu"; return $null }

    $argv = @((Join-Path $CONG "train_gnn_c0_spatial.py"),
              "--run-label", $label,
              "--rf-data-file", $rf,
              "--graph-file", $gpu,
              "--graph-dir", $V3,
              "--evid-dir", $TREI,
              "--epochs", $ep, "--lr", $lr,
              "--seed", $seed, "--split-seed", 42,
              "--grid-km", 10.0, "--buffer-km", 2.0,
              "--split-frac", "0.70,0.15,0.15",
              "--freq-mhz", 900, "--diag-freq-mhz", 1800, "--hash")
    if ($prev) { $argv = $argv + @("--transfer-from", $prev) }

    L "$ $label"
    $t0 = Get-Date
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PY -u @argv *>> $LOG
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    L "   rc=$rc em $min min"
    if ($rc -ne 0) { L "  [$label] FALHOU rc=$rc"; return $null }
    L "  [$label] ok em $min min"
    return $ckpt
}

L "===== fila_controles_c1v2 iniciada (pid $PID) ====="

# Control A: no chaining, from-scratch regime (35 epochs, lr 1e-3).
foreach ($c in @("lins", "bauru", "campinas", "sorocaba")) {
    foreach ($seed in @(42, 43)) {
        foreach ($q in @("Q2", "Q3", "Q4")) {
            $null = Treinar "c0c1cfsc_${c}_s${seed}_${q}_g10b2" $c $q $seed "" 35 "1e-3"
        }
    }
}

$null = Treinar "c0c1cfsc_campinas_s44_Q3_g10b2" "campinas" "Q3" 44 "" 35 "1e-3"

# Head of the reversed chain: reuses the Q4 from-scratch checkpoint of control A (seed 42).
$headQ4 = Join-Path $TREI "c0c1cfsc_lins_s42_Q4_g10b2\checkpoints\checkpoint_best.pt"
if (-not (Test-Path $headQ4)) {
    L "[PARE] cabeca Q4 do controle B ausente (controle A falhou?)"
} else {
# Control B: reversed chain Q4->Q3->Q2->Q1 (20-epoch transfers, lr 2e-4).
    $prev = $headQ4
    foreach ($q in @("Q3", "Q2", "Q1")) {
        $ck = Treinar "c0c1cfinv_lins_s42_${q}_g10b2" "lins" $q 42 $prev 20 "2e-4"
        if (-not $ck) { L "[PARE] cadeia invertida interrompida em $q"; break }
        $prev = $ck
    }
}

L "===== fila_controles_c1v2 concluida ====="
