# Protocol, same as the GNN queue (c0c1cf campaign): 10 km grid, 2 km buffer,
# split-seed 42, 70/15/15 split, 900 MHz in the loss and 1800 MHz in diagnostics,
# sequential Q1->Q4 transfer within each city (MLP transfers from MLP),
# 35 epochs/lr 1e-3 in Q1 and 20 epochs/lr 2e-4 in the rest, seeds 42-46,
# same 8 graph_data_v3 .pt datasets (Lins and Bauru).
# script: train_mlp_c0_spatial.py; exact parity of 1,812,515 trainable
# parameters with the GNN, checked at runtime by the script itself;
# run label: mlpcf_<city>_s<seed>_<Q>_g10b2 (does not collide with c0c1cf_*);
# no automatic aggregation: the aggregator writes agregado_c1v2_<city>_g10b2.json
# without a distinguishing prefix and would overwrite the GNN aggregate;
# MLP aggregation needs its own wrapper and is deliberately left out of this queue.
param(
    [int[]] $Seeds = @(42, 43, 44, 45, 46),
    [string[]] $Cidades = @("lins", "bauru")
)
$ErrorActionPreference = "Stop"

$PY   = "F:\S33_dslm\s33_amb_virtual\.venv\Scripts\python.exe"
$BASE = "D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
$S    = Join-Path $BASE "scripts"
$TREI = Join-Path $BASE "treinos"
$V3   = "F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3"
$LOG  = Join-Path $TREI "fila_mlp_c1v2.log"
$VERD = Join-Path $BASE "VERDITO_MLP_C1V2.md"

$estado = New-Object System.Collections.Specialized.OrderedDictionary
function L($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -Encoding utf8 $LOG }
function Marcar($k, $v) { $estado[$k] = $v; L "  [$k] $v" }

function Concluido($runJson) {
    if (-not (Test-Path $runJson)) { return $false }
    try {
        $d = Get-Content $runJson -Raw | ConvertFrom-Json
        return ($null -ne $d.custo.tempo_total_s)
    } catch { return $false }
}

# returns @(folder, dataset_name) for the tile, or $null; requires the
# regenerated graph_data_v3 lineage (>20 GB) - aborts the tile rather than
# silently training a mixed lineage.
function Localizar($c, $q) {
    $a = "$V3\transfer_dataset_${c}_v19_${q}_enriched_cftudo.pt"
    if ((Test-Path $a) -and ((Get-Item $a).Length -gt 20e9)) {
        return @($V3, (Split-Path $a -Leaf))
    }
    return $null
}

function Treinar($c, $q, $seed, $prev, $ep, $lr) {
    $label = "mlpcf_${c}_s${seed}_${q}_g10b2"
    $runJson = Join-Path $TREI "$label\run_$label.json"
    $ckpt = Join-Path $TREI "$label\checkpoints\checkpoint_best.pt"
    if (Concluido $runJson) { L "pula $label (concluido)"; return $ckpt }
    if (Test-Path $runJson) { L "$label tem JSON incompleto - refazendo" }

    $loc = Localizar $c $q
    if (-not $loc) { L "ABORTA $label : dataset com alvo corrigido nao encontrado"; return $null }
    $pasta = $loc[0]; $rf = $loc[1]
# gpu.pt: the MLP control does not load it for edges; required here only as
# a feature fallback (when features_raw is absent) and for provenance.
    $gpu = "${c}_v19_${q}_gpu.pt"
    if (-not (Test-Path (Join-Path $pasta $gpu))) { L "ABORTA $label : falta $gpu em $pasta"; return $null }

    $argv = @((Join-Path $BASE "dados\scripts_congelados\train_mlp_c0_spatial.py"),
              "--run-label", $label,
              "--rf-data-file", $rf,
              "--graph-file", $gpu,
              "--graph-dir", $pasta,
              "--evid-dir", $TREI,
              "--epochs", $ep, "--lr", $lr,
              "--seed", $seed, "--split-seed", 42,
              "--grid-km", 10.0, "--buffer-km", 2.0,
              "--split-frac", "0.70,0.15,0.15",
              "--freq-mhz", 900, "--diag-freq-mhz", 1800, "--hash")
    if ($prev) { $argv = $argv + @("--transfer-from", $prev) }

    L "$ $label  (pasta: $(Split-Path $pasta -Leaf))"
    $t0 = Get-Date
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PY -u @argv *>> $LOG
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    L "   rc=$rc em $min min"

    if ($rc -ne 0) { Marcar $label "FALHOU rc=$rc"; return $null }
    Marcar $label "ok em $min min"
    return $ckpt
}

L "===== fila_mlp_c1v2 iniciada (pid $PID) ====="
L ("seeds: " + ($Seeds -join ", ") + " | cidades: " + ($Cidades -join ", "))

foreach ($seed in $Seeds) {
    foreach ($c in $Cidades) {
        L "--- $c seed $seed ---"
        $prev = ""
        foreach ($q in @("Q1", "Q2", "Q3", "Q4")) {
            if ($q -eq "Q1") { $ep = 35; $lr = "1e-3" } else { $ep = 20; $lr = "2e-4" }
            $ck = Treinar $c $q $seed $prev $ep $lr
            if (-not $ck) { L "[PARE] cadeia $c s$seed interrompida em $q"; break }
            $prev = $ck
        }
    }
}

$linhas = New-Object System.Collections.Generic.List[string]
$linhas.Add('# Veredito - controle E1/B4: MLP sem grafo na regua c0c1cf')
$linhas.Add('')
$linhas.Add('Gerado em ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '. Log: ' + $LOG)
$linhas.Add('')
$linhas.Add('| corrida | resultado |')
$linhas.Add('|---|---|')
foreach ($k in $estado.Keys) { $linhas.Add("| $k | $($estado[$k]) |") }
$linhas.Add('')
$linhas.Add('## Runs gravados')
$linhas.Add('')
$linhas.Add('| run | json | checkpoint |')
$linhas.Add('|---|---|---|')
foreach ($r in @(Get-ChildItem $TREI -Directory -Filter "mlpcf_*" -EA SilentlyContinue | Sort-Object Name)) {
    $j = Join-Path $r.FullName "run_$($r.Name).json"
    $ck = Join-Path $r.FullName "checkpoints\checkpoint_best.pt"
    if (Test-Path $j) { $tj = 'sim' } else { $tj = 'NAO' }
    if (Test-Path $ck) { $tc = 'sim' } else { $tc = 'NAO' }
    $linhas.Add("| $($r.Name) | $tj | $tc |")
}
$linhas.Add('')
$linhas.Add('## Leitura obrigatoria antes de qualquer numero virar prosa')
$linhas.Add('')
$linhas.Add('- A comparacao MLP x GNN e pareada por celula: mesmo dataset, mesmo')
$linhas.Add('  split (conferir particoes.test.idx_sha256_global identico ao do run')
$linhas.Add('  c0c1cf da mesma celula), mesma selecao por val.mae_rssi_db.')
$linhas.Add('- Unidade estatistica: cidade (n=2 nesta fila), nunca as 8 celulas -')
$linhas.Add('  Q2-Q4 herdam o checkpoint Q1 da mesma seed (parecer, secao 1).')
$linhas.Add('- Agregacao dos mlpcf_* fica fora desta fila (colisao de nome com o')
$linhas.Add('  agregado do GNN); exige wrapper proprio, contra-auditado.')
$linhas.Add('- Se o MLP empatar com a GNN, isso e um ACHADO que reformula o claim,')
$linhas.Add('  nao um fracasso (parecer, E1 - criterio de sucesso).')
$linhas.Add('')
$linhas.Add('Nenhum numero daqui e citavel antes de contra-auditoria por quem nao')
$linhas.Add('produziu a corrida.')

Set-Content -Path $VERD -Value ($linhas -join "`r`n") -Encoding utf8
L "veredito: $VERD"
L "===== fila_mlp_c1v2 concluida ====="
