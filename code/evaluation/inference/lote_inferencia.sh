#!/usr/bin/env bash
# Lote de inferencia da fase C (protocolo PROTOCOLO_faseC_C1_C2_2026-10-14.md, secao Avaliacao).
# 160 corridas: bracos c1noreg c2nomono x cidades bauru campinas lins sorocaba x seeds 42-46 x Q1-Q4, UMA por vez.
# Pula corrida com json (status de fim) + npz validos (verificar_saida.py). Erro de uma corrida e' registrado e o lote segue.
set -u
A=/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_C/avaliacao
SAIDA=/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_C/e3_predicoes
PY=/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
SCRIPT=$A/e3_inferencia_faseC.py
LOG=$A/lote_inferencia.log
mkdir -p "$A/logs" "$SAIDA"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "=== LOTE inicio $(ts) pid=$$ script_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)" >> "$LOG"
n_ok=0; n_pulo=0; n_erro=0
for BR in c1noreg c2nomono; do
 for C in bauru campinas lins sorocaba; do
  for S in 42 43 44 45 46; do
   for Q in 1 2 3 4; do
    LAB="${BR}_${C}_Q${Q}_s${S}"; V=""
    J="$SAIDA/${LAB}.json"; N="$SAIDA/${LAB}.npz"
    if [[ -f "$J" && -f "$N" ]] && V=$("$PY" "$A/verificar_saida.py" "$J" "$N" 2>&1); then
      echo "PULO $LAB $(ts) $V" >> "$LOG"; n_pulo=$((n_pulo+1)); continue
    fi
    echo "INICIO $LAB $(ts)" >> "$LOG"
    t0=$(date +%s)
    "$PY" -u "$SCRIPT" --celula "${C}_Q${Q}" --braco gnn --seed "$S" --prefixo-run "$BR" > "$A/logs/${LAB}.log" 2>&1 &
    PID=$!
    echo "RODANDO $LAB pid=$PID $(ts)" >> "$LOG"
    wait "$PID"; RC=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ $RC -eq 0 ]] && V=$("$PY" "$A/verificar_saida.py" "$J" "$N" 2>&1); then
      echo "FIM $LAB rc=$RC dt=${dt}s $V" >> "$LOG"; n_ok=$((n_ok+1))
    else
      echo "FIM $LAB rc=$RC dt=${dt}s ERRO ${V} (log: logs/${LAB}.log)" >> "$LOG"; n_erro=$((n_erro+1))
    fi
   done
  done
 done
done
echo "=== LOTE fim $(ts) ok=$n_ok pulo=$n_pulo erro=$n_erro" >> "$LOG"
