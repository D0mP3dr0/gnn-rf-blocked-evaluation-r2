#!/usr/bin/env bash
# Lote de inferencia por populacao do BA1 (protocolo PROTOCOLO_faseC_baselines §7), padrao de fase_C/avaliacao/lote_inferencia.sh.
# 80 corridas: ba1rus x cidades bauru campinas lins sorocaba x seeds 42-46 x Q1-Q4, UMA por vez, DEPOIS da fila de treino.
# Saida: fase_BA1/e3_predicoes/ba1rus_<cidade>_Q<q>_s<seed>.npz/.json (formato da fase B/C: idx_global, target, pred, sentinela).
# Pula corrida com json (status de fim) + npz validos (fase_C/avaliacao/verificar_saida.py, so leitura). Erro nao aborta o lote.
# NAO LANCADO por quem escreveu. Lancamento sugerido:
#   systemd-run --user --unit=ieee-r4-ba1-inferencia --collect --description="R4 IEEE BA1: inferencia por populacao (80)" \
#     /usr/bin/bash /trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_BA1/avaliacao/lote_inferencia_ba1.sh
set -u
A=/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_BA1/avaliacao
SAIDA=/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_BA1/e3_predicoes
VERIF=/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_C/avaliacao/verificar_saida.py
PY=/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
SCRIPT=$A/e3_inferencia_ba1.py
LOG=$A/lote_inferencia_ba1.log
mkdir -p "$A/logs" "$SAIDA"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "=== LOTE inicio $(ts) pid=$$ script_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)" >> "$LOG"
n_ok=0; n_pulo=0; n_erro=0
for C in bauru campinas lins sorocaba; do
 for S in 42 43 44 45 46; do
  for Q in 1 2 3 4; do
   LAB="ba1rus_${C}_Q${Q}_s${S}"; V=""
   J="$SAIDA/${LAB}.json"; N="$SAIDA/${LAB}.npz"
   if [[ -f "$J" && -f "$N" ]] && V=$("$PY" -I "$VERIF" "$J" "$N" 2>&1); then
     echo "PULO $LAB $(ts) $V" >> "$LOG"; n_pulo=$((n_pulo+1)); continue
   fi
   echo "INICIO $LAB $(ts)" >> "$LOG"
   t0=$(date +%s)
   "$PY" -I -u "$SCRIPT" --celula "${C}_Q${Q}" --braco ba1 --seed "$S" > "$A/logs/${LAB}.log" 2>&1 &
   PID=$!
   echo "RODANDO $LAB pid=$PID $(ts)" >> "$LOG"
   wait "$PID"; RC=$?
   dt=$(( $(date +%s) - t0 ))
   if [[ $RC -eq 0 ]] && V=$("$PY" -I "$VERIF" "$J" "$N" 2>&1); then
     echo "FIM $LAB rc=$RC dt=${dt}s $V" >> "$LOG"; n_ok=$((n_ok+1))
   else
     echo "FIM $LAB rc=$RC dt=${dt}s ERRO ${V} (log: logs/${LAB}.log)" >> "$LOG"; n_erro=$((n_erro+1))
   fi
  done
 done
done
echo "=== LOTE fim $(ts) ok=$n_ok pulo=$n_pulo erro=$n_erro" >> "$LOG"
