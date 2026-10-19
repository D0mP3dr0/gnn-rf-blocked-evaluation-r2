#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gnn_c0_spatial.py — itens C0 e C1 do ROADMAP_RESUBMISSAO.md (Access-2026-29524).

ARQUIVO NOVO. Nada em GNN_RF_V2 é modificado por este script; ele apenas LÊ
(modelo, loss, métricas, spatial_cv) e ESCREVE em
FIRST_RESPONSE_REVIEW_IEEE_ACESSES\\EVIDENCIA_RESUBMISSAO\\treinos\\<run_label>\\.

=============================================================================
PROVENIÊNCIA
=============================================================================
origem            : D:\\_ARQUIVO_SSD_F\\TOPO_RF\\GNN_RF_V2\\train_lins_physics.py
origem_sha256     : 13ea2ff0fcaca2776d8c0f3603b122efa509854c19fdac7e1d5d5e7d128826e1
origem_bytes      : 55313
metodo            : cópia + patches documentados (lista abaixo)
data              : 2026-09-01

=============================================================================
C0 — CORREÇÕES ANTES DE TREINAR (diff vs origem, arquivo:linha do ORIGINAL)
=============================================================================

[C0-1] train_lins_physics.py:857-863  — termo FSPL da loss recebia distância errada.
    ORIGINAL:
        distances = (ea[:n_seed, 0] * 15000.0 if ea.shape[0] >= n_seed else None)
    DOIS DEFEITOS INDEPENDENTES:
      (a) ESCALA. O ETL normaliza a distância da aresta por 30000, não por 15000
          (01_data/prepare_transfer_dataset_v19.py:256 —
           `ea_flat = np.stack([dist_flat / 30_000.0, freq_flat / 2600.0], axis=1)`).
          O fator 15000 devolvia METADE da distância real, deslocando o piso FSPL
          em 20*log10(0.5) = -6,02 dB (piso frouxo por 6 dB).
      (b) INDEXAÇÃO. `ea` é um tensor de ARESTAS (E, 2); `preds`/`targets` são de
          NÓS-SEMENTE (n_seed, 5). `ea[:n_seed, 0]` fatia as n_seed PRIMEIRAS
          ARESTAS do batch e as parea, posição a posição, com os n_seed primeiros
          NÓS. Não existe alinhamento entre as duas ordens: o vetor entregue ao
          `_fspl_constraint` era uma permutação arbitrária de distâncias de outros
          nós. O termo virou ruído, não física.
    CORREÇÃO:
        distances = dist_to_ant_batch   # metros, alinhado nó-a-nó com preds
    `dist_to_ant_batch` já era computado no original (:865-869) como
    `dist_all[n_id[:n_seed]]` e já era passado como `dist_to_ant=` (penalidade de
    gradiente). Agora é a MESMA grandeza usada nos dois termos físicos.
    NOTA DE SEMÂNTICA: `dist_all` é a distância à antena MAIS PRÓXIMA, enquanto o
    alvo de path loss vem da antena de MAIOR RSSI. Como d_nearest <= d_best,
    FSPL(d_nearest) <= FSPL(d_best): o piso usado é o mais FROUXO dos dois, logo
    a restrição nunca gera violação falsa. Escolha conservadora, declarada.
    Teste unitário do termo: `--selftest` (bloco T1/T2/T3).

[C0-2] train_lins_physics.py:847-994 — métricas diagnósticas desalinhadas (achado A8).
    ORIGINAL: `all_preds`/`all_tgts` acumulavam `preds` inteiro (n_seed linhas)
    e `diag.compute` recebia N != n_terrain; RFDiagnosticMetrics então reamostrava
    por `torch.linspace` (03_training/rf_diagnostic_metrics.py:206-214), o que
    pareia predição do nó i com a máscara espacial do nó j. Consequência medida:
    `physics_distance_gradient` ~ 0 (correlação entre vetores desalinhados).
    CORREÇÃO: acumular somente as SEMENTES (preds[:bs], targets[:bs], n_id[:bs]
    com bs = batch["terrain"].batch_size) e, antes de `diag.compute`, ESPALHAR as
    predições num array ordenado por nó (`_scatter_por_semente`), com máscara de
    nós visitados. Só entram no cálculo os nós efetivamente visitados na época;
    a instância de RFDiagnosticMetrics é construída sobre exatamente esses nós
    (`_DiagCache`), de modo que N == n_mask e a reamostragem do linspace NUNCA
    dispara. Referência esperada: physics_distance_gradient ~ 0,1-0,2
    (fase0_metrics_bauru_s42_Q1.json / rem_gnn: 0,15758708119392395).

[C0-3] train_lins_physics.py:361 — `base_dir` hardcoded em F:\.
    ORIGINAL: base_dir = Path(r"f:\\arpia_topo_refinado\\TOPO_RF\\GNN_RF_V2")
    CORREÇÃO: argumento `--base-dir`, default
    D:\\_ARQUIVO_SSD_F\\TOPO_RF\\GNN_RF_V2. `--base-dir` é usado APENAS para
    importar os módulos (02_models, 03_training, 04_baselines) e localizar
    graph_data; NENHUMA saída é escrita lá. Saídas vão para `--evid-dir`
    (default EVIDENCIA_RESUBMISSAO\\treinos), jamais para GNN_RF_V2\\logs.

[C0-4] FREQUÊNCIA DO FSPL — divergência 900/1800 MHz, agora explícita.
    ORIGINAL train_lins_physics.py:682-688 instancia CurriculumRFLoss SEM
    `frequency_mhz`; o default herdado de PhysicsRFLoss.__init__
    (02_models/physics_loss.py:36) é 900.0 MHz. O termo FSPL da LOSS rodou
    portanto a 900 MHz. Ao mesmo tempo, RFDiagnosticMetrics usa
    FREQUENCY_MHZ = 1800.0 (03_training/rf_diagnostic_metrics.py:38) para o
    `physics_fspl_compliance`, e `--freq-mhz` (default 900.0) alimenta os
    baselines analíticos. Ou seja: a loss penalizava contra um piso de 900 MHz e
    a métrica auditava contra um piso de 1800 MHz (6,02 dB acima).
    CORREÇÃO: `--freq-mhz` (default 900.0) é passado EXPLICITAMENTE à
    CurriculumRFLoss e aos baselines; `--diag-freq-mhz` (default 1800.0) é
    passado explicitamente a RFDiagnosticMetrics. Os dois valores efetivos são
    gravados no JSON (`config.freq_mhz`, `config.diag_freq_mhz`) e a distribuição
    empírica de frequência das antenas do dataset (edge_attr[:,1]*2600) é
    reportada em `dataset.freq_antenas_mhz` para que o revisor confira a escolha.

[C0-5] AMP — `torch.cuda.amp.GradScaler` / `torch.autocast(device_type=...)`
    substituídos pela API não-depreciada `torch.amp.GradScaler("cuda")` /
    `torch.autocast("cuda")` (torch 2.10). Sem efeito numérico.

=============================================================================
C1 — AVALIAÇÃO ESPACIALMENTE BLOQUEADA (R1-1)
=============================================================================

[C1-5] Split por blocos, com a lógica de 03_training/spatial_cv.py
    (SpatialKFold, grid 5 km, buffer 2 km), com DUAS adaptações obrigatórias:
    (a) PROJEÇÃO. `terrain.pos` está em GRAUS (lon, lat) — spatial_cv.py:60
        assume metros (`pos_km = pos / 1000.0`). Usar graus ali daria blocos de
        5000 graus (um único bloco) e buffer de 2000 graus. Conversão prévia com
        a MESMA equiretangular do ETL
        (01_data/prepare_transfer_dataset_v19.py:181-184):
            dy = (lat - lat_min) * 111000
            dx = (lon - lon_min) * 111000 * cos(lat)     <- cos da lat DO NÓ
        Usa-se cos φ por nó (e não cos da latitude média) exatamente porque é a
        convenção com que as distâncias nó-antena do dataset foram geradas: o
        buffer passa a ser medido na mesma métrica do dado.
    (b) TRÊS partições por GRUPOS DE BLOCOS (default 70/15/15, `--split-seed 42`).
        `SpatialKFold._assign_groups` é reusado tal como está para atribuir o
        bloco de cada nó; a divisão em três conjuntos e o buffer de três vias são
        implementados aqui (spatial_cv.py só faz 2 vias).
    BUFFER DE TRÊS VIAS (generalização do spatial_cv.py:105-138, que apara só a
    validação contra o treino): o treino mantém seus blocos intactos; a validação
    perde os nós a menos de `--buffer-km` de qualquer nó de treino; o teste perde
    os nós a menos de `--buffer-km` de qualquer nó de treino OU da validação já
    aparada. Resultado: as três distâncias mínimas par-a-par >= buffer.
    O TESTE NUNCA É USADO EM SELEÇÃO: `_selecionar_melhor_epoca()` recebe apenas
    a série de val e levanta AssertionError se receber chave de teste.

[C1-6] SEM ARESTAS CRUZANDO A FRONTEIRA NO FITTING.
    Cada partição vira um HeteroData próprio, subgrafo INDUZIDO pelos seus nós
    terrain (+ TODAS as antenas), com as 3 relações reconstruídas por
    torch_geometric.utils.subgraph / bipartite_subgraph:
        ("antenna","propagates_to","terrain") : bipartite_subgraph
        ("terrain","in_range_of","antenna")   : reverso do anterior (mesmo par)
        ("terrain","connects_to","terrain")   : subgraph
    Consequência declarada: a INFERÊNCIA em val e em test usa exclusivamente o
    contexto de vizinhança da própria região — nenhuma mensagem chega de um nó
    de treino, em nenhuma das 4 camadas do encoder. O número de arestas
    descartadas por relação e por partição é gravado no JSON
    (`subgrafos.<part>.arestas_removidas_*`), e a contagem de arestas que
    cruzam fronteira nos grafos induzidos é verificada e gravada como 0.

[C1-7] SAÍDA: um JSON por run em EVIDENCIA_RESUBMISSAO\\treinos\\<run_label>\\
    (`run_<label>.json`) no padrão do manifest.jsonl: contagens por partição,
    sha256 dos índices GLOBAIS de cada partição, MAE/RMSE de RSSI e de PL
    (alvo < 299 dB) por partição por época, config efetiva completa, sha256 do
    próprio script e (com `--hash`) do dataset, seed, ambiente e determinismo.

=============================================================================
DECISÕES DE PROJETO ALÉM DA ESPECIFICAÇÃO (todas registradas no JSON)
=============================================================================
[D1] Estatísticas de normalização da feature col 17 (dist_to_antenna) passam a
     ser calculadas SOMENTE no treino (original: sobre todos os nós,
     train_lins_physics.py:569-571). Usar média/desvio do conjunto completo é
     padronização com estatística de teste — vazamento leve, mas é exatamente o
     tipo de coisa que o Revisor 1 (ponto 1) está caçando. Ambos os valores são
     gravados (`normalizacao.dist_mean_train` e `..._todos`).
[D2] Calibração dos baselines analíticos (P_tx_eff = mediana(rssi+PL), original
     train_lins_physics.py:192) passa a ser feita SÓ com alvos de TREINO e
     aplicada inalterada a val e test. Calibrar no próprio conjunto avaliado dá
     vantagem indevida ao baseline e quebra a paridade da comparação.
[D3] Limiares de landcover (p75 roughness / p90 slope,
     train_lins_physics.py:260-261) calculados no TREINO e aplicados às três
     partições, para que as classes sejam as mesmas nos três conjuntos.
[D4] O teste é avaliado a cada época (`--test-every 1`, atende à especificação)
     E, ao final, novamente sobre o checkpoint best-por-val recarregado
     (`selecao.test_no_melhor_ckpt`). O primeiro é diagnóstico; o segundo é o
     número reportável. Ambos rotulados no JSON.
[D5] `--smoke` reduz grid/buffer proporcionalmente quando a subamostra é pequena
     demais para 5 km de bloco (senão a janela do smoke tem 2-4 blocos e o split
     70/15/15 é degenerado). O fato é logado e gravado em
     `geometria.grid_km_ajustado_por_smoke`. A geometria do smoke NÃO é a de
     produção e não deve gerar número citável.
[D6] `--window-anchor cobertura` (default): a janela de `--max-nodes` é centrada
     na mediana dos nós COM alvo de PL válido, não na mediana de todos os nós.
     Descoberto rodando o smoke: em Bauru Q1 só 16,2 % dos nós (2.101.544 de
     12.960.000) têm enlace explícito, e a mediana geométrica de um quadrante de
     103x111 km cai em campo aberto — a primeira execução
     (run c0c1_smoke_bauru_Q1_s42) saiu com 0 arestas antena->terreno e 0 alvos
     válidos nas 3 partições, terminou com exit 0 e MAE_RSSI de 0,72 dB, que era
     regressão sobre o sentinela constante de -110 dBm. Guarda associada:
     `guarda_particao_degenerada` levanta RuntimeError se qualquer partição sair
     com 0 arestas antena->terreno — "não crashou" não é "funciona".
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import bipartite_subgraph, subgraph

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------
# Constantes herdadas do original (train_lins_physics.py:62-66)
# --------------------------------------------------------------------------
COL_ELEV, COL_SLOPE, COL_ROUGH, COL_NDVI, COL_NDWI = 0, 1, 6, 12, 13

RSSI_THRESHOLD_DBM = -100.0
RSSI_THRESHOLD_CONF_DBM = -85.0

# Sentinela do alvo de path loss: prepare_transfer_dataset_v19 satura o PL em
# 300 dB para nós sem enlace. "Alvo válido" = PL < 299 dB (pedido do C1-7).
PL_TARGET_MAX_VALID = 299.0

ORIGEM_PATH = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\train_lins_physics.py"
ORIGEM_SHA256 = "13ea2ff0fcaca2776d8c0f3603b122efa509854c19fdac7e1d5d5e7d128826e1"

DEFAULT_BASE_DIR = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2"
DEFAULT_EVID_DIR = (r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
                    r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO\treinos")


# ==========================================================================
# parse_args
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="C0+C1: treino GNN-RF com correcao da loss e split espacial bloqueado.")
    # --- herdados do original ---
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24576)
    p.add_argument("--k-antenna", type=int, default=20)
    p.add_argument("--k-terrain", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--version", type=int, default=21)
    p.add_argument("--run-label", type=str, default="c0c1_smoke")
    p.add_argument("--variance-weight", type=float, default=0.02)
    p.add_argument("--ndvi-weight", type=float, default=0.03)
    p.add_argument("--dist-grad-weight", type=float, default=0.05)
    p.add_argument("--max-norm", type=float, default=0.5)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["plateau", "cosine", "cosine_simple"])
    p.add_argument("--scheduler-patience", type=int, default=2)
    p.add_argument("--cosine-t0", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--transfer-from", type=str, default="")
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--p-tx-dbm", type=float, default=43.0)
    p.add_argument("--graph-file", type=str, default="bauru_v19_Q1_gpu.pt")
    p.add_argument("--graph-dir", type=str, default="")
    p.add_argument("--rf-data-file", type=str,
                   default="transfer_dataset_bauru_v19_Q1_enriched.pt")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # --- C0 ---
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR,
                   help="[C0-3] Raiz do GNN_RF_V2 usada SO PARA LEITURA (modulos + graph_data).")
    p.add_argument("--evid-dir", type=str, default=DEFAULT_EVID_DIR,
                   help="[C0-3] Raiz de saida (EVIDENCIA_RESUBMISSAO/treinos).")
    p.add_argument("--freq-mhz", type=float, default=900.0,
                   help="[C0-4] Frequencia (MHz) do termo FSPL da LOSS e dos baselines analiticos.")
    p.add_argument("--diag-freq-mhz", type=float, default=1800.0,
                   help="[C0-4] Frequencia (MHz) do FSPL de RFDiagnosticMetrics (physics_fspl_compliance).")

    # --- C1 ---
    p.add_argument("--grid-km", type=float, default=5.0,
                   help="[C1-5] Lado do bloco espacial (km). Default do spatial_cv.py.")
    p.add_argument("--buffer-km", type=float, default=2.0,
                   help="[C1-5] Buffer entre particoes (km). Default do spatial_cv.py.")
    p.add_argument("--split-frac", type=str, default="0.70,0.15,0.15",
                   help="[C1-5] Fracao de BLOCOS train,val,test.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="[C1-5] Seed do embaralhamento de blocos (independente de --seed).")
    p.add_argument("--test-every", type=int, default=1,
                   help="[D4] Avaliar teste a cada N epocas (0 = so no final). Nunca usado em selecao.")
    p.add_argument("--eval-batch-size", type=int, default=0,
                   help="Batch de inferencia (0 = igual a --batch-size).")

    # --- operacao ---
    p.add_argument("--max-nodes", type=int, default=0,
                   help="Subamostra espacialmente CONTIGUA de ate N nos (0 = todos).")
    p.add_argument("--window-anchor", type=str, default="cobertura",
                   choices=["cobertura", "mediana"],
                   help="Centro da janela de --max-nodes: 'cobertura' = mediana dos nos "
                        "com alvo de PL valido (evita janela sem antena); 'mediana' = "
                        "mediana de todos os nos (comportamento ingenuo).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 2 epocas, --max-nodes 200000 se nao dado, workers=0, verificacoes extras.")
    p.add_argument("--selftest", action="store_true",
                   help="Roda so os testes unitarios do termo de distancia (C0-1) e sai.")
    p.add_argument("--hash", action="store_true",
                   help="Calcular sha256 do dataset (lento: ~28 GB).")
    p.add_argument("--mmap", action="store_true",
                   help="torch.load(mmap=True) — util no smoke para nao materializar 28 GB.")
    p.add_argument("--no-baselines", action="store_true")
    return p.parse_args(argv)


# ==========================================================================
# Helpers gerais
# ==========================================================================
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    """Hash canonico de um conjunto de indices: int64, ordenado, little-endian."""
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def log_vram() -> str:
    if torch.cuda.is_available():
        return (f"VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB/"
                f"{torch.cuda.memory_reserved()/1024**3:.2f}GB")
    return "CPU"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return str(o)


# ==========================================================================
# [C1-5a] Projecao equiretangular — MESMA convencao do ETL
#         (01_data/prepare_transfer_dataset_v19.py:181-184)
# ==========================================================================
def latlon_graus_para_metros(pos: torch.Tensor) -> tuple[np.ndarray, dict]:
    """
    pos: (N,2) float, col0 = lon (graus), col1 = lat (graus).
    Retorna (N,2) float64 em METROS, origem no canto (lon_min, lat_min).

        dy = (lat - lat_min) * 111000
        dx = (lon - lon_min) * 111000 * cos(lat_do_no)
    """
    lon = pos[:, 0].double().numpy()
    lat = pos[:, 1].double().numpy()
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    meta = {
        "convencao": "equiretangular do ETL (prepare_transfer_dataset_v19.py:181-184)",
        "deg_para_m": 111000.0,
        "cos_phi": "por no (cos da latitude do proprio no)",
        "lon_min_deg": lon_min, "lat_min_deg": lat_min,
        "lon_max_deg": float(lon.max()), "lat_max_deg": float(lat.max()),
        "extensao_x_km": float((x.max() - x.min()) / 1000.0),
        "extensao_y_km": float((y.max() - y.min()) / 1000.0),
    }
    return np.stack([x, y], axis=1), meta


def janela_contigua(pos_m: np.ndarray, max_nodes: int,
                    ancora: np.ndarray | None = None) -> np.ndarray:
    """
    Subamostra espacialmente CONTIGUA: janela quadrada.

    `ancora` (bool, N) define o centro. Sem ancora, o centro e a mediana de
    TODOS os nos — o que, em Bauru Q1 (103x111 km, cobertura de antena em
    ~16 % dos nos e concentrada), cai em campo aberto SEM NENHUMA aresta
    antena->terreno: o smoke roda, mas nao exercita o caminho de mensagem da
    antena nem a metrica de PL (alvo todo sentinela). Ancorar na mediana dos
    nos COBERTOS evita a janela degenerada. Medido no run
    c0c1_smoke_bauru_Q1_s42 (2026-09-01): 199.676 nos, 0 arestas ant->ter.
    """
    n = pos_m.shape[0]
    if max_nodes <= 0 or n <= max_nodes:
        return np.arange(n, dtype=np.int64)
    x, y = pos_m[:, 0], pos_m[:, 1]
    if ancora is not None and bool(ancora.any()):
        cx, cy = float(np.median(x[ancora])), float(np.median(y[ancora]))
    else:
        cx, cy = float(np.median(x)), float(np.median(y))
    lo, hi = 0.0, float(max(x.max() - x.min(), y.max() - y.min()))
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        cnt = int(((np.abs(x - cx) <= mid) & (np.abs(y - cy) <= mid)).sum())
        if cnt > max_nodes:
            hi = mid
        else:
            lo = mid
    sel = np.where((np.abs(x - cx) <= lo) & (np.abs(y - cy) <= lo))[0]
    return sel.astype(np.int64)


# ==========================================================================
# [C1-5b] Split de tres vias por blocos, com buffer de tres vias
# ==========================================================================
def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                         fracs: tuple[float, float, float], split_seed: int,
                         SpatialKFold, log) -> tuple[dict, dict]:
    """
    Usa SpatialKFold._assign_groups (03_training/spatial_cv.py:94-103) para
    atribuir blocos; divide BLOCOS em 3 grupos; apara val contra train e test
    contra (train U val aparado).
    """
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    skf = SpatialKFold(n_splits=3, buffer_km=buffer_km,
                       grid_size_km=grid_km, random_state=split_seed)
    group_ids = skf._assign_groups(pos_km)          # reuso literal da logica da casa
    grupos = np.unique(group_ids)
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)

    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr = grupos_emb[:n_tr]
    g_va = grupos_emb[n_tr:n_tr + n_va]
    g_te = grupos_emb[n_tr + n_va:]

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)
    n_va_bruto, n_te_bruto = int(m_va.sum()), int(m_te.sum())

    log(f"  Blocos {grid_km:.3f} km: {n_g} total -> train {len(g_tr)} | "
        f"val {len(g_va)} | test {len(g_te)}")
    log(f"  Nos por bloco (antes do buffer): train {int(m_tr.sum()):,} | "
        f"val {n_va_bruto:,} | test {n_te_bruto:,}")

    kw = dict(compact_nodes=False, balanced_tree=False)
    if buffer_km > 0:
        # val aparado contra train
        if m_tr.any() and m_va.any():
            tree_tr = cKDTree(pos_km[m_tr], **kw)
            d, _ = tree_tr.query(pos_km[m_va], k=1, workers=-1)
            idx_va = np.where(m_va)[0]
            m_va[idx_va[d < buffer_km]] = False
        # test aparado contra train U val(aparado)
        m_trva = m_tr | m_va
        if m_trva.any() and m_te.any():
            tree_trva = cKDTree(pos_km[m_trva], **kw)
            d, _ = tree_trva.query(pos_km[m_te], k=1, workers=-1)
            idx_te = np.where(m_te)[0]
            m_te[idx_te[d < buffer_km]] = False

    parts = {"train": np.where(m_tr)[0].astype(np.int64),
             "val": np.where(m_va)[0].astype(np.int64),
             "test": np.where(m_te)[0].astype(np.int64)}

    info = {
        "grid_km": grid_km, "buffer_km": buffer_km,
        "n_blocos_total": int(n_g),
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)), "test": int(len(g_te))},
        "fracs_blocos_pedidas": list(fracs),
        "split_seed": split_seed,
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "n_nos_apos_buffer": {k: int(len(v)) for k, v in parts.items()},
        "retencao_apos_buffer": {
            "val": (len(parts["val"]) / n_va_bruto) if n_va_bruto else None,
            "test": (len(parts["test"]) / n_te_bruto) if n_te_bruto else None},
        "n_descartados_pelo_buffer": int(n_va_bruto - len(parts["val"])
                                         + n_te_bruto - len(parts["test"])),
        "regra_buffer": ("train intacto; val aparado contra train; "
                         "test aparado contra train U val(aparado)"),
    }
    return parts, info


def verificar_split(parts: dict, pos_m: np.ndarray, buffer_km: float,
                    validate_spatial_cv, log) -> dict:
    """
    Verificacao EXATA (cKDTree sobre todos os pontos) + reuso de
    validate_spatial_cv (03_training/spatial_cv.py:141) sobre subamostra.
    """
    from scipy.spatial import cKDTree
    pos_km = pos_m / 1000.0
    out = {"buffer_km_exigido": buffer_km, "pares": {}, "intersecoes": {}, "ok": True}

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ia, ib = parts[a], parts[b]
        inter = int(np.intersect1d(ia, ib, assume_unique=True).size)
        out["intersecoes"][f"{a}|{b}"] = inter
        if inter:
            out["ok"] = False
        if len(ia) and len(ib):
            tree = cKDTree(pos_km[ia], compact_nodes=False, balanced_tree=False)
            d, _ = tree.query(pos_km[ib], k=1, workers=-1)
            dmin = float(d.min())
        else:
            dmin = float("nan")
        out["pares"][f"{a}|{b}"] = {"dist_min_km": dmin,
                                    "respeita_buffer": bool(dmin >= buffer_km)}
        if not (dmin >= buffer_km):
            out["ok"] = False
        log(f"  [SPLIT] {a} x {b}: intersecao={inter} | dist_min={dmin:.3f} km "
            f"({'OK' if dmin >= buffer_km else 'VIOLADO'})")

    # reuso da funcao da casa (subamostra: cdist e O(n*m))
    rs = np.random.RandomState(0)
    sub_tr = parts["train"][rs.choice(len(parts["train"]),
                                      min(20000, len(parts["train"])), replace=False)]
    out["validate_spatial_cv_casa"] = {}
    for b in ("val", "test"):
        if len(parts[b]) == 0:
            out["validate_spatial_cv_casa"][b] = None
            continue
        sub_b = parts[b][rs.choice(len(parts[b]), min(20000, len(parts[b])), replace=False)]
        ok = bool(validate_spatial_cv(sub_tr, sub_b, pos_m, buffer_km=buffer_km, verbose=False))
        out["validate_spatial_cv_casa"][b] = ok
        out["ok"] = out["ok"] and ok
        log(f"  [SPLIT] validate_spatial_cv(train~{len(sub_tr)}, {b}~{len(sub_b)}) = {ok}")
    out["nota_validate_spatial_cv"] = ("executada sobre subamostra de <=20k por conjunto "
                                       "(cdist e O(n*m)); a verificacao exaustiva e a "
                                       "cKDTree acima")
    return out


# ==========================================================================
# [C1-6] Subgrafo induzido por particao
# ==========================================================================
ET_AT = ("antenna", "propagates_to", "terrain")
ET_TA = ("terrain", "in_range_of", "antenna")
ET_TT = ("terrain", "connects_to", "terrain")


def induzir_particao(base: HeteroData, idx_t: torch.Tensor, n_antenna: int,
                     log, nome: str) -> tuple[HeteroData, dict]:
    """
    Subgrafo induzido pelos nos terrain `idx_t` (+ TODAS as antenas).
    Nenhuma aresta cruza a fronteira da particao.
    """
    n_ter_base = base["terrain"].x.shape[0]
    idx_t = idx_t.long()
    ant_all = torch.arange(n_antenna, dtype=torch.long)

    d = HeteroData()
    d["terrain"].x = base["terrain"].x[idx_t]
    d["terrain"].pos = base["terrain"].pos[idx_t]
    d["terrain"].rf_targets = base["terrain"].rf_targets[idx_t]
    d["antenna"].x = base["antenna"].x
    d["antenna"].num_nodes = n_antenna

    ei_at = base[ET_AT].edge_index.long()
    ea_at = getattr(base[ET_AT], "edge_attr", None)
    ei_at_new, ea_at_new = bipartite_subgraph(
        (ant_all, idx_t), ei_at, ea_at, relabel_nodes=True,
        size=(n_antenna, n_ter_base))
    d[ET_AT].edge_index = ei_at_new
    if ea_at_new is not None:
        d[ET_AT].edge_attr = ea_at_new
    d[ET_TA].edge_index = ei_at_new[[1, 0]]

    ei_tt = base[ET_TT].edge_index.long()
    ea_tt = getattr(base[ET_TT], "edge_attr", None)
    ei_tt_new, ea_tt_new = subgraph(idx_t, ei_tt, ea_tt, relabel_nodes=True,
                                    num_nodes=n_ter_base)
    d[ET_TT].edge_index = ei_tt_new
    d[ET_TT].edge_attr = (ea_tt_new if ea_tt_new is not None
                          else torch.zeros((ei_tt_new.shape[1], 2), dtype=torch.float32))

    n_p = int(idx_t.numel())
    info = {
        "n_terrain": n_p, "n_antenna": int(n_antenna),
        "n_arestas_ant_ter": int(ei_at_new.shape[1]),
        "n_arestas_ter_ter": int(ei_tt_new.shape[1]),
        "n_arestas_ant_ter_base": int(ei_at.shape[1]),
        "n_arestas_ter_ter_base": int(ei_tt.shape[1]),
        "arestas_removidas_ant_ter": int(ei_at.shape[1] - ei_at_new.shape[1]),
        "arestas_removidas_ter_ter": int(ei_tt.shape[1] - ei_tt_new.shape[1]),
        # verificacao explicita: nenhum endpoint fora de [0, n_p)
        "arestas_cruzando_fronteira": int(
            (ei_at_new[1] >= n_p).sum() + (ei_tt_new[0] >= n_p).sum()
            + (ei_tt_new[1] >= n_p).sum()),
    }
    log(f"  [{nome}] {n_p:,} nos terrain | ant->ter {info['n_arestas_ant_ter']:,} "
        f"(-{info['arestas_removidas_ant_ter']:,}) | ter->ter "
        f"{info['n_arestas_ter_ter']:,} (-{info['arestas_removidas_ter_ter']:,}) | "
        f"cruzando fronteira: {info['arestas_cruzando_fronteira']}")
    return d, info


# ==========================================================================
# [C0-2] Acumulacao por semente + scatter por no
# ==========================================================================
class _DiagCache:
    """
    Fabrica de RFDiagnosticMetrics restrita a um subconjunto de nos, para que
    N == n_mask em diag.compute() e a reamostragem por linspace
    (rf_diagnostic_metrics.py:206-214) NUNCA dispare.
    """

    def __init__(self, RFDiagnosticMetrics, pos, feats, dist, antenna_pos, freq_mhz):
        self._cls = RFDiagnosticMetrics
        self._pos, self._feats, self._dist = pos, feats, dist
        self._ant = antenna_pos
        self._freq = freq_mhz
        self._cache = {}

    def get(self, key: str, sub: torch.Tensor | None):
        if key in self._cache:
            return self._cache[key]
        if sub is None:
            obj = self._cls(terrain_pos=self._pos, antenna_pos=self._ant,
                            terrain_features=self._feats,
                            frequency_mhz=self._freq, dist_nearest_m=self._dist)
        else:
            obj = self._cls(terrain_pos=self._pos[sub], antenna_pos=self._ant,
                            terrain_features=self._feats[sub],
                            frequency_mhz=self._freq, dist_nearest_m=self._dist[sub])
        self._cache[key] = obj
        return obj


def _scatter_por_semente(seed_ids: torch.Tensor, preds: torch.Tensor,
                         tgts: torch.Tensor, n_nodes: int):
    """
    [C0-2] Espalha as predicoes das sementes num array ORDENADO POR NO.
    Retorna (preds_ord, tgts_ord, visto) com preds_ord[i] = predicao do no i.
    """
    preds_ord = torch.zeros((n_nodes, preds.shape[1]), dtype=torch.float32)
    tgts_ord = torch.zeros((n_nodes, tgts.shape[1]), dtype=torch.float32)
    visto = torch.zeros(n_nodes, dtype=torch.bool)
    preds_ord[seed_ids] = preds.float()
    tgts_ord[seed_ids] = tgts.float()
    visto[seed_ids] = True
    return preds_ord, tgts_ord, visto


def metricas_particao(preds_ord, tgts_ord, visto, diag_cache, chave, pl_valid_full):
    """
    MAE/RMSE de RSSI (todos os nos vistos) e de PL restrito a alvo < 299 dB,
    mais a linha completa de RFDiagnosticMetrics alinhada nó-a-nó.
    """
    n_nodes = visto.numel()
    cobertura = float(visto.float().mean())
    if cobertura >= 1.0:
        sub = None
        key = f"{chave}:full"
        p, t = preds_ord, tgts_ord
    else:
        sub = torch.where(visto)[0]
        key = f"{chave}:{int(sub.numel())}:{int(sub[0])}:{int(sub[-1])}"
        p, t = preds_ord[sub], tgts_ord[sub]

    err_rssi = (p[:, 3] - t[:, 3])
    pl_valid = pl_valid_full if sub is None else pl_valid_full[sub]
    err_pl = (p[:, 0] - t[:, 0])[pl_valid]

    m = {
        "n_nos": int(n_nodes),
        "n_nos_avaliados": int(p.shape[0]),
        "cobertura_epoca": cobertura,
        "mae_rssi_db": float(err_rssi.abs().mean()),
        "rmse_rssi_db": float(torch.sqrt((err_rssi ** 2).mean())),
        "bias_rssi_db": float(err_rssi.mean()),
        "p90_rssi_db": float(np.percentile(err_rssi.abs().numpy(), 90)),
        "n_pl_alvo_valido": int(pl_valid.sum()),
        "definicao_pl_valido": f"rf_targets[:,0] < {PL_TARGET_MAX_VALID} dB",
    }
    if err_pl.numel() > 0:
        m["mae_pl_db"] = float(err_pl.abs().mean())
        m["rmse_pl_db"] = float(torch.sqrt((err_pl ** 2).mean()))
        m["bias_pl_db"] = float(err_pl.mean())
        m["p90_pl_db"] = float(np.percentile(err_pl.abs().numpy(), 90))
    else:
        m.update({"mae_pl_db": None, "rmse_pl_db": None,
                  "bias_pl_db": None, "p90_pl_db": None})

    diag = diag_cache.get(key, sub)
    m["diag"] = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                     else float(v)) for k, v in diag.compute(p, t).items()}
    m["physics_distance_gradient"] = m["diag"].get("physics_distance_gradient")
    return m


# ==========================================================================
# Baselines analiticos — [D2] calibrados SO no treino
# ==========================================================================
def baselines_por_particao(base_dir: Path, dist_m: np.ndarray, rssi_tgt: np.ndarray,
                           p_tx_eff: float | None, freq_mhz: float, log):
    sys.path.insert(0, str(base_dir / "04_baselines"))
    from empirical_models import (cost231_hata, free_space_path_loss,
                                  okumura_hata_rural)
    out, calib = {}, {}
    for name, fn, kw in [("fspl", free_space_path_loss, {}),
                         ("hata_rural", okumura_hata_rural, {}),
                         ("cost231_sub", cost231_hata, {"environment": "suburban"})]:
        pl = fn(dist_m, freq_mhz, **kw)
        if p_tx_eff is None:
            ptx = float(np.median(rssi_tgt + pl))     # calibracao (so no treino)
        else:
            ptx = float(p_tx_eff[name])
        calib[name] = ptx
        rssi_bl = ptx - pl
        e = np.abs(rssi_bl - rssi_tgt)
        out[f"baseline_{name}_mae"] = float(np.mean(e))
        out[f"baseline_{name}_p90"] = float(np.percentile(e, 90))
        out[f"baseline_{name}_bias"] = float(np.mean(rssi_bl - rssi_tgt))
    out["p_tx_eff_dbm"] = calib
    out["freq_mhz"] = freq_mhz
    return out, calib


# ==========================================================================
# [C0-1] TESTES UNITARIOS DO TERMO DE DISTANCIA
# ==========================================================================
def selftest_distancia(base_dir: Path) -> bool:
    sys.path.append(str(base_dir / "02_models"))
    from physics_loss import CurriculumRFLoss

    print("=" * 72)
    print("SELFTEST C0-1 — termo de distancia do FSPL")
    print("=" * 72)
    ok = True

    # T1 — escala: o ETL normaliza por 30000 (prepare_transfer_dataset_v19.py:256)
    d_real = np.array([300.0, 1200.0, 9000.0, 27000.0], dtype=np.float32)
    ea0 = d_real / 30_000.0
    d_antigo = ea0 * 15_000.0
    d_novo = ea0 * 30_000.0
    t1 = np.allclose(d_novo, d_real) and np.allclose(d_antigo, d_real / 2)
    vies_db = 20 * np.log10(0.5)
    print(f"T1 escala   : formula antiga devolve d/2 -> vies FSPL "
          f"{vies_db:.2f} dB | {'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # T2 — indexacao: ea[:n_seed,0] fatia ARESTAS, nao NOS.
    n_seed, n_edges = 4, 9
    torch.manual_seed(0)
    ea = torch.rand(n_edges, 2)
    edge_dst = torch.tensor([3, 1, 0, 2, 1, 3, 0, 2, 1])   # no destino de cada aresta
    dist_por_no = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fatia_arestas = ea[:n_seed, 0]
    t2 = not torch.allclose(fatia_arestas, dist_por_no / dist_por_no.max())
    # prova positiva: agregar por no (min por destino) != fatiar as 4 primeiras arestas
    agreg = torch.full((n_seed,), float("inf"))
    for e in range(n_edges):
        agreg[edge_dst[e]] = torch.minimum(agreg[edge_dst[e]], ea[e, 0])
    t2 = t2 and not torch.allclose(agreg, fatia_arestas)
    print(f"T2 indexacao: ea[:n_seed,0] != distancia por no  | {'PASS' if t2 else 'FAIL'}")
    ok &= t2

    # T3 — monotonicidade do termo corrigido: FSPL(d) cresce com d,
    #      logo a violacao relu(FSPL - pl_pred) cresce com d para pl_pred fixo.
    loss_fn = CurriculumRFLoss(frequency_mhz=900.0, distance_gradient_weight=0.0,
                               variance_weight=0.0, shadowing_ndvi_weight=0.0)
    loss_fn.set_phase(5)
    n = 256
    preds = torch.zeros(n, 5)
    preds[:, 0] = 60.0                     # path loss previsto baixo -> viola em d grande
    tgts = torch.zeros(n, 5)
    vals = []
    for d in [100.0, 1000.0, 10000.0, 30000.0]:
        dd = torch.full((n,), d)
        _, ld = loss_fn(preds, tgts, distances=dd)
        vals.append(float(ld["constraint_fspl"]))
    t3 = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print(f"T3 monotonia: constraint_fspl por d(100/1k/10k/30k m) = "
          f"{[round(v,3) for v in vals]} | {'PASS' if t3 else 'FAIL'}")
    ok &= t3

    # T4 — a loss reage a distancia (nao e constante): sanidade contra "termo morto"
    t4 = (max(vals) - min(vals)) > 1.0
    print(f"T4 termo vivo: amplitude {max(vals)-min(vals):.3f} dB | "
          f"{'PASS' if t4 else 'FAIL'}")
    ok &= t4

    print("=" * 72)
    print("SELFTEST: " + ("PASS" if ok else "FAIL"))
    print("=" * 72)
    return bool(ok)


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] (+{time.perf_counter()-t_start:6.1f}s) {msg}", flush=True)

    base_dir = Path(args.base_dir)
    if args.selftest:
        return 0 if selftest_distancia(base_dir) else 1

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        if args.max_nodes <= 0:
            args.max_nodes = 200_000
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 8192)

    eval_bs = args.eval_batch_size or args.batch_size
    fracs = tuple(float(v) for v in args.split_frac.split(","))
    assert len(fracs) == 3, "--split-frac precisa de 3 valores"

    # --- reprodutibilidade ---
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 72)
    print(f"C0+C1 — GNN-RF com split espacial bloqueado [{args.run_label}]")
    print("=" * 72)
    log(f"base_dir (LEITURA)  : {base_dir}")
    out_dir = Path(args.evid_dir) / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"run_{args.run_label}.json"
    csv_path = out_dir / "training_log.csv"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log(f"saida (ESCRITA)     : {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"Device: cuda | {torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        log("Device: cpu (GPU indisponivel)")

    sys.path.append(str(base_dir / "02_models"))
    sys.path.append(str(base_dir / "03_training"))
    from gnn_rf_model import GNNRFModel
    from physics_loss import CurriculumRFLoss
    from rf_diagnostic_metrics import RFDiagnosticMetrics
    from spatial_cv import SpatialKFold, validate_spatial_cv

    rec: dict = {
        "artefato_tipo": "run_treino_c0c1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "origem_script": ORIGEM_PATH,
        "origem_script_sha256": ORIGEM_SHA256,
        "roadmap_itens": ["C0", "C1"],
        "pontos_revisor": "R1-1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "smoke": bool(args.smoke),
        "status": "provisorio",
        "config": vars(args).copy(),
        "ambiente": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_disponivel": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "executavel": sys.executable,
        },
    }

    def gravar():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=_jsonable)

    gravar()   # grava cedo; atualizado a cada bloco

    # ======================================================================
    # CARREGAR DADOS
    # ======================================================================
    graph_dir = Path(args.graph_dir).resolve() if args.graph_dir else (base_dir / "graph_data")
    rf_path = graph_dir / args.rf_data_file
    struct_path = graph_dir / args.graph_file
    log(f"Carregando RF targets: {rf_path}  ({rf_path.stat().st_size/1e9:.1f} GB)")
    load_kw = dict(map_location="cpu", weights_only=False)
    if args.mmap:
        try:
            rf_data = torch.load(rf_path, mmap=True, **load_kw)
            log("  torch.load(mmap=True) OK")
        except Exception as e:
            log(f"  mmap falhou ({type(e).__name__}: {e}) — carga normal")
            rf_data = torch.load(rf_path, **load_kw)
    else:
        rf_data = torch.load(rf_path, **load_kw)

    rec["dataset"] = {
        "rf_data_file": str(rf_path), "rf_data_bytes": rf_path.stat().st_size,
        "graph_file": str(struct_path),
        "graph_bytes": struct_path.stat().st_size if struct_path.exists() else None,
    }
    if args.hash:
        log("Calculando sha256 do dataset (lento)...")
        rec["dataset"]["rf_data_sha256"] = sha256_file(rf_path)
    gravar()

    ty = rf_data["terrain"].y
    n_ter_total = int(ty.shape[0])
    log(f"Nos terrain no dataset: {n_ter_total:,}")

    dist_pre = None
    if hasattr(rf_data["terrain"], "dist_nearest_m"):
        c = rf_data["terrain"].dist_nearest_m.float()
        if c.shape[0] == n_ter_total and float(c.std()) > 1.0:
            dist_pre = c
            log(f"  dist_nearest_m embutido: mean={float(c.mean()):.0f}m "
                f"std={float(c.std()):.0f}m")

    # features
    if getattr(rf_data["terrain"], "features_raw", None) is not None:
        feats = rf_data["terrain"].features_raw
        feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
        log(f"Features embutidas no rf_data: {tuple(feats.shape)}")
    else:
        log(f"Carregando estrutura DEM para features: {struct_path.name}")
        sd = torch.load(struct_path, **load_kw)
        feats = torch.as_tensor(sd["dem"].x)
        del sd
        gc.collect()
    feats = feats.float()

    pos_deg = rf_data["terrain"].pos.float()
    tgt_all = torch.as_tensor(rf_data["terrain"].y).float()
    ant_x = torch.as_tensor(rf_data["antenna"].x).float()
    n_antenna = int(ant_x.shape[0])
    ant_pos = (rf_data["antenna"].pos.float()
               if hasattr(rf_data["antenna"], "pos") else ant_x[:, :2])

    ei_at_full = rf_data[ET_AT].edge_index.long()
    ea_at_full = (rf_data[ET_AT].edge_attr.float()
                  if hasattr(rf_data[ET_AT], "edge_attr") else None)

    # [C0-4] frequencia empirica das antenas do dataset (edge_attr[:,1]*2600)
    if ea_at_full is not None and ea_at_full.shape[1] >= 2:
        fq = (ea_at_full[:, 1] * 2600.0)
        uq, cnt = torch.unique(torch.round(fq), return_counts=True)
        rec.setdefault("dataset", {})["freq_antenas_mhz"] = {
            "fonte": "edge_attr[:,1] * 2600 (prepare_transfer_dataset_v19.py:256-257)",
            "min": float(fq.min()), "max": float(fq.max()), "media": float(fq.mean()),
            "valores_mais_comuns": [[float(u), int(c)] for u, c in
                                    zip(uq[cnt.argsort(descending=True)][:6],
                                        cnt.sort(descending=True).values[:6])],
        }

    # ter->ter: no V19 as arestas t2t vivem no gpu.pt, nao no rf_data
    tt_in_rf = ET_TT in rf_data.edge_types
    if tt_in_rf:
        ei_tt_full = rf_data[ET_TT].edge_index.long()
        ea_tt_full = (rf_data[ET_TT].edge_attr.float()
                      if hasattr(rf_data[ET_TT], "edge_attr") else None)
        log("  arestas ter->ter vindas do rf_data")
    else:
        log(f"Carregando estrutura DEM para arestas ter->ter: {struct_path.name}")
        sd = torch.load(struct_path, **load_kw)
        t2t = sd["dem", "adjacent_to", "dem"]
        ei_tt_full = t2t.edge_index.long().clone()
        ea_tt_full = (t2t.edge_attr.float().clone()
                      if hasattr(t2t, "edge_attr") and t2t.edge_attr is not None else None)
        del sd, t2t
        gc.collect()
        log("  gpu.pt liberado da memoria apos copiar as arestas")

    del rf_data
    gc.collect()
    log(f"Grafo completo: {n_ter_total:,} terrain | {n_antenna} antenas | "
        f"{ei_at_full.shape[1]:,} ant->ter | {ei_tt_full.shape[1]:,} ter->ter")

    # ======================================================================
    # [C1-5a] PROJECAO + subamostra contigua
    # ======================================================================
    pos_m_full, geo = latlon_graus_para_metros(pos_deg)
    log(f"Extensao: {geo['extensao_x_km']:.2f} x {geo['extensao_y_km']:.2f} km "
        f"(lon {geo['lon_min_deg']:.4f}..{geo['lon_max_deg']:.4f}, "
        f"lat {geo['lat_min_deg']:.4f}..{geo['lat_max_deg']:.4f})")

    # ancora da janela: nos COM alvo de PL valido (= nos com enlace explicito).
    ancora = None
    if args.max_nodes > 0 and args.window_anchor == "cobertura":
        ancora = (tgt_all[:, 0] < PL_TARGET_MAX_VALID).numpy()
        log(f"[JANELA] ancora=cobertura: {int(ancora.sum()):,} de {n_ter_total:,} nos "
            f"com alvo de PL valido ({100*ancora.mean():.1f}%)")
    g_idx = janela_contigua(pos_m_full, args.max_nodes, ancora)   # indices GLOBAIS
    subamostrado = int(g_idx.size) != n_ter_total
    if subamostrado:
        log(f"[JANELA] subamostra contigua: {g_idx.size:,} de {n_ter_total:,} nos")
    g_idx_t = torch.from_numpy(g_idx)

    base = HeteroData()
    base["terrain"].pos = pos_deg[g_idx_t]
    base["terrain"].rf_targets = tgt_all[g_idx_t]
    base["antenna"].x = ant_x
    base["antenna"].num_nodes = n_antenna
    x_base = feats[g_idx_t]                                  # 17 colunas (sem dist)
    if subamostrado:
        ei_at_full, ea_at_full = bipartite_subgraph(
            (torch.arange(n_antenna), g_idx_t), ei_at_full, ea_at_full,
            relabel_nodes=True, size=(n_antenna, n_ter_total))
        ei_tt_full, ea_tt_full = subgraph(g_idx_t, ei_tt_full, ea_tt_full,
                                          relabel_nodes=True, num_nodes=n_ter_total)
        log(f"[JANELA] arestas apos recorte: ant->ter {ei_at_full.shape[1]:,} | "
            f"ter->ter {ei_tt_full.shape[1]:,}")
    base[ET_AT].edge_index = ei_at_full
    if ea_at_full is not None:
        base[ET_AT].edge_attr = ea_at_full
    base[ET_TA].edge_index = ei_at_full[[1, 0]]
    base[ET_TT].edge_index = ei_tt_full
    base[ET_TT].edge_attr = (ea_tt_full if ea_tt_full is not None
                             else torch.zeros((ei_tt_full.shape[1], 2), dtype=torch.float32))
    del feats, tgt_all, pos_deg
    gc.collect()

    n_ter = int(x_base.shape[0])
    pos_m = pos_m_full[g_idx]
    del pos_m_full
    dist_all = (dist_pre[g_idx_t] if dist_pre is not None else None)
    if dist_all is None:
        log("dist_nearest_m ausente — calculando Haversine (RFDiagnosticMetrics)")
        tmp = RFDiagnosticMetrics(terrain_pos=base["terrain"].pos, antenna_pos=ant_pos,
                                  terrain_features=x_base,
                                  frequency_mhz=args.diag_freq_mhz)
        dist_all = tmp.dist_nearest.clone()
        del tmp
    log(f"dist_to_antenna: min={float(dist_all.min()):.0f}m "
        f"max={float(dist_all.max()):.0f}m mean={float(dist_all.mean()):.0f}m")

    # ======================================================================
    # [C1-5b] SPLIT ESPACIAL DE TRES VIAS
    # ======================================================================
    grid_km, buffer_km, ajustado = args.grid_km, args.buffer_km, None
    ext = min((pos_m[:, 0].max() - pos_m[:, 0].min()) / 1000.0,
              (pos_m[:, 1].max() - pos_m[:, 1].min()) / 1000.0)
    if args.smoke and ext / max(grid_km, 1e-9) < 6.0:
        novo = float(ext / 6.0)
        ajustado = {"grid_km_original": grid_km, "buffer_km_original": buffer_km,
                    "grid_km_usado": novo, "buffer_km_usado": novo * (buffer_km / grid_km),
                    "motivo": ("janela do smoke tem menos de 6 blocos por lado; "
                               "grid e buffer reduzidos na mesma proporcao 5:2. "
                               "GEOMETRIA DE SMOKE — NAO E A DE PRODUCAO")}
        buffer_km = novo * (buffer_km / grid_km)
        grid_km = novo
        log(f"[SMOKE] grid ajustado {args.grid_km} -> {grid_km:.3f} km | "
            f"buffer {args.buffer_km} -> {buffer_km:.3f} km (nao e geometria de producao)")

    log("Split espacial por blocos (logica de spatial_cv.SpatialKFold)...")
    parts_local, split_info = split_espacial_3vias(
        pos_m, grid_km, buffer_km, fracs, args.split_seed, SpatialKFold, log)
    for k, v in parts_local.items():
        if len(v) == 0:
            raise RuntimeError(f"Particao '{k}' ficou VAZIA apos o buffer — "
                               f"reduza --buffer-km ou aumente --grid-km.")

    verif = verificar_split(parts_local, pos_m, buffer_km, validate_spatial_cv, log)

    rec["geometria"] = dict(geo)
    rec["geometria"].update({"grid_km_usado": grid_km, "buffer_km_usado": buffer_km,
                             "grid_km_ajustado_por_smoke": ajustado,
                             "n_nos_no_grafo_de_trabalho": n_ter,
                             "subamostrado": subamostrado,
                             "n_nos_dataset_completo": n_ter_total})
    rec["split"] = split_info
    rec["split"]["verificacao"] = verif
    rec["particoes"] = {}
    pl_valid = {}
    for k, loc in parts_local.items():
        glob = g_idx[loc]
        pv = base["terrain"].rf_targets[torch.from_numpy(loc), 0] < PL_TARGET_MAX_VALID
        pl_valid[k] = pv
        bb = pos_m[loc]
        rec["particoes"][k] = {
            "n": int(loc.size),
            "frac_dos_nos": float(loc.size / n_ter),
            "idx_sha256_global": sha256_idx(glob),
            "idx_sha256_local_janela": sha256_idx(loc),
            "n_pl_alvo_valido": int(pv.sum()),
            "frac_pl_alvo_valido": float(pv.float().mean()),
            "bbox_km": [float(bb[:, 0].min() / 1000), float(bb[:, 1].min() / 1000),
                        float(bb[:, 0].max() / 1000), float(bb[:, 1].max() / 1000)],
            "nota_indices": ("idx_sha256_global = sha256 dos indices int64 ORDENADOS "
                             "no dataset original; idx_sha256_local_janela idem, "
                             "relativos a subamostra contigua"),
        }
        log(f"  {k}: {loc.size:,} nos ({100*loc.size/n_ter:.1f}%) | "
            f"PL alvo valido: {int(pv.sum()):,} ({100*float(pv.float().mean()):.1f}%)")
    gravar()

    if not verif["ok"]:
        raise RuntimeError("Verificacao do split espacial FALHOU — ver split.verificacao")

    # ======================================================================
    # [D1] feature col 17 normalizada com estatistica de TREINO
    # ======================================================================
    tr_loc = torch.from_numpy(parts_local["train"])
    # dist (metros) reindexada para a numeracao LOCAL do subgrafo de treino:
    # batch["terrain"].n_id ja e local a particao, entao este vetor e o correto
    # para o termo FSPL corrigido [C0-1].
    dist_train_local = dist_all[tr_loc].contiguous()
    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    d_mean_all = float(dist_all.mean())
    d_std_all = max(float(dist_all.std()), 1.0)
    x_full = torch.cat([x_base, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    base["terrain"].x = x_full
    rec["normalizacao"] = {
        "feature_col17": "dist_to_nearest_antenna",
        "dist_mean_train": d_mean_tr, "dist_std_train": d_std_tr,
        "dist_mean_todos": d_mean_all, "dist_std_todos": d_std_all,
        "decisao": ("[D1] media/desvio do TREINO (original train_lins_physics.py:569-571 "
                    "usava todos os nos = estatistica de teste na padronizacao)"),
    }
    log(f"[D1] col17 normalizada com estatistica de treino "
        f"(mean={d_mean_tr:.0f}m std={d_std_tr:.0f}m; todos: {d_mean_all:.0f}/{d_std_all:.0f})")

    # ======================================================================
    # [C1-6] SUBGRAFOS INDUZIDOS
    # ======================================================================
    log("Induzindo subgrafos por particao (sem arestas cruzando fronteira)...")
    graphs, sub_info = {}, {}
    for k in ("train", "val", "test"):
        graphs[k], sub_info[k] = induzir_particao(
            base, torch.from_numpy(parts_local[k]), n_antenna, log, k)
    # GUARDA: particao sem aresta antena->terreno nao exercita o caminho de
    # mensagem da antena; qualquer metrica dali e sobre alvo sentinela.
    degen = [k for k in ("train", "val", "test")
             if sub_info[k]["n_arestas_ant_ter"] == 0]
    rec["guarda_particao_degenerada"] = {
        "particoes_sem_aresta_antena": degen,
        "n_pl_alvo_valido": {k: rec["particoes"][k]["n_pl_alvo_valido"]
                             for k in ("train", "val", "test")},
        "significado": ("particao com 0 arestas ant->ter e 0 alvo de PL valido nao "
                        "testa o caminho da antena; MAE de RSSI ali e regressao "
                        "sobre o sentinela constante de -110 dBm"),
    }
    if degen:
        msg = (f"PARTICAO DEGENERADA: {degen} sem nenhuma aresta antena->terreno. "
               f"Use --window-anchor cobertura ou aumente --max-nodes.")
        log(f"  ERRO: {msg}")
        gravar()
        raise RuntimeError(msg)

    rec["subgrafos"] = sub_info
    rec["subgrafos"]["declaracao"] = (
        "Cada particao e o subgrafo INDUZIDO pelos seus proprios nos terrain mais "
        "TODAS as antenas. A inferencia em val e em test usa exclusivamente contexto "
        "da propria regiao: nenhuma aresta liga nos de particoes diferentes, em "
        "nenhuma das 4 camadas do encoder.")
    del base
    gc.collect()
    gravar()

    # ======================================================================
    # LOADERS
    # ======================================================================
    nn_kw = {ET_AT: [args.k_antenna], ET_TT: [args.k_terrain], ET_TA: [args.k_antenna]}

    def make_loader(g, bs, shuffle):
        kw = dict(data=g, num_neighbors=nn_kw, input_nodes=("terrain", None),
                  batch_size=bs, shuffle=shuffle, num_workers=args.num_workers)
        if args.num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = args.prefetch
        return NeighborLoader(**kw)

    loaders = {"train": make_loader(graphs["train"], args.batch_size, True),
               "val": make_loader(graphs["val"], eval_bs, False),
               "test": make_loader(graphs["test"], eval_bs, False)}
    n_batches = len(loaders["train"])
    log(f"Loaders: train {n_batches} batches (bs={args.batch_size}) | "
        f"val {len(loaders['val'])} | test {len(loaders['test'])} (bs={eval_bs})")

    # diag por particao (alinhado no-a-no)
    diag_caches = {}
    for k in ("train", "val", "test"):
        loc = torch.from_numpy(parts_local[k])
        diag_caches[k] = _DiagCache(RFDiagnosticMetrics,
                                    pos=graphs[k]["terrain"].pos,
                                    feats=x_base[loc],
                                    dist=dist_all[loc],
                                    antenna_pos=ant_pos,
                                    freq_mhz=args.diag_freq_mhz)

    # ======================================================================
    # [D2] BASELINES ANALITICOS — calibrados no treino, aplicados as 3 particoes
    # ======================================================================
    if not args.no_baselines:
        try:
            log(f"Baselines analiticos (freq={args.freq_mhz} MHz, calibracao SO no treino)...")
            bl = {}
            d_tr = dist_all[tr_loc].numpy()
            r_tr = graphs["train"]["terrain"].rf_targets[:, 3].numpy()
            bl["train"], calib = baselines_por_particao(
                base_dir, d_tr, r_tr, None, args.freq_mhz, log)
            for k in ("val", "test"):
                loc = torch.from_numpy(parts_local[k])
                bl[k], _ = baselines_por_particao(
                    base_dir, dist_all[loc].numpy(),
                    graphs[k]["terrain"].rf_targets[:, 3].numpy(),
                    calib, args.freq_mhz, log)
            bl["decisao"] = ("[D2] P_tx_eff calibrado SO com alvos de treino e aplicado "
                             "inalterado a val/test (original calibrava no proprio "
                             "conjunto avaliado: train_lins_physics.py:186-193)")
            rec["baselines_analiticos"] = bl
            for k in ("train", "val", "test"):
                log(f"  {k}: FSPL {bl[k]['baseline_fspl_mae']:.2f} | "
                    f"Hata {bl[k]['baseline_hata_rural_mae']:.2f} | "
                    f"COST231 {bl[k]['baseline_cost231_sub_mae']:.2f} dB")
        except Exception as e:
            log(f"  AVISO: baselines falharam ({type(e).__name__}: {e})")
            rec["baselines_analiticos"] = {"erro": f"{type(e).__name__}: {e}"}
    gravar()

    # ======================================================================
    # MODELO / OTIMIZADOR / LOSS
    # ======================================================================
    model = GNNRFModel(terrain_dim=x_full.shape[1], antenna_dim=ant_x.shape[1],
                       hidden_dim=args.hidden_dim, num_layers=4, heads=4,
                       edge_dim=2, output_dim=5, dropout=0.1,
                       use_physics_constraints=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Modelo: {n_params:,} parametros ({n_params_tr:,} treinaveis) | {log_vram()}")
    rec["modelo"] = {"classe": "GNNRFModel", "hidden_dim": args.hidden_dim,
                     "num_layers": 4, "heads": 4, "output_dim": 5, "dropout": 0.1,
                     "terrain_dim": int(x_full.shape[1]),
                     "antenna_dim": int(ant_x.shape[1]),
                     "n_params": int(n_params), "n_params_treinaveis": int(n_params_tr)}

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0,
                                                T_mult=2, eta_min=1e-6)
    elif args.scheduler == "cosine_simple":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_t0, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=args.scheduler_patience, min_lr=1e-6)

    # [C0-4] frequencia do FSPL da LOSS agora e EXPLICITA
    loss_fn = CurriculumRFLoss(
        frequency_mhz=args.freq_mhz,
        distance_gradient_weight=args.dist_grad_weight,
        distance_gradient_n_pairs=512,
        variance_weight=args.variance_weight,
        shadowing_ndvi_weight=args.ndvi_weight,
        shadowing_ndvi_min_corr=0.15,
    ).to(device)
    rec["loss"] = {
        "classe": "CurriculumRFLoss",
        "frequency_mhz_loss_fspl": args.freq_mhz,
        "frequency_mhz_diagnostico": args.diag_freq_mhz,
        "nota_frequencia": ("[C0-4] original nao passava frequency_mhz -> default 900 MHz "
                            "(physics_loss.py:36) na loss, enquanto RFDiagnosticMetrics "
                            "auditava com 1800 MHz (rf_diagnostic_metrics.py:38)"),
        "distance_gradient_weight": args.dist_grad_weight,
        "variance_weight": args.variance_weight,
        "shadowing_ndvi_weight": args.ndvi_weight,
        "fonte_de_distances": ("[C0-1] dist_to_nearest_antenna em metros, alinhada no-a-no "
                               "(antes: edge_attr[:n_seed,0]*15000)"),
    }

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.transfer_from and Path(args.transfer_from).exists():
        ck = torch.load(args.transfer_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        log(f"[TRANSFER] pesos carregados de {args.transfer_from}")
        rec["transfer_from"] = args.transfer_from

    # ======================================================================
    # LOOP
    # ======================================================================
    def rodar_avaliacao(k: str) -> dict:
        model.eval()
        n_p = graphs[k]["terrain"].x.shape[0]
        P, T, I = [], [], []
        with torch.no_grad():
            for b in loaders[k]:
                b = b.to(device)
                bs = b["terrain"].batch_size
                ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    out = model(b)
                P.append(out["predictions"][:bs].float().detach().cpu())
                T.append(b["terrain"].rf_targets[:bs].float().detach().cpu())
                I.append(b["terrain"].n_id[:bs].cpu())
        po, to, vi = _scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
        return metricas_particao(po, to, vi, diag_caches[k], k, pl_valid[k])

    rec["epocas"] = []
    best_val, best_epoch, sem_melhora = float("inf"), -1, 0
    csv_fields = ["epoch", "train_loss", "lr", "elapsed_s",
                  "train_mae_rssi", "train_rmse_rssi", "train_mae_pl",
                  "val_mae_rssi", "val_rmse_rssi", "val_mae_pl",
                  "test_mae_rssi", "test_rmse_rssi", "test_mae_pl",
                  "train_dist_grad", "val_dist_grad", "test_dist_grad"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore").writeheader()

    log(f"Iniciando treino: {args.epochs} epocas x {n_batches} batches")
    print("-" * 72, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total_loss, nb = 0.0, 0
        P, T, I = [], [], []
        comp_dist_gradiente_ok = True

        for bi, batch in enumerate(loaders["train"], 1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                preds = out["predictions"]
                # [C0-2] bs = numero de SEMENTES do batch
                bs = batch["terrain"].batch_size
                preds_s = preds[:bs]
                targets_s = batch["terrain"].rf_targets[:bs]
                seed_ids = batch["terrain"].n_id[:bs].cpu()

                # [C0-1] distancia por NO, em metros, alinhada com preds_s
                dist_b = dist_train_local[seed_ids].to(device)

                ndvi_b = (batch["terrain"].x[:bs, COL_NDVI]
                          if batch["terrain"].x.shape[1] > COL_NDVI else None)

                loss, ldict = loss_fn(preds_s, targets_s,
                                      distances=dist_b,       # <- C0-1
                                      dist_to_ant=dist_b,
                                      ndvi=ndvi_b)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss nao-finita na epoca {epoch}, batch {bi}")
            scaler.scale(loss).backward()
            if args.max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            nb += 1
            P.append(preds_s.float().detach().cpu())
            T.append(targets_s.float().detach().cpu())
            I.append(seed_ids)
            if bi % max(1, n_batches // 5) == 0 or bi == n_batches:
                log(f"  Epoch {epoch}/{args.epochs} | batch {bi}/{n_batches} | "
                    f"loss={total_loss/nb:.3f} | {log_vram()}")

        avg_loss = total_loss / max(nb, 1)
        if args.scheduler in ("cosine", "cosine_simple"):
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        n_tr_nodes = graphs["train"]["terrain"].x.shape[0]
        po, to, vi = _scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_tr_nodes)
        m_train = metricas_particao(po, to, vi, diag_caches["train"], "train", pl_valid["train"])
        del P, T, I, po, to, vi
        gc.collect()

        m_val = rodar_avaliacao("val")
        m_test = (rodar_avaliacao("test")
                  if (args.test_every > 0 and epoch % args.test_every == 0) else None)

        elapsed = time.perf_counter() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]
        linha = {"epoch": epoch, "train_loss": round(avg_loss, 4),
                 "lr": lr_now, "elapsed_s": round(elapsed, 1),
                 "loss_componentes": {k: float(v) for k, v in ldict.items()
                                      if torch.is_tensor(v) and v.numel() == 1},
                 "train": m_train, "val": m_val, "test": m_test,
                 "test_usado_na_selecao": False}
        if device == "cuda":
            linha["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        rec["epocas"].append(linha)

        log(f"Epoch {epoch:02d} | loss={avg_loss:.3f} | "
            f"MAE_RSSI train={m_train['mae_rssi_db']:.3f} val={m_val['mae_rssi_db']:.3f}"
            + (f" test={m_test['mae_rssi_db']:.3f}" if m_test else "") + " dB")
        log(f"          MAE_PL(alvo<299) train={m_train['mae_pl_db']} "
            f"val={m_val['mae_pl_db']}" + (f" test={m_test['mae_pl_db']}" if m_test else ""))
        log(f"          dist_gradient train={m_train['physics_distance_gradient']} "
            f"val={m_val['physics_distance_gradient']} | cobertura da epoca "
            f"train={m_train['cobertura_epoca']:.3f} | {elapsed:.0f}s")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore").writerow({
                "epoch": epoch, "train_loss": round(avg_loss, 4), "lr": lr_now,
                "elapsed_s": round(elapsed, 1),
                "train_mae_rssi": m_train["mae_rssi_db"], "train_rmse_rssi": m_train["rmse_rssi_db"],
                "train_mae_pl": m_train["mae_pl_db"],
                "val_mae_rssi": m_val["mae_rssi_db"], "val_rmse_rssi": m_val["rmse_rssi_db"],
                "val_mae_pl": m_val["mae_pl_db"],
                "test_mae_rssi": (m_test or {}).get("mae_rssi_db"),
                "test_rmse_rssi": (m_test or {}).get("rmse_rssi_db"),
                "test_mae_pl": (m_test or {}).get("mae_pl_db"),
                "train_dist_grad": m_train["physics_distance_gradient"],
                "val_dist_grad": m_val["physics_distance_gradient"],
                "test_dist_grad": (m_test or {}).get("physics_distance_gradient")})

        # ---- SELECAO: SO VAL (guarda explicita) ----
        criterio = _selecionar_melhor_epoca(m_val)
        if criterio < best_val:
            best_val, best_epoch, sem_melhora = criterio, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mae_rssi": criterio, "args": vars(args),
                        "selecao": "val mae_rssi_db (test nunca usado)"},
                       ckpt_dir / "checkpoint_best.pt")
            log(f"  CKPT best (val MAE_RSSI={criterio:.4f} dB)")
        else:
            sem_melhora += 1

        gravar()
        if args.early_stopping_patience and sem_melhora >= args.early_stopping_patience:
            log(f"  EARLY STOP na epoca {epoch}")
            break
        print("-" * 72, flush=True)

    # ======================================================================
    # [D4] TESTE FINAL sobre o checkpoint best-por-val
    # ======================================================================
    ck = ckpt_dir / "checkpoint_best.pt"
    m_test_final = None
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        log(f"[FINAL] recarregado checkpoint best (epoca {st['epoch']}) — avaliando test")
        m_test_final = rodar_avaliacao("test")
        m_val_final = rodar_avaliacao("val")
        log(f"[FINAL] test  MAE_RSSI={m_test_final['mae_rssi_db']:.4f} dB | "
            f"RMSE={m_test_final['rmse_rssi_db']:.4f} | MAE_PL={m_test_final['mae_pl_db']}")
    else:
        m_val_final = None

    rec["selecao"] = {
        "criterio": "menor mae_rssi_db em VAL",
        "melhor_epoca": best_epoch,
        "melhor_val_mae_rssi_db": (None if best_val == float("inf") else best_val),
        "test_usado_na_selecao": False,
        "checkpoint": str(ck),
        "test_no_melhor_ckpt": m_test_final,
        "val_no_melhor_ckpt": m_val_final,
        "nota": ("[D4] o teste por epoca em `epocas[].test` e diagnostico; o numero "
                 "reportavel e `selecao.test_no_melhor_ckpt`, medido uma unica vez "
                 "sobre o checkpoint escolhido por val."),
    }
    rec["custo"] = {
        "tempo_total_s": round(time.perf_counter() - t_start, 1),
        "n_epocas_rodadas": len(rec["epocas"]),
        "vram_peak_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                         if device == "cuda" else None),
    }
    rec["_fontes"] = {
        "n_nos_dataset": "rf_data['terrain'].y.shape[0] (torch.load do rf-data-file)",
        "particoes.*.n": "len(np.where(mask)[0]) apos buffer — split_espacial_3vias()",
        "particoes.*.idx_sha256_global": "sha256(int64 ordenados) — sha256_idx()",
        "split.verificacao.pares.*.dist_min_km": "scipy.spatial.cKDTree.query(k=1) exato",
        "epocas.*.<part>.mae_rssi_db": "metricas_particao() sobre array ordenado por no",
        "epocas.*.<part>.mae_pl_db": f"idem, restrito a rf_targets[:,0] < {PL_TARGET_MAX_VALID}",
        "epocas.*.<part>.physics_distance_gradient": ("RFDiagnosticMetrics.compute() com "
                                                      "N == n_mask (sem reamostragem)"),
        "selecao.test_no_melhor_ckpt": "rodar_avaliacao('test') apos recarregar checkpoint_best.pt",
        "modelo.n_params_treinaveis": "sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "dataset.freq_antenas_mhz": "edge_attr[:,1] * 2600 do proprio .pt",
    }
    gravar()
    log(f"JSON gravado: {json_path}")
    log(f"CSV  gravado: {csv_path}")
    print("=" * 72)
    return 0


def _selecionar_melhor_epoca(m_val: dict) -> float:
    """
    Guarda explicita do C1-5: a selecao de checkpoint le SOMENTE metricas de val.
    Passar um dicionario que contenha chave de teste levanta AssertionError.
    """
    assert "test" not in m_val, "selecao nunca pode ver o conjunto de teste"
    return float(m_val["mae_rssi_db"])


if __name__ == "__main__":
    sys.exit(main())
