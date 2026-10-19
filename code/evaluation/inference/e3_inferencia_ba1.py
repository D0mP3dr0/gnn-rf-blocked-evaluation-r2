#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
e3_inferencia_ba1.py — COPIA ADAPTADA (braco BA1 do R2-3, R4 IEEE Access 47637)
=============================================================================
Origem: _revisao_2026-10-14/fase_C/avaliacao/e3_inferencia_faseC.py
        (sha256 bfa568f1bbe166b7293edc2ad5ba6361521b3788b488a7142f8cc50714d3c904).
Protocolo: PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md §7 + "Decisao do
dono (15/10/2026)". Patch MINIMO (lista I; diff em fase_BA1/diffs/):
  (I1) --braco ganha "ba1" (run_label ba1rus_<cidade>_s<seed>_Q<q>_g10b2,
       evid default fase_BA1/treinos, saida default fase_BA1/e3_predicoes/
       ba1rus_<cidade>_<Q>_s<seed>.npz/.json — nome da fase B,
       common.npz_path(braco="ba1rus", ...)).
  (I2) o modelo e as entradas do BA1 NAO sao copiados: RadioUNetStyleRF,
       bloco_enlaces_bruto e normalizar_bloco sao importados do proprio
       treinador fase_BA1/train_ba1_radiounet_style.py, cujo sha256 tem de
       ser igual ao run_json["script_sha256"] da corrida (aborta se nao).
  (I3) bloco de enlaces recomputado com as escolhas gravadas no run JSON
       (config.desempate_enlaces, config.preench_altura) e normalizado com a
       estatistica do treino recomputada; confere media/desvio contra
       run_json["entradas_ba1"]["normalizacao"] (aborta se |dif| > 1e-5).
  (I4) autocast do braco ba1 = o do treino: (device=='cuda') e
       config.amp == "on" (e nao --sem-autocast).
  (I5) testemunha ganha o criterio da §7 (|dif MAE RSSI| <= 2,5e-4 dB).
Split, [D1], forward por lotes (rodar_avaliacao_mlp), metricas por
populacao e formato do .npz: INTACTOS.
=============================================================================

=============================================================================
e3_inferencia_faseC.py — COPIA ADAPTADA (fase C, R4 IEEE Access 47637)
=============================================================================
Origem: e3_inferencia_por_populacao.py v2
        (sha256 c8e4356f1842fb2f9c10e6d8f0020a2990ab72b2f649feb79d9af98d960ad96c).
Protocolo: _revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md, secao
"Avaliacao" (aprovado pelo dono 14/10/2026; "va" da avaliacao 18/10/2026).
Patch MINIMO (diff -u contra a origem em avaliacao/e3_inferencia_faseC.diff);
nenhuma logica de split, normalizacao, forward, metricas ou formato de saida
foi alterada:
  (a) --prefixo-run {c0c1cf,c1noreg,c2nomono} (default c0c1cf) substitui o
      prefixo fixo "c0c1cf_" do braco gnn (run_label = <prefixo>_<cidade>_
      s<seed>_Q<q>_g10b2). Braco mlp inalterado (mlpcf_; so aceita c0c1cf).
  (b) --evid-dir default por prefixo: c0c1cf -> EVIDENCIA_RESUBMISSAO/treinos
      (o default de antes); c1noreg/c2nomono -> fase_C/treinos.
  (c) saida default em fase_C/e3_predicoes/: braco gnn ->
      <prefixo>_<cidade>_<Q>_s<seed>.npz/.json (formato de nome da fase B,
      common.npz_path(braco=prefixo,...)); braco mlp mantem o nome
      e3_mlp_<celula>_s<seed> mas tambem nessa pasta (o default antigo
      gravava fora desta pasta).
  (d) --rng-seed default = --seed da corrida (convencao conferida nos .json
      do C0: config_efetiva.rng_seed == seed em gnn_bauru_Q2_s43,
      gnn_campinas_Q3_s45, gnn_lins_Q1_s44, gnn_sorocaba_Q4_s46,
      gnn_bauru_Q1_s42). Passar --rng-seed explicito continua valendo.
  (e) metadado: rec["patch_faseC"] (prefixo, evid_dir, protocolo, origem);
      nao altera nenhum campo existente.
=============================================================================

e3_inferencia_por_populacao.py — E3 (fio c0-inference-2026-09-24):
inferencia SEM TREINO dos checkpoints ja existentes de mlpcf_* (MLP sem grafo,
controle E1/B4) e c0c1cf_* (GNN, 4 camadas GATv2/SAGE), com MAE de RSSI (dB)
decomposto por populacao (todos / cobertos / sentinela) na particao de TESTE
bloqueada espacialmente.

ARQUIVO NOVO. So LE checkpoints/run JSON/rf_data ja existentes em
EVIDENCIA_RESUBMISSAO (READ-ONLY); nunca escreve la. NAO TREINA: carrega
`checkpoint_best.pt` e roda so `model.eval()` + forward (ou, com
--somente-construcao, nem isso — so `load_state_dict(strict=True)`).

Pre-registro: E3_protocolo_preregistro.md (aprovado pelo dono 24/09/2026,
fio c0-inference-2026-09-24).

=============================================================================
VERSAO
=============================================================================
v2 — patch minimo, decisao do chefe apos o portao ia-bug-silencioso
(E3_portao_bug_silencioso.json, achados A1-A3; forum-eng-ia_parecer_R2.md,
adendo "patch no MESMO arquivo"). v1 preservada em
artefatos/scripts/e3_inferencia_por_populacao_v1_ae2e9731.py
(sha256 ae2e97318de...664566ac6).
  1. [C1] autocast identico ao congelado (train_mlp_c0_spatial.py:1436-1437 /
     train_gnn_c0_spatial.py:1298: mesmo device_type "cuda", dtype padrao,
     use_amp=(device=='cuda')) nos dois bracos; flag --sem-autocast forca
     FP32 puro. `autocast`/dtype efetivo gravados em config_efetiva.
  2. [C2] flag --rng-seed (default 42): manual_seed/np.random.seed/
     random.seed/cuda.manual_seed_all fixados imediatamente antes do
     NeighborLoader (gnn) e do laco de lotes (mlp); gravado em config_efetiva.
  3. [C4] guarda de particao de teste degenerada (0 arestas antena->terreno)
     + log de arestas_cruzando_fronteira, portados de
     train_gnn_c0_spatial.py:583-600 (calculo) e :1143-1156 (guarda).
  4. [C5, cosmetico] assert que amarra idx_test_global a parts['test'] (valor
     E sha256), fechando o buraco que o mutante M5 do portao explorou sem
     guarda estrutural (antes so a testemunha numerica detectava).
  5. Nenhuma outra logica alterada (mesmo split, mesma normalizacao [D1],
     mesmas metricas por populacao, mesmo formato de saida .json/.npz).

=============================================================================
PROVENIENCIA — de onde vem cada pedaco (arquivo:linha do CONGELADO)
=============================================================================
Os dois scripts congelados abaixo tem as mesmas funcoes auxiliares
BYTE-IDENTICAS (a doutrina do proprio train_mlp_c0_spatial.py, secao "O QUE
FICA IGUAL", diz isso explicitamente); por isso cada funcao abaixo cita as
DUAS localizacoes quando coincidem.

  GNN (origem)  : EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/train_gnn_c0_spatial.py
  MLP (controle): EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/train_mlp_c0_spatial.py

  sha256_file()              <- train_gnn_c0_spatial.py:312-317   (== train_mlp:437-442)
  sha256_idx()                <- train_gnn_c0_spatial.py:320-323   (== train_mlp:445-448)
  latlon_graus_para_metros()  <- train_gnn_c0_spatial.py:351-373   (== train_mlp:476-498)
                                  [C1-5a] projecao equiretangular, MESMA
                                  convencao do ETL (prepare_transfer_dataset_v19.py:181-184).
  split_espacial_3vias()      <- train_gnn_c0_spatial.py:412-487   (== train_mlp:537-612)
                                  [C1-5b] SpatialKFold._assign_groups (casa) +
                                  3 vias com buffer sequencial train->val->test.
  metricas_populacao()        <- deriva de metricas_particao(), o NUCLEO
                                  (err_rssi/err_pl) e' COPIA LITERAL de
                                  train_gnn_c0_spatial.py:666-668
                                  (== train_mlp:790-792); o resto de
                                  metricas_particao() (RFDiagnosticMetrics,
                                  linhas 690-693/814-817) NAO e' usado aqui —
                                  E3 pede so MAE/mediana/IQR por populacao,
                                  nao os ~66 diagnosticos de fisica.
  MLPRFModel                  <- COPIA de train_mlp_c0_spatial.py:691-724
                                  (MLP_LARGURAS = mesma tupla; paridade de
                                  1.812.515 parametros treinaveis com o GNN,
                                  conferida em runtime como o congelado faz
                                  em :1361-1370).
  construir_modelo_gnn()      <- reproduz a chamada de
                                  train_gnn_c0_spatial.py:1232-1235
                                  (GNNRFModel importado de
                                  02_models/gnn_rf_model.py, NAO copiado).
  induzir_subgrafo_teste()    <- deriva de induzir_particao(),
                                  train_gnn_c0_spatial.py:547-600 (mesma
                                  bipartite_subgraph/subgraph com
                                  relabel_nodes=True), restrita a UMA
                                  particao (aqui so "test"; o congelado
                                  induzia as 3 porque TREINAVA).
  _scatter_por_semente()      <- COPIA de train_gnn_c0_spatial.py:635-648
                                  (== train_mlp:759-771).
  rodar_avaliacao_mlp()       <- deriva de train_mlp_c0_spatial.py:1426-1443
                                  (mesmo laco de lotes(), sem grafo).
  rodar_avaliacao_gnn()       <- deriva de train_gnn_c0_spatial.py:1290-1305
                                  (mesmo laco de NeighborLoader, model(batch),
                                  preds[:bs]/targets[:bs]/n_id[:bs]).
  [D1] normalizacao da col 17 <- train_gnn_c0_spatial.py:1108-1129
                                  (== train_mlp:1211-1232): media/desvio SO
                                  do TREINO, aplicada 1x (nao se refaz por
                                  epoca porque aqui nao ha epoca).
  PL_TARGET_MAX_VALID = 299.0  <- train_gnn_c0_spatial.py:219 (== train_mlp:326).
  ET_AT/ET_TA/ET_TT            <- train_gnn_c0_spatial.py:542-544.

=============================================================================
O QUE ESTE SCRIPT NAO REPRODUZ DO CONGELADO (declarado, nao omitido)
=============================================================================
  - janela_contigua()/--max-nodes>0: os runs mlpcf_*/c0c1cf_* usados aqui
    (config.max_nodes == 0 em todos os run JSON conferidos) NUNCA
    subamostram; se algum run tiver max_nodes>0 este script levanta
    NotImplementedError em vez de silenciosamente medir a celula errada.
  - verificar_split() (cKDTree exaustivo par-a-par + validate_spatial_cv da
    casa, train_gnn_c0_spatial.py:490-533): o pre-registro so pede
    reconferir idx_sha256_global das 3 particoes: se o hash bate, a
    particao e' bit-a-bit a mesma que passou pela verificacao no run
    original (o log daquele run ja atesta buffer >= exigido). Reexecutar a
    verificacao geometrica aqui seria auditar de novo algo ja carimbado; o
    hash e' a prova de que e' o MESMO conjunto.
  - RFDiagnosticMetrics / CurriculumRFLoss: nao entram (E3 nao mede fisica
    nem perda; so MAE por populacao). Nao importados.
  - guarda_particao_degenerada [M3]/[C1-6]: guarda de TREINO (0 arestas
    antena->terreno = partição inutil para caber loss); aqui so se le teste
    ja treinado com sucesso, guarda nao se aplica.
  - Para o braco GNN, ter->ter vem do gpu.pt (~15 GB) quando ausente do
    rf_data (confirmado: rf_data desta campanha so tem a relacao
    antenna->terreno; train_gnn_c0_spatial.py:963-979). Carregar esse
    arquivo so acontece se --somente-construcao NAO for passado.

=============================================================================
CLI
=============================================================================
  --celula <cidade>_Q<n>     ex.: campinas_Q1
  --braco {mlp,gnn}
  --seed INT                 default 42
  --cpu                      forca CPU mesmo se CUDA estiver disponivel
  --somente-construcao       so carrega o checkpoint e instancia o modelo
                              (load_state_dict(strict=True)); NAO faz forward,
                              NAO le rf_data/grafo. Usado no braco GNN desta
                              rodada porque montar o grafo completo (gpu.pt,
                              ~15 GB) para inferencia sem GPU passaria dos
                              ~10 min combinados no protocolo.
  --orcamento-min FLOAT       default 60; aborta e grava estado parcial se
                              o tempo de parede ultrapassar este valor
                              (checado antes das fases pesadas: hash do
                              rf_data, carga do grafo, forward).
  --saida-json PATH           default: producao (ver DEFAULT_SAIDA_DIR)
  --saida-npz PATH            idem
  --hash                      imprime sha256 do proprio script e sai (0).
  --base-dir PATH              default /trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2
                              (raiz de LEITURA para 02_models/03_training).
  --evid-dir PATH              default EVIDENCIA_RESUBMISSAO/treinos (onde
                              estao os checkpoints e run JSON dos 80 runs).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------
# Constantes herdadas do congelado (train_gnn_c0_spatial.py:219,542-544)
# --------------------------------------------------------------------------
PL_TARGET_MAX_VALID = 299.0
ET_AT = ("antenna", "propagates_to", "terrain")
ET_TA = ("terrain", "in_range_of", "antenna")
ET_TT = ("terrain", "connects_to", "terrain")

# [M1] train_mlp_c0_spatial.py:337-338
N_PARAMS_ALVO_GNN = 1_812_515
MLP_LARGURAS = (712, 712, 720, 688, 256)

DEFAULT_BASE_DIR = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2"
DEFAULT_EVID_DIR = ("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                     "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/treinos")
DEFAULT_SAIDA_DIR_ORIGEM_E3 = Path("predictions/"
                                    "c0")  # [faseC] so referencia; nao se grava la
# [faseC (b),(c)] pastas da fase C
_FASEC_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                  "_revisao_2026-10-14/fase_C")
DEFAULT_EVID_DIR_FASEC = str(_FASEC_DIR / "treinos")
DEFAULT_SAIDA_DIR = _FASEC_DIR / "e3_predicoes"
PREFIXOS_RUN = ("c0c1cf", "c1noreg", "c2nomono")
# [I1] pastas do BA1
_BA1_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                "_revisao_2026-10-14/fase_BA1")
DEFAULT_EVID_DIR_BA1 = str(_BA1_DIR / "treinos")
DEFAULT_SAIDA_DIR_BA1 = _BA1_DIR / "e3_predicoes"
TREINADOR_BA1 = _BA1_DIR / "train_ba1_radiounet_style.py"
PREFIXO_BA1 = "ba1rus"


def carregar_treinador_ba1(sha_esperado: str):
    """[I2] importa o treinador do BA1 so se o sha bater com o do run JSON."""
    import importlib.util
    s = sha256_file(TREINADOR_BA1)
    if s != sha_esperado:
        raise RuntimeError(f"[I2] sha do treinador BA1 {s} != run_json.script_sha256 {sha_esperado}")
    spec = importlib.util.spec_from_file_location("treinador_ba1", str(TREINADOR_BA1))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# mapa de caminho: os run JSON foram gravados no Windows da campanha; os
# dados vivem no mesmo drive espelhado no Linux (path novo, nao existe no
# congelado — este script roda so no Linux).
_WIN_DRIVE_PREFIXES = [
    (r"F:\TOPO_RF_DOWNLOAD_DRIVE", "/trabalho/TOPO_RF_DOWNLOAD_DRIVE"),
    ("F:/TOPO_RF_DOWNLOAD_DRIVE", "/trabalho/TOPO_RF_DOWNLOAD_DRIVE"),
]


def windows_para_linux(path_str: str) -> str:
    """Converte caminho Windows do run JSON para o espelho Linux (novo; nao
    existe equivalente no congelado, que so rodava no Windows da campanha)."""
    p = path_str
    for win, lin in _WIN_DRIVE_PREFIXES:
        if p.startswith(win):
            p = lin + p[len(win):]
            break
    return p.replace("\\", "/")


# ==========================================================================
# Helpers — sha256_file/sha256_idx: COPIA de train_gnn_c0_spatial.py:312-323
# ==========================================================================
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    """Hash canonico de um conjunto de indices: int64, ordenado, little-endian.
    train_gnn_c0_spatial.py:320-323 (== train_mlp:445-448)."""
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


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
# [C1-5a] Projecao equiretangular — COPIA de train_gnn_c0_spatial.py:351-373
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


# ==========================================================================
# [C1-5b] Split 3 vias — COPIA de train_gnn_c0_spatial.py:412-487
# ==========================================================================
def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                          fracs: tuple[float, float, float], split_seed: int,
                          SpatialKFold, log) -> tuple[dict, dict]:
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    skf = SpatialKFold(n_splits=3, buffer_km=buffer_km,
                        grid_size_km=grid_km, random_state=split_seed)
    group_ids = skf._assign_groups(pos_km)
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

    kw = dict(compact_nodes=False, balanced_tree=False)
    if buffer_km > 0:
        if m_tr.any() and m_va.any():
            tree_tr = cKDTree(pos_km[m_tr], **kw)
            d, _ = tree_tr.query(pos_km[m_va], k=1, workers=-1)
            idx_va = np.where(m_va)[0]
            m_va[idx_va[d < buffer_km]] = False
        m_trva = m_tr | m_va
        if m_trva.any() and m_te.any():
            tree_trva = cKDTree(pos_km[m_trva], **kw)
            d, _ = tree_trva.query(pos_km[m_te], k=1, workers=-1)
            idx_te = np.where(m_te)[0]
            m_te[idx_te[d < buffer_km]] = False

    parts = {"train": np.where(m_tr)[0].astype(np.int64),
             "val": np.where(m_va)[0].astype(np.int64),
             "test": np.where(m_te)[0].astype(np.int64)}
    info = {"grid_km": grid_km, "buffer_km": buffer_km, "n_blocos_total": int(n_g),
            "n_nos_apos_buffer": {k: int(len(v)) for k, v in parts.items()}}
    return parts, info


# ==========================================================================
# [C2, v2] fixar_seed — NOVO (nao existe no congelado, que fixa seed uma vez
# no inicio do treino inteiro; aqui fixa seed logo antes de amostrar
# vizinhos/lotes na inferencia, achado A2 do portao ia-bug-silencioso: o E3
# v1 nunca chamava nenhum destes, deixando a amostragem do NeighborLoader
# nao reprodutiva run-a-run).
# ==========================================================================
def fixar_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================================
# _scatter_por_semente — COPIA de train_gnn_c0_spatial.py:635-648
# ==========================================================================
def _scatter_por_semente(seed_ids: torch.Tensor, preds: torch.Tensor,
                          tgts: torch.Tensor, n_nodes: int):
    preds_ord = torch.zeros((n_nodes, preds.shape[1]), dtype=torch.float32)
    tgts_ord = torch.zeros((n_nodes, tgts.shape[1]), dtype=torch.float32)
    visto = torch.zeros(n_nodes, dtype=torch.bool)
    preds_ord[seed_ids] = preds.float()
    tgts_ord[seed_ids] = tgts.float()
    visto[seed_ids] = True
    return preds_ord, tgts_ord, visto


# ==========================================================================
# Metricas por populacao — nucleo err_rssi/err_pl e' COPIA de
# train_gnn_c0_spatial.py:666-668 (== train_mlp:790-792); o resto
# (mediana/IQR/pi) e' NOVO, pedido pelo protocolo E3 (nao existe no
# congelado, que so agregava "todos" e "cobertos" via mae_pl_db).
# ==========================================================================
def estatisticas_populacao(pred: torch.Tensor, tgt: torch.Tensor,
                            mask: torch.Tensor, com_pl: bool) -> dict:
    n = int(mask.sum())
    out = {"n": n}
    if n == 0:
        out.update({"mae_rssi_db": None, "mediana_pred_rssi_db": None,
                     "p25_pred_rssi_db": None, "p75_pred_rssi_db": None})
        if com_pl:
            out["mae_pl_db"] = None
        return out
    p, t = pred[mask], tgt[mask]
    err_rssi = (p[:, 3] - t[:, 3])          # COPIA train_gnn_c0_spatial.py:666
    out["mae_rssi_db"] = float(err_rssi.abs().mean())
    pred_rssi = p[:, 3].numpy()
    out["mediana_pred_rssi_db"] = float(np.median(pred_rssi))
    out["p25_pred_rssi_db"] = float(np.percentile(pred_rssi, 25))
    out["p75_pred_rssi_db"] = float(np.percentile(pred_rssi, 75))
    if com_pl:
        err_pl = (p[:, 0] - t[:, 0])        # COPIA train_gnn_c0_spatial.py:668
        out["mae_pl_db"] = float(err_pl.abs().mean())
    return out


# ==========================================================================
# [M1] MLPRFModel — COPIA de train_mlp_c0_spatial.py:691-724
# ==========================================================================
class MLPRFModel(nn.Module):
    def __init__(self, PhysicsConstrainedDecoder, terrain_dim: int,
                 larguras=MLP_LARGURAS, dropout: float = 0.1):
        super().__init__()
        dims = [terrain_dim] + list(larguras)
        camadas = []
        for i, (m, n) in enumerate(zip(dims, dims[1:])):
            camadas += [nn.Linear(m, n), nn.BatchNorm1d(n), nn.LeakyReLU(0.2)]
            if i < len(dims) - 2:
                camadas.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*camadas)
        self.decoder = PhysicsConstrainedDecoder(
            in_channels=dims[-1], hidden_channels=256,
            out_channels=5, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict:
        emb = self.encoder(x)
        preds = self.decoder(emb)
        return {"terrain_embeddings": emb, "predictions": preds}


def construir_modelo_mlp(base_dir: Path, terrain_dim: int) -> nn.Module:
    """train_mlp_c0_spatial.py:1359-1360 (mesma chamada)."""
    sys.path.append(str(base_dir / "02_models"))
    from rf_decoder import PhysicsConstrainedDecoder
    return MLPRFModel(PhysicsConstrainedDecoder, terrain_dim=terrain_dim,
                       larguras=MLP_LARGURAS, dropout=0.1)


def construir_modelo_gnn(base_dir: Path, terrain_dim: int, antenna_dim: int,
                          hidden_dim: int) -> nn.Module:
    """train_gnn_c0_spatial.py:1232-1235 (mesma chamada; GNNRFModel
    IMPORTADO de 02_models/gnn_rf_model.py, nao copiado)."""
    sys.path.append(str(base_dir / "02_models"))
    from gnn_rf_model import GNNRFModel
    return GNNRFModel(terrain_dim=terrain_dim, antenna_dim=antenna_dim,
                       hidden_dim=hidden_dim, num_layers=4, heads=4,
                       edge_dim=2, output_dim=5, dropout=0.1,
                       use_physics_constraints=True)


# ==========================================================================
# rodar_avaliacao — MLP: deriva de train_mlp_c0_spatial.py:1287-1302 (lotes)
#                        + :1426-1443 (rodar_avaliacao)
# ==========================================================================
def rodar_avaliacao_mlp(model: nn.Module, x_test: torch.Tensor,
                         tgt_test: torch.Tensor, eval_bs: int,
                         device: str, use_amp: bool, rng_seed: int) -> torch.Tensor:
    # [C2] seed fixada imediatamente antes de construir os lotes.
    fixar_seed(rng_seed)
    model.eval()
    n_p = x_test.shape[0]
    P = []
    with torch.no_grad():
        for i in range(0, n_p, eval_bs):
            xb = x_test[i:i + eval_bs].to(device)
            # [C1] mesmo autocast do congelado: train_mlp_c0_spatial.py:1436-1437
            # (device_type "cuda", dtype padrao, use_amp=(device=='cuda')).
            ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(xb)
            P.append(out["predictions"].float().detach().cpu())
    return torch.cat(P, dim=0)


# ==========================================================================
# induzir_subgrafo_teste — deriva de train_gnn_c0_spatial.py:547-600
# (induzir_particao), restrita a UMA particao (o congelado induzia as 3
# porque TREINAVA; aqui so se infere sobre "test").
# ==========================================================================
def induzir_subgrafo_teste(x_full: torch.Tensor, pos_deg: torch.Tensor,
                            tgt_all: torch.Tensor, ant_x: torch.Tensor,
                            ei_at_full: torch.Tensor, ea_at_full,
                            ei_tt_full: torch.Tensor, ea_tt_full,
                            idx_test_global: np.ndarray, n_antenna: int,
                            n_ter_total: int, log):
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import bipartite_subgraph, subgraph

    idx_t = torch.from_numpy(idx_test_global).long()
    ant_all = torch.arange(n_antenna, dtype=torch.long)

    d = HeteroData()
    d["terrain"].x = x_full[idx_t]
    d["terrain"].pos = pos_deg[idx_t]
    d["terrain"].rf_targets = tgt_all[idx_t]
    d["antenna"].x = ant_x
    d["antenna"].num_nodes = n_antenna

    ei_at_new, ea_at_new = bipartite_subgraph(
        (ant_all, idx_t), ei_at_full, ea_at_full, relabel_nodes=True,
        size=(n_antenna, n_ter_total))
    d[ET_AT].edge_index = ei_at_new
    if ea_at_new is not None:
        d[ET_AT].edge_attr = ea_at_new
    d[ET_TA].edge_index = ei_at_new[[1, 0]]

    ei_tt_new, ea_tt_new = subgraph(idx_t, ei_tt_full, ea_tt_full,
                                     relabel_nodes=True, num_nodes=n_ter_total)
    d[ET_TT].edge_index = ei_tt_new
    d[ET_TT].edge_attr = (ea_tt_new if ea_tt_new is not None
                           else torch.zeros((ei_tt_new.shape[1], 2), dtype=torch.float32))

    # [C4, v2] guarda de particao degenerada + log de arestas_cruzando_fronteira,
    # portados de train_gnn_c0_spatial.py:583-600 (calculo) e :1143-1156
    # (RuntimeError). O congelado so verifica no TREINO; aqui e' o mesmo
    # calculo restrito ao subgrafo de teste (achado A3 do portao).
    n_p = int(idx_t.numel())
    info = {
        "n_terrain": n_p, "n_antenna": int(n_antenna),
        "n_arestas_ant_ter": int(ei_at_new.shape[1]),
        "n_arestas_ter_ter": int(ei_tt_new.shape[1]),
        "arestas_cruzando_fronteira": int(
            (ei_at_new[1] >= n_p).sum() + (ei_tt_new[0] >= n_p).sum()
            + (ei_tt_new[1] >= n_p).sum()),
    }
    log(f"  [test] {n_p:,} nos terrain | ant->ter {info['n_arestas_ant_ter']:,} | "
        f"ter->ter {info['n_arestas_ter_ter']:,} | "
        f"cruzando fronteira: {info['arestas_cruzando_fronteira']}")
    if info["n_arestas_ant_ter"] == 0:
        msg = ("PARTICAO DE TESTE DEGENERADA: 0 arestas antena->terreno no "
               "subgrafo de teste induzido — nenhuma metrica dali passaria "
               "pelo caminho de mensagem da antena.")
        log(f"  ERRO: {msg}")
        raise RuntimeError(msg)

    return d, info


def rodar_avaliacao_gnn(model: nn.Module, graph, eval_bs: int, k_antenna: int,
                         k_terrain: int, num_workers: int, device: str,
                         use_amp: bool, rng_seed: int) -> torch.Tensor:
    """train_gnn_c0_spatial.py:1171-1186 (loader) + :1290-1305 (forward)."""
    from torch_geometric.loader import NeighborLoader

    nn_kw = {ET_AT: [k_antenna], ET_TT: [k_terrain], ET_TA: [k_antenna]}
    # [C2] seed fixada imediatamente antes de construir o NeighborLoader —
    # achado A2 do portao: k_antenna/k_terrain sao contagens finitas, e sem
    # seed a amostragem de vizinhos consumia o estado global do RNG e nao
    # era reprodutivel run-a-run.
    fixar_seed(rng_seed)
    loader = NeighborLoader(data=graph, num_neighbors=nn_kw,
                             input_nodes=("terrain", None), batch_size=eval_bs,
                             shuffle=False, num_workers=num_workers)
    model.eval()
    n_p = graph["terrain"].x.shape[0]
    P, I = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            bs = b["terrain"].batch_size
            # [C1] mesmo autocast do congelado: train_gnn_c0_spatial.py:1298
            # (device_type "cuda", dtype padrao, use_amp=(device=='cuda')).
            ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(b)
            P.append(out["predictions"][:bs].float().detach().cpu())
            I.append(b["terrain"].n_id[:bs].cpu())
    preds_ord = torch.zeros((n_p, P[0].shape[1]), dtype=torch.float32)
    preds_ord[torch.cat(I)] = torch.cat(P)
    return preds_ord


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="E3: inferencia por populacao (cobertos/sentinela/todos) "
                     "dos checkpoints ja treinados mlpcf_*/c0c1cf_*. NAO TREINA.")
    p.add_argument("--celula", type=str, default="",
                    help="<cidade>_Q<n>, ex.: campinas_Q1")
    p.add_argument("--braco", type=str, choices=["mlp", "gnn", "ba1"], default="")  # [I1]
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sem-autocast", action="store_true",
                    help="[C1, v2] Forca FP32 puro no forward (nos dois "
                         "bracos), mesmo em CUDA. Padrao: autocast('cuda', "
                         "enabled=(device=='cuda')) identico ao congelado "
                         "(train_mlp_c0_spatial.py:1436-1437 / "
                         "train_gnn_c0_spatial.py:1298).")
    p.add_argument("--rng-seed", type=int, default=None,  # [faseC (d)] None -> --seed
                    help="[C2, v2] Seed de amostragem (torch/numpy/random/"
                         "cuda) fixada imediatamente antes do NeighborLoader "
                         "(gnn) ou dos lotes (mlp). Independente de --seed, "
                         "que so rotula arquivos de saida.")
    p.add_argument("--cpu", action="store_true",
                    help="Forca CPU mesmo se CUDA estiver disponivel.")
    p.add_argument("--somente-construcao", action="store_true",
                    help="So carrega checkpoint + instancia modelo "
                         "(load_state_dict(strict=True)); NAO le rf_data/grafo, "
                         "NAO faz forward. Prova a construcao sem montar o grafo.")
    p.add_argument("--orcamento-min", type=float, default=60.0,
                    help="Aborta e grava estado parcial se o tempo de parede "
                         "ultrapassar este valor (minutos). 0 = sem limite.")
    p.add_argument("--saida-json", type=str, default="")
    p.add_argument("--saida-npz", type=str, default="")
    p.add_argument("--hash", action="store_true",
                    help="Imprime sha256 do proprio script e sai.")
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--evid-dir", type=str, default="")  # [faseC (b)] "" -> default por prefixo
    p.add_argument("--prefixo-run", type=str, choices=list(PREFIXOS_RUN),
                    default="c0c1cf",
                    help="[faseC (a)] prefixo do run_label do braco gnn.")
    p.add_argument("--num-workers", type=int, default=0,
                    help="NeighborLoader (so braco gnn, sem --somente-construcao).")
    args = p.parse_args(argv)
    # [faseC (a),(b),(d)] resolucao dos defaults dependentes
    if args.braco in ("mlp", "ba1") and args.prefixo_run != "c0c1cf":
        p.error("--prefixo-run so se aplica ao braco gnn")
    if not args.evid_dir:
        if args.braco == "ba1":                                   # [I1]
            args.evid_dir = DEFAULT_EVID_DIR_BA1
        else:
            args.evid_dir = DEFAULT_EVID_DIR if args.prefixo_run == "c0c1cf" else DEFAULT_EVID_DIR_FASEC
    if args.rng_seed is None:
        args.rng_seed = args.seed
    return args


def _parse_celula(celula: str) -> tuple[str, str]:
    if "_Q" not in celula:
        raise ValueError(f"--celula deve ser <cidade>_Q<n>, recebi {celula!r}")
    cidade, q = celula.rsplit("_Q", 1)
    return cidade, f"Q{q}"


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()

    if args.hash:
        print(sha256_file(script_path))
        return 0

    if not args.celula or not args.braco:
        print("ERRO: --celula e --braco sao obrigatorios (exceto com --hash).",
              file=sys.stderr)
        return 2

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] (+{time.perf_counter()-t_start:6.1f}s) {msg}", flush=True)

    def orcamento_min_restante() -> float:
        if args.orcamento_min <= 0:
            return float("inf")
        return args.orcamento_min - (time.perf_counter() - t_start) / 60.0

    rec: dict = {
        "artefato_tipo": "e3_inferencia_por_populacao",
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "fio": "c0-inference-2026-09-24",
        "protocolo": "E3_protocolo_preregistro.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "celula": args.celula,
        "braco": args.braco,
        "seed": args.seed,
        "somente_construcao": bool(args.somente_construcao),
        "status": "em_andamento",
        "ambiente": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_disponivel": torch.cuda.is_available(),
            "executavel": sys.executable,
        },
    }

    # [faseC (c)] nome default: gnn -> <prefixo>_<cidade>_<Q>_s<seed>; mlp inalterado
    _stem = (f"{args.prefixo_run}_{args.celula}_s{args.seed}" if args.braco == "gnn"
             else f"{PREFIXO_BA1}_{args.celula}_s{args.seed}" if args.braco == "ba1"   # [I1]
             else f"e3_{args.braco}_{args.celula}_s{args.seed}")
    _saida_dir = DEFAULT_SAIDA_DIR_BA1 if args.braco == "ba1" else DEFAULT_SAIDA_DIR  # [I1]
    out_json = Path(args.saida_json) if args.saida_json else (
        _saida_dir / f"{_stem}.json")
    out_npz = Path(args.saida_npz) if args.saida_npz else (
        _saida_dir / f"{_stem}.npz")
    rec["patch_faseC"] = {  # [faseC (e)] metadado
        "origem_script": "e3_inferencia_por_populacao.py",
        "origem_sha256": "c8e4356f1842fb2f9c10e6d8f0020a2990ab72b2f649feb79d9af98d960ad96c",
        "protocolo": "_revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md#Avaliacao",
        "prefixo_run": args.prefixo_run, "evid_dir": args.evid_dir,
        "rng_seed": args.rng_seed,
    }
    if args.braco == "ba1":                                       # [I1]
        rec["patch_ba1"] = {
            "origem_script": "_revisao_2026-10-14/fase_C/avaliacao/e3_inferencia_faseC.py",
            "origem_sha256": "bfa568f1bbe166b7293edc2ad5ba6361521b3788b488a7142f8cc50714d3c904",
            "protocolo": "PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md §7 + Decisao 15/10/2026",
            "treinador": str(TREINADOR_BA1),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def gravar():
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=_jsonable)

    def abortar_por_orcamento(fase: str):
        rec["status"] = f"abortado_orcamento_min_na_fase_{fase}"
        rec["custo"] = {"tempo_parede_s": round(time.perf_counter() - t_start, 1)}
        gravar()
        log(f"ORCAMENTO ESTOURADO na fase '{fase}' — abortando, estado gravado em {out_json}")
        sys.exit(3)

    gravar()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    rec["device"] = device
    # [C1, v2] mesma condicao do congelado: use_amp=(device=='cuda'); a nova
    # flag --sem-autocast forca FP32 puro mesmo em CUDA.
    use_amp = (device == "cuda") and not args.sem_autocast
    dtype_efetivo = "float16" if use_amp else "float32"
    rec["autocast"] = use_amp
    rec["dtype_efetivo"] = dtype_efetivo
    log(f"braco={args.braco} celula={args.celula} seed={args.seed} device={device} "
        f"somente_construcao={args.somente_construcao} autocast={use_amp} "
        f"dtype_efetivo={dtype_efetivo} rng_seed={args.rng_seed}")

    cidade, q = _parse_celula(args.celula)
    prefixo = ("mlpcf_" if args.braco == "mlp" else f"{PREFIXO_BA1}_" if args.braco == "ba1"
               else f"{args.prefixo_run}_")  # [faseC (a)] [I1]
    run_label = f"{prefixo}{cidade}_s{args.seed}_{q}_g10b2"
    run_dir = Path(args.evid_dir) / run_label
    run_json_path = run_dir / f"run_{run_label}.json"
    ckpt_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    rec["run_label"] = run_label
    rec["run_json_origem"] = str(run_json_path)
    rec["checkpoint_origem"] = str(ckpt_path)

    if not run_json_path.exists():
        raise FileNotFoundError(f"run JSON nao encontrado: {run_json_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint nao encontrado: {ckpt_path}")

    with open(run_json_path, "r", encoding="utf-8") as f:
        run_rec = json.load(f)
    cfg = run_rec["config"]

    fontes = {
        "run_json_sha256": sha256_file(run_json_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "script_sha256": rec["script_sha256"],
    }
    rec["_fontes"] = fontes
    trein_ba1 = None
    if args.braco == "ba1":                                       # [I2]
        trein_ba1 = carregar_treinador_ba1(run_rec["script_sha256"])
        rec["_fontes"]["treinador_ba1_sha256"] = run_rec["script_sha256"]
    gravar()
    log(f"run JSON e checkpoint lidos; sha256 do checkpoint={fontes['checkpoint_sha256'][:16]}...")

    base_dir = Path(args.base_dir)

    # ======================================================================
    # BRACO GNN, --somente-construcao: so carrega + instancia, sem forward,
    # sem tocar rf_data/grafo. Prova a construcao (load_state_dict strict).
    # ======================================================================
    if args.somente_construcao:
        terrain_dim = int(run_rec["modelo"]["terrain_dim"])
        if args.braco == "mlp":
            model = construir_modelo_mlp(base_dir, terrain_dim=terrain_dim)
        elif args.braco == "ba1":                                 # [I2]
            sys.path.append(str(base_dir / "02_models"))
            from rf_decoder import PhysicsConstrainedDecoder
            model = trein_ba1.RadioUNetStyleRF(PhysicsConstrainedDecoder,
                                               input_dim=int(run_rec["modelo"]["input_dim"]),
                                               hidden_dim=int(cfg["hidden_dim"]), decoder_dropout=0.1)
        else:
            antenna_dim = int(run_rec["modelo"]["antenna_dim"])
            hidden_dim = int(cfg["hidden_dim"])
            model = construir_modelo_gnn(base_dir, terrain_dim=terrain_dim,
                                          antenna_dim=antenna_dim, hidden_dim=hidden_dim)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_params_ckpt = sum(v.numel() for v in ck["model_state_dict"].values())
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device)
        n_params_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"[SOMENTE-CONSTRUCAO] load_state_dict(strict=True) OK | "
            f"n_params_checkpoint={n_params_ckpt:,} | n_params_treinaveis_modelo={n_params_tr:,} | "
            f"epoch_do_ckpt={ck.get('epoch')}")
        rec["status"] = "construcao_ok_sem_inferencia"
        rec["construcao"] = {
            "n_params_checkpoint": int(n_params_ckpt),
            "n_params_treinaveis_modelo": int(n_params_tr),
            "epoch_do_checkpoint": ck.get("epoch"),
            "strict_load_state_dict": True,
            "terrain_dim": terrain_dim,
        }
        rec["custo"] = {"tempo_parede_s": round(time.perf_counter() - t_start, 1)}
        gravar()
        log(f"JSON gravado: {out_json}")
        return 0

    # ======================================================================
    # PIPELINE COMPLETO (forward de verdade) — MLP sempre, GNN so sem
    # --somente-construcao.
    # ======================================================================
    if int(cfg.get("max_nodes", 0) or 0) != 0:
        raise NotImplementedError(
            f"config.max_nodes={cfg.get('max_nodes')} != 0: este run usou "
            f"janela_contigua (--max-nodes>0), que este script de inferencia "
            f"NAO reimplementa (ver docstring, secao 'O QUE NAO REPRODUZ').")

    rf_path = Path(windows_para_linux(run_rec["dataset"]["rf_data_file"]))
    graph_path = Path(windows_para_linux(run_rec["dataset"]["graph_file"]))
    expected_rf_sha = run_rec["dataset"].get("rf_data_sha256")
    if not expected_rf_sha:
        raise RuntimeError("run JSON sem dataset.rf_data_sha256 — nao da para "
                            "verificar ANTES de medir; abortando por seguranca.")

    if orcamento_min_restante() <= 0:
        abortar_por_orcamento("antes_do_hash_rf_data")

    log(f"Verificando sha256 de {rf_path.name} (ANTES de medir)...")
    actual_rf_sha = sha256_file(rf_path)
    rf_sha_ok = (actual_rf_sha == expected_rf_sha)
    rec["verificacoes"] = {
        "rf_data_sha256_esperado": expected_rf_sha,
        "rf_data_sha256_calculado": actual_rf_sha,
        "rf_data_sha256_ok": rf_sha_ok,
    }
    gravar()
    if not rf_sha_ok:
        rec["status"] = "erro_hash_rf_data_divergente"
        gravar()
        raise RuntimeError(f"rf_data_sha256 divergente para {rf_path}: "
                            f"esperado {expected_rf_sha}, calculado {actual_rf_sha}. "
                            f"Abortando SEM medir.")
    log(f"sha256 OK: {actual_rf_sha[:16]}...")

    if orcamento_min_restante() <= 0:
        abortar_por_orcamento("apos_hash_rf_data")

    # ---- carregar rf_data (tensores por no; sem gpu.pt aqui) ----
    log(f"Carregando rf_data: {rf_path} ({rf_path.stat().st_size/1e9:.1f} GB)")
    load_kw = dict(map_location="cpu", weights_only=False)
    rf_data = torch.load(rf_path, **load_kw)

    ty = rf_data["terrain"].y
    n_ter_total = int(ty.shape[0])
    if getattr(rf_data["terrain"], "features_raw", None) is None:
        raise RuntimeError("rf_data sem features_raw embutido — fallback do "
                            "gpu.pt para features [M2] NAO implementado neste "
                            "script (nao ocorreu em nenhum dos runs conferidos).")
    feats = rf_data["terrain"].features_raw
    feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
    feats = feats.float()

    if not hasattr(rf_data["terrain"], "dist_nearest_m"):
        raise RuntimeError("rf_data sem dist_nearest_m embutido — fallback "
                            "Haversine (RFDiagnosticMetrics) NAO implementado "
                            "neste script (nao ocorreu em nenhum dos runs conferidos).")
    dist_all = rf_data["terrain"].dist_nearest_m.float()

    pos_deg = rf_data["terrain"].pos.float()
    tgt_all = torch.as_tensor(ty).float()
    ant_x = torch.as_tensor(rf_data["antenna"].x).float()
    n_antenna = int(ant_x.shape[0])

    ei_at_full = ea_at_full = ei_tt_full = ea_tt_full = None
    if args.braco in ("gnn", "ba1"):                              # [I3] ba1 usa as arestas ant->ter
        ei_at_full = rf_data[ET_AT].edge_index.long()
        ea_at_full = (rf_data[ET_AT].edge_attr.float()
                      if hasattr(rf_data[ET_AT], "edge_attr") else None)

    del rf_data
    gc.collect()
    log(f"Dados por no: {n_ter_total:,} terrain | {n_antenna} antenas")

    # ---- [C1-5a] projecao + [C1-5b] split ----
    pos_m, geo = latlon_graus_para_metros(pos_deg)
    rec["geometria"] = geo

    sys.path.append(str(base_dir / "03_training"))
    from spatial_cv import SpatialKFold

    grid_km = float(cfg["grid_km"])
    buffer_km = float(cfg["buffer_km"])
    fracs = tuple(float(v) for v in cfg["split_frac"].split(","))
    split_seed = int(cfg["split_seed"])
    log(f"Split espacial: grid={grid_km}km buffer={buffer_km}km "
        f"fracs={fracs} split_seed={split_seed}")
    parts, split_info = split_espacial_3vias(pos_m, grid_km, buffer_km, fracs,
                                              split_seed, SpatialKFold, log)
    rec["split"] = split_info

    if orcamento_min_restante() <= 0:
        abortar_por_orcamento("apos_split")

    rec["particoes"] = {}
    sha_ok_all = True
    for k, loc in parts.items():
        sha_calc = sha256_idx(loc)
        sha_esp = run_rec["particoes"][k]["idx_sha256_global"]
        ok = (sha_calc == sha_esp)
        sha_ok_all = sha_ok_all and ok
        rec["particoes"][k] = {
            "n": int(loc.size),
            "idx_sha256_global_recomputado": sha_calc,
            "idx_sha256_global_run_json": sha_esp,
            "ok": ok,
        }
        log(f"  particao {k}: n={loc.size:,} idx_sha256_global "
            f"{'OK' if ok else 'DIVERGENTE'}")
    gravar()
    if not sha_ok_all:
        rec["status"] = "erro_hash_particao_divergente"
        gravar()
        raise RuntimeError("idx_sha256_global divergente em pelo menos uma "
                            "particao — abortando SEM medir (ver rec.particoes).")

    # ---- [D1] normalizacao col 17: SO estatistica de treino ----
    tr_loc = torch.from_numpy(parts["train"])
    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    x_full = torch.cat([feats, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    rec["normalizacao"] = {
        "dist_mean_train_recomputado": d_mean_tr,
        "dist_std_train_recomputado": d_std_tr,
        "dist_mean_train_run_json": run_rec.get("normalizacao", {}).get("dist_mean_train"),
        "dist_std_train_run_json": run_rec.get("normalizacao", {}).get("dist_std_train"),
    }
    log(f"[D1] normalizacao recomputada: mean={d_mean_tr:.1f}m std={d_std_tr:.1f}m "
        f"(run json: mean={rec['normalizacao']['dist_mean_train_run_json']})")

    if args.braco == "ba1":                                       # [I3]
        blk_bruto, meta_blk = trein_ba1.bloco_enlaces_bruto(
            ei_at_full[0].numpy(), ei_at_full[1].numpy(),
            ea_at_full[:, 0].numpy(), ea_at_full[:, 1].numpy(),
            ant_x.numpy(), n_ter_total, cfg["desempate_enlaces"], float(cfg["preench_altura"]))
        blk_norm, stats_blk = trein_ba1.normalizar_bloco(blk_bruto, parts["train"])
        ref = run_rec["entradas_ba1"]["normalizacao"]
        dmu = float(np.max(np.abs(np.array(stats_blk["media_treino"]) - np.array(ref["media_treino"]))))
        dsd = float(np.max(np.abs(np.array(stats_blk["desvio_treino"]) - np.array(ref["desvio_treino"]))))
        rec["entradas_ba1"] = {"desempate": cfg["desempate_enlaces"],
                               "preench_altura": cfg["preench_altura"],
                               "hist_n_enlaces_por_no": meta_blk["hist_n_enlaces_por_no"],
                               "max_dif_media_vs_run_json": dmu, "max_dif_desvio_vs_run_json": dsd}
        gravar()
        if dmu > 1e-5 or dsd > 1e-5:
            raise RuntimeError(f"[I3] estatistica do bloco diverge do run JSON (media {dmu}, desvio {dsd})")
        x_full = torch.cat([x_full, torch.from_numpy(blk_norm)], dim=1)
        del blk_bruto, blk_norm, ei_at_full, ea_at_full
        gc.collect()
        log(f"[I3] bloco de enlaces recomputado; |dif| media {dmu:.2e} desvio {dsd:.2e}")

    idx_test_global = parts["test"]
    # [C5, v2] amarra idx_test_global a parts['test'] (valor E sha256), em
    # vez de so' confiar que a variavel apontada para o forward e' a mesma
    # que passou pela checagem de hash acima — fecha o buraco que o mutante
    # M5 do portao explorou (nada impedia medir silenciosamente em 'val').
    assert np.array_equal(idx_test_global, parts["test"]), (
        "C5: idx_test_global diverge de parts['test'] — abortando antes de medir.")
    assert sha256_idx(idx_test_global) == rec["particoes"]["test"]["idx_sha256_global_recomputado"], (
        "C5: sha256(idx_test_global) diverge do sha256 ja conferido de "
        "parts['test'] — abortando antes de medir.")
    test_loc = torch.from_numpy(idx_test_global)
    tgt_test = tgt_all[test_loc]
    pl_valid_test = tgt_test[:, 0] < PL_TARGET_MAX_VALID

    eval_bs = int(cfg.get("eval_batch_size") or cfg["batch_size"])

    if orcamento_min_restante() <= 0:
        abortar_por_orcamento("antes_do_forward")

    # ======================================================================
    # FORWARD (sem treino) — MLP ou GNN
    # ======================================================================
    t_fwd = time.perf_counter()
    if args.braco == "mlp":
        x_test = x_full[test_loc]
        model = construir_modelo_mlp(base_dir, terrain_dim=x_full.shape[1])
        n_params_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_params_tr != N_PARAMS_ALVO_GNN:
            raise RuntimeError(f"paridade de parametros violada: MLP tem "
                                f"{n_params_tr:,}, esperado {N_PARAMS_ALVO_GNN:,}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device)
        pred_test = rodar_avaliacao_mlp(model, x_test, tgt_test, eval_bs, device,
                                         use_amp, args.rng_seed)
    elif args.braco == "ba1":                                     # [I2] [I4]
        use_amp = use_amp and (cfg.get("amp") == "on")
        dtype_efetivo = "float16" if use_amp else "float32"
        rec["autocast"] = use_amp
        rec["dtype_efetivo"] = dtype_efetivo
        x_test = x_full[test_loc]
        sys.path.append(str(base_dir / "02_models"))
        from rf_decoder import PhysicsConstrainedDecoder
        model = trein_ba1.RadioUNetStyleRF(PhysicsConstrainedDecoder, input_dim=x_full.shape[1],
                                           hidden_dim=int(cfg["hidden_dim"]), decoder_dropout=0.1)
        n_params_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_params_tr != int(run_rec["modelo"]["n_params_treinaveis"]):
            raise RuntimeError(f"n_params {n_params_tr:,} != run JSON {run_rec['modelo']['n_params_treinaveis']:,}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device)
        pred_test = rodar_avaliacao_mlp(model, x_test, tgt_test, eval_bs, device,
                                         use_amp, args.rng_seed)
    else:
        graph_bytes = graph_path.stat().st_size if graph_path.exists() else None
        log(f"Carregando estrutura de arestas ter->ter: {graph_path} "
            f"({(graph_bytes or 0)/1e9:.1f} GB) — necessario p/ forward GNN")
        sd = torch.load(graph_path, **load_kw)
        t2t = sd["dem", "adjacent_to", "dem"]
        ei_tt_full = t2t.edge_index.long().clone()
        ea_tt_full = (t2t.edge_attr.float().clone()
                      if hasattr(t2t, "edge_attr") and t2t.edge_attr is not None else None)
        del sd, t2t
        gc.collect()

        if orcamento_min_restante() <= 0:
            abortar_por_orcamento("apos_carga_grafo_ter_ter")

        graph, subgrafo_teste_info = induzir_subgrafo_teste(
            x_full, pos_deg, tgt_all, ant_x, ei_at_full, ea_at_full,
            ei_tt_full, ea_tt_full, idx_test_global, n_antenna, n_ter_total, log)
        rec["subgrafo_teste"] = subgrafo_teste_info
        gravar()
        del ei_tt_full, ea_tt_full, ei_at_full, ea_at_full
        gc.collect()

        antenna_dim = int(run_rec["modelo"]["antenna_dim"])
        hidden_dim = int(cfg["hidden_dim"])
        model = construir_modelo_gnn(base_dir, terrain_dim=x_full.shape[1],
                                      antenna_dim=antenna_dim, hidden_dim=hidden_dim)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device)
        k_antenna = int(cfg["k_antenna"])
        k_terrain = int(cfg["k_terrain"])
        pred_test = rodar_avaliacao_gnn(model, graph, eval_bs, k_antenna,
                                         k_terrain, args.num_workers, device,
                                         use_amp, args.rng_seed)

    tempo_forward_s = time.perf_counter() - t_fwd
    log(f"Forward concluido em {tempo_forward_s:.1f}s | device={device}")

    # ======================================================================
    # METRICAS POR POPULACAO
    # ======================================================================
    mask_todos = torch.ones(pred_test.shape[0], dtype=torch.bool)
    mask_cobertos = pl_valid_test
    mask_sentinela = ~pl_valid_test
    n_todos = int(mask_todos.sum())
    n_sentinela = int(mask_sentinela.sum())

    populacoes = {
        "todos": estatisticas_populacao(pred_test, tgt_test, mask_todos, com_pl=False),
        "cobertos": estatisticas_populacao(pred_test, tgt_test, mask_cobertos, com_pl=True),
        "sentinela": estatisticas_populacao(pred_test, tgt_test, mask_sentinela, com_pl=False),
    }
    populacoes["sentinela"]["frac_pi_sentinela"] = (n_sentinela / n_todos) if n_todos else None
    rec["populacoes"] = populacoes

    # ---- testemunha: mae_rssi_db (todos) e mae_pl_db (cobertos) EXATAMENTE
    # como metricas_particao() do congelado, lado a lado com o run JSON ----
    sel_run = run_rec["selecao"]["test_no_melhor_ckpt"]
    mae_rssi_novo = populacoes["todos"]["mae_rssi_db"]
    mae_pl_novo = populacoes["cobertos"]["mae_pl_db"]
    diff_rssi = abs(mae_rssi_novo - sel_run["mae_rssi_db"])
    diff_pl = abs(mae_pl_novo - sel_run["mae_pl_db"])
    rec["testemunha_contra_run_json"] = {
        "mae_rssi_db_recomputado": mae_rssi_novo,
        "mae_rssi_db_run_json": sel_run["mae_rssi_db"],
        "diff_abs_mae_rssi_db": diff_rssi,
        "mae_rssi_db_criterio_1e-3_ok": diff_rssi <= 1e-3,
        "mae_pl_db_recomputado": mae_pl_novo,
        "mae_pl_db_run_json": sel_run["mae_pl_db"],
        "diff_abs_mae_pl_db": diff_pl,
        "mae_pl_db_criterio_1e-3_ok": diff_pl <= 1e-3,
        "mae_rssi_db_criterio_protocolo_2.5e-4_ok": diff_rssi <= 2.5e-4,   # [I5]
    }
    log(f"TESTEMUNHA: mae_rssi_db novo={mae_rssi_novo:.6f} run_json="
        f"{sel_run['mae_rssi_db']:.6f} diff={diff_rssi:.2e} "
        f"({'OK' if diff_rssi <= 1e-3 else 'DIVERGENTE'})")
    log(f"TESTEMUNHA: mae_pl_db   novo={mae_pl_novo:.6f} run_json="
        f"{sel_run['mae_pl_db']:.6f} diff={diff_pl:.2e} "
        f"({'OK' if diff_pl <= 1e-3 else 'DIVERGENTE'})")

    # ======================================================================
    # SAIDA .npz — indice global do no, alvo, predicao, mascara de populacao
    # ======================================================================
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        idx_global=idx_test_global.astype(np.int64),
        target=tgt_test.numpy().astype(np.float32),
        pred=pred_test.numpy().astype(np.float32),
        sentinela=mask_sentinela.numpy().astype(np.bool_),
    )
    rec["saida_npz"] = str(out_npz)
    rec["saida_npz_sha256"] = sha256_file(out_npz)
    log(f".npz gravado: {out_npz}")

    rec["config_efetiva"] = {
        "grid_km": grid_km, "buffer_km": buffer_km, "split_frac": list(fracs),
        "split_seed": split_seed, "eval_batch_size": eval_bs,
        "k_antenna": cfg.get("k_antenna"), "k_terrain": cfg.get("k_terrain"),
        "hidden_dim": cfg.get("hidden_dim"), "max_nodes": cfg.get("max_nodes"),
        # [C1/C2, v2]
        "autocast": use_amp, "dtype_efetivo": dtype_efetivo,
        "sem_autocast_flag": bool(args.sem_autocast), "rng_seed": args.rng_seed,
    }
    rec["custo"] = {
        "tempo_parede_total_s": round(time.perf_counter() - t_start, 1),
        "tempo_forward_s": round(tempo_forward_s, 1),
    }
    rec["status"] = "provisorio_ate_contra_auditoria"
    gravar()
    log(f"JSON gravado: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
