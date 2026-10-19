# -*- coding: utf-8 -*-
"""Guarda de configuracao do braco BA1 (RadioUNet-style in-house adaptation) - R2-3, R4 IEEE Access.

Protocolo: _revisao_2026-10-14/PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md, "Decisao do dono (15/10/2026)" e §4/§6.
Padrao: copia do metodo de fase_C/guarda_config.py (sha256 conferivel no RELATORIO_BA1.md), sem monkeypatch: o treinador
do BA1 e um arquivo novo (fase_BA1/train_ba1_radiounet_style.py), conferido por sha256 fixo abaixo.

O que esta guarda faz (e SO isto):
  * monta o argv de uma corrida a partir do run JSON C0 (c0c1cf_*) da mesma celula + as chaves do BA1 em config_ba1.json;
  * compara a config efetiva com a do C0: so podem diferir caminhos (graph_dir, base_dir, evid_dir), run_label,
    transfer_from, mmap e as chaves do BA1 (weight_decay, amp, desempate_enlaces, preench_altura, e scheduler/max_norm
    quando config_ba1.json as fixa). Qualquer outra diferenca aborta (GuardaErro);
  * chave do BA1 com valor null (ABERTA, decisao do dono) recusa o modo lote; o smoke usa smoke_provisorio;
  * B1 (transferencia) antes e depois, S2 (concluida por conteudo), sha dos insumos contra HASHES_SHA256.txt;
  * guarda de particao (§4): idx_sha256_global de train/val/test == o do run JSON C0 da mesma celula (lote);
  * POS do BA1: classe RadioUNetStyleRF, sem passo terreno-terreno, entrada 44, auditoria das sementes de gradiente
    igual a particao de treino em todas as epocas, censo de gradiente gravado.

Uso de linha de comando:  python guarda_ba1.py concluida|pre|pos --cidade bauru --seed 42 --q 1 [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

sys.dont_write_bytecode = True

FASE_BA1 = Path(__file__).resolve().parent
REV = FASE_BA1.parent
EVID_C0 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO")
TREINOS_C0 = EVID_C0 / "treinos"
TREINADOR = FASE_BA1 / "train_ba1_radiounet_style.py"
TREINADOR_SHA256 = "593a3514977732e948559a833020ae1b994c2d22826582dbf697f18812296fd5"
CONFIG_BA1 = FASE_BA1 / "config_ba1.json"
GRAPH_DIR = "/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3"
BASE_DIR = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2"
HASHES_TXT = Path(GRAPH_DIR) / "HASHES_SHA256.txt"
EVID_DIR = FASE_BA1 / "treinos"
EVID_DIR_SMOKE = FASE_BA1 / "smoke_treinos"
CACHE_SHA = REV / "fase_C" / "insumos_sha_cache.json"   # SO LEITURA (cache da fase C); gravacao vai no cache proprio
CACHE_SHA_BA1 = FASE_BA1 / "insumos_sha_cache.json"

BRACO = "ba1rus"
CIDADES = ("bauru", "campinas", "lins", "sorocaba")
SEEDS = (42, 43, 44, 45, 46)
QS = (1, 2, 3, 4)

CHAVES_LIVRES = ("graph_dir", "base_dir", "evid_dir", "run_label", "transfer_from", "mmap")
CHAVES_BA1 = ("weight_decay", "scheduler", "max_norm", "amp", "desempate_enlaces", "preench_altura")
CHAVES_BA1_NOVAS = ("weight_decay", "amp", "desempate_enlaces", "preench_altura", "selftest_ba1")
SMOKE_VALORES = {"epochs": 2, "smoke": True, "max_nodes": 200000, "batch_size": 8192, "num_workers": 0,
                 "hash": False, "no_baselines": True}
SUFIXO_SMOKE = "_smoke"
BOOL_FLAGS = ("smoke", "selftest", "hash", "mmap", "no_baselines", "selftest_ba1")
INPUT_DIM_BA1 = 44


class GuardaErro(RuntimeError):
    """Violacao do protocolo: a corrida NAO pode seguir."""

    def __init__(self, codigo: str, msg: str):
        super().__init__(f"[{codigo}] {msg}")
        self.codigo = codigo
        self.msg = msg


def sha256_arquivo(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_TREINADOR_MOD = None


def carregar_treinador():
    global _TREINADOR_MOD
    if _TREINADOR_MOD is None:
        conferir_sha_treinador()
        spec = importlib.util.spec_from_file_location("treino_ba1", str(TREINADOR))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _TREINADOR_MOD = m
    return _TREINADOR_MOD


def conferir_sha_treinador() -> str:
    s = sha256_arquivo(TREINADOR)
    if s != TREINADOR_SHA256:
        raise GuardaErro("SHA_TREINADOR", f"sha do treinador {s} != esperado {TREINADOR_SHA256}")
    return s


def rotulo(cidade: str, seed: int, q: int, sufixo: str = "") -> str:
    return f"{BRACO}_{cidade}_s{seed}_Q{q}_g10b2{sufixo}"


def rotulo_c0(cidade: str, seed: int, q: int) -> str:
    return f"c0c1cf_{cidade}_s{seed}_Q{q}_g10b2"


def run_json_path(evid_dir, label: str) -> Path:
    return Path(evid_dir) / label / f"run_{label}.json"


def ckpt_path(evid_dir, label: str) -> Path:
    return Path(evid_dir) / label / "checkpoints" / "checkpoint_best.pt"


def carregar_c0(cidade: str, seed: int, q: int) -> dict:
    p = run_json_path(TREINOS_C0, rotulo_c0(cidade, seed, q))
    if not p.is_file():
        raise GuardaErro("C0_AUSENTE", f"run JSON C0 de referencia nao existe: {p}")
    return json.loads(p.read_text())


def validar_celula(cidade: str, seed: int, q: int) -> None:
    if cidade not in CIDADES or seed not in SEEDS or q not in QS:
        raise GuardaErro("ARGUMENTO", f"celula fora do desenho: {cidade} s{seed} Q{q}")


def evid_do_modo(modo: str) -> Path:
    return EVID_DIR_SMOKE if modo == "smoke" else EVID_DIR


def sufixo_do_modo(modo: str) -> str:
    return SUFIXO_SMOKE if modo == "smoke" else ""


def transfer_esperado(cidade, seed, q, evid_dir, sufixo) -> str:
    if q == 1:
        return ""
    return str(ckpt_path(evid_dir, rotulo(cidade, seed, q - 1, sufixo)))


# ----------------------------------------------------------------------------------------------------------------
# chaves do BA1 (config_ba1.json)
# ----------------------------------------------------------------------------------------------------------------
def config_ba1_sha() -> str:
    return sha256_arquivo(CONFIG_BA1)


def abertas() -> list[str]:
    c = json.loads(CONFIG_BA1.read_text())["chaves"]
    return [k for k in CHAVES_BA1 if c[k]["valor"] is None]


def valores_ba1(modo: str) -> dict:
    c = json.loads(CONFIG_BA1.read_text())["chaves"]
    if set(c) != set(CHAVES_BA1):
        raise GuardaErro("CONFIG_BA1", f"chaves de config_ba1.json {sorted(c)} != {sorted(CHAVES_BA1)}")
    if modo == "smoke":
        return {k: c[k]["smoke_provisorio"] for k in CHAVES_BA1}
    ab = [k for k in CHAVES_BA1 if c[k]["valor"] is None]
    if ab:
        raise GuardaErro("ABERTA", f"escolhas ABERTAS sem decisao do dono em config_ba1.json: {ab} "
                                   f"(o modo lote nao roda com palpite)")
    return {k: c[k]["valor"] for k in CHAVES_BA1}


# ----------------------------------------------------------------------------------------------------------------
# argv a partir do C0
# ----------------------------------------------------------------------------------------------------------------
def montar_argv(cidade: str, seed: int, q: int, modo: str = "lote", mmap: bool = False, evid_dir=None,
                valores=None) -> list[str]:
    """argv do treinador: config do run JSON C0 da celula, caminhos Linux, rotulo do BA1, transfer_from do BA1 (Q>=2),
    chaves do BA1. valores: so para teste (dry-run com valores provisorios); em producao vem de config_ba1.json."""
    validar_celula(cidade, seed, q)
    c0 = carregar_c0(cidade, seed, q)["config"]
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    cfg = dict(c0)
    cfg["graph_dir"] = GRAPH_DIR
    cfg["base_dir"] = BASE_DIR
    cfg["evid_dir"] = str(evid)
    cfg["run_label"] = rotulo(cidade, seed, q, suf)
    cfg["transfer_from"] = transfer_esperado(cidade, seed, q, evid, suf)
    cfg["mmap"] = bool(mmap) or modo == "smoke"
    cfg.update(valores if valores is not None else valores_ba1(modo))
    if modo == "smoke":
        cfg.update(SMOKE_VALORES)
    argv: list[str] = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if k in BOOL_FLAGS:
            if v:
                argv.append(flag)
        elif v == "" or v is None:
            continue
        else:
            argv += [flag, str(v)]
    return argv


def config_do_argv(argv: list[str]) -> dict:
    m = carregar_treinador()
    return vars(m.parse_args(list(argv)))


def diff_config(efetiva: dict, cidade: str, seed: int, q: int, modo: str, evid_dir=None, valores=None) -> list[dict]:
    validar_celula(cidade, seed, q)
    c0 = carregar_c0(cidade, seed, q)["config"]
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    esperado = {
        "graph_dir": GRAPH_DIR, "base_dir": BASE_DIR, "evid_dir": str(evid),
        "run_label": rotulo(cidade, seed, q, suf),
        "transfer_from": transfer_esperado(cidade, seed, q, evid, suf),
        "selftest_ba1": False,
    }
    esperado.update(valores if valores is not None else valores_ba1(modo))
    if modo == "smoke":
        esperado.update(SMOKE_VALORES)
    viol: list[dict] = []
    faltam = sorted(set(c0) - set(efetiva))
    sobram = sorted(set(efetiva) - set(c0) - set(CHAVES_BA1_NOVAS))
    if faltam or sobram:
        viol.append({"chave": "<conjunto de chaves>", "faltam": faltam, "sobram": sobram})
    for k in sorted(set(CHAVES_BA1_NOVAS) - set(efetiva)):
        viol.append({"chave": k, "motivo": "chave do BA1 ausente da config efetiva"})
    for k in sorted(set(efetiva)):
        ef = efetiva[k]
        if k == "mmap":
            if not isinstance(ef, bool):
                viol.append({"chave": k, "efetivo": ef, "esperado": "bool"})
            continue
        if k in esperado:
            if ef != esperado[k] or isinstance(ef, bool) != isinstance(esperado[k], bool):
                viol.append({"chave": k, "efetivo": ef, "esperado": esperado[k], "c0": c0.get(k)})
            continue
        if k in c0 and ef != c0[k]:
            viol.append({"chave": k, "efetivo": ef, "c0": c0[k], "motivo": "diverge do C0 e nao e chave livre"})
    return viol


def exigir_config(efetiva, cidade, seed, q, modo, evid_dir=None, valores=None) -> None:
    v = diff_config(efetiva, cidade, seed, q, modo, evid_dir, valores)
    if v:
        raise GuardaErro("CONFIG_DIVERGE", json.dumps(v, ensure_ascii=False, default=str))


# ----------------------------------------------------------------------------------------------------------------
# S2 - conclusao por conteudo
# ----------------------------------------------------------------------------------------------------------------
def epocas_esperadas(cidade, seed, q, modo) -> int:
    if modo == "smoke":
        return SMOKE_VALORES["epochs"]
    c0 = carregar_c0(cidade, seed, q)
    n = c0.get("custo", {}).get("n_epocas_rodadas")
    if n is None:
        n = len(c0["epocas"])
    return int(n)


def concluida_por_conteudo(cidade, seed, q, modo="lote", evid_dir=None) -> tuple[bool, str]:
    validar_celula(cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    label = rotulo(cidade, seed, q, sufixo_do_modo(modo))
    p = run_json_path(evid, label)
    if not p.is_file():
        return False, "run JSON ausente"
    try:
        rj = json.loads(p.read_text())
    except Exception as e:
        return False, f"run JSON ilegivel ({type(e).__name__})"
    if rj.get("run_label") != label:
        return False, f"run_label {rj.get('run_label')!r} != {label!r}"
    if bool(rj.get("smoke")) != (modo == "smoke"):
        return False, "flag smoke incompativel com o modo"
    sel = rj.get("selecao") or {}
    t = sel.get("test_no_melhor_ckpt")
    if not isinstance(t, dict) or not isinstance(t.get("mae_rssi_db"), (int, float)):
        return False, "selecao.test_no_melhor_ckpt ausente"
    n_ep = (rj.get("custo") or {}).get("n_epocas_rodadas")
    if n_ep is None:
        n_ep = len(rj.get("epocas") or [])
    esp = epocas_esperadas(cidade, seed, q, modo)
    if int(n_ep) != esp or len(rj.get("epocas") or []) != esp:
        return False, f"epocas rodadas {n_ep} / gravadas {len(rj.get('epocas') or [])} != {esp} do C0"
    ck = ckpt_path(evid, label)
    if not ck.is_file() or ck.stat().st_size == 0:
        return False, "checkpoint_best.pt ausente"
    return True, "ok"


# ----------------------------------------------------------------------------------------------------------------
# sha dos insumos
# ----------------------------------------------------------------------------------------------------------------
_LOCK_SHA = threading.Lock()


def hashes_oficiais() -> dict:
    d = {}
    for ln in HASHES_TXT.read_text().splitlines():
        p = ln.split()
        if len(p) >= 2:
            d[p[1].lstrip("*")] = p[0].lower()
    return d


def _cache_ler() -> dict:
    out = {}
    for c in (CACHE_SHA, CACHE_SHA_BA1):     # o da fase C so e lido
        if c.is_file():
            try:
                out.update(json.loads(c.read_text()))
            except Exception:
                pass
    return out


def conferir_sha_insumo(nome: str) -> str:
    ofic = hashes_oficiais()
    if nome not in ofic:
        raise GuardaErro("SHA_INSUMO", f"{nome} nao consta em HASHES_SHA256.txt")
    f = Path(GRAPH_DIR) / nome
    if not f.is_file():
        raise GuardaErro("SHA_INSUMO", f"insumo ausente: {f}")
    st = f.stat()
    with _LOCK_SHA:
        c = _cache_ler().get(nome)
        if c and c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns and c["sha256"] == ofic[nome]:
            return c["sha256"]
    s = sha256_arquivo(f)
    if s != ofic[nome]:
        raise GuardaErro("SHA_INSUMO", f"{nome}: sha {s} != HASHES_SHA256.txt {ofic[nome]}")
    with _LOCK_SHA:
        cache = json.loads(CACHE_SHA_BA1.read_text()) if CACHE_SHA_BA1.is_file() else {}
        cache[nome] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": s}
        tmp = CACHE_SHA_BA1.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1))
        os.replace(tmp, CACHE_SHA_BA1)
    return s


# ----------------------------------------------------------------------------------------------------------------
# ANTES e DEPOIS
# ----------------------------------------------------------------------------------------------------------------
def checar_pre(cidade, seed, q, argv, modo="lote", evid_dir=None, verificar_sha=True, valores=None) -> dict:
    validar_celula(cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    rel = {"fase": "pre", "braco": BRACO, "cidade": cidade, "seed": seed, "q": q, "modo": modo,
           "config_ba1_sha256": config_ba1_sha(), "abertas_em_config_ba1": abertas()}
    rel["sha_treinador"] = conferir_sha_treinador()
    ef = config_do_argv(argv)
    exigir_config(ef, cidade, seed, q, modo, evid, valores)
    rel["config_diff_ok"] = True
    c0 = carregar_c0(cidade, seed, q)
    ds = c0.get("dataset", {})
    f = Path(ef["graph_dir"]) / ef["rf_data_file"]
    if not f.is_file():
        raise GuardaErro("INSUMO", f"insumo ausente: {f}")
    if ds.get("rf_data_bytes") is not None and f.stat().st_size != ds["rf_data_bytes"]:
        raise GuardaErro("INSUMO", f"{f}: {f.stat().st_size} B != {ds['rf_data_bytes']} B do C0")
    if verificar_sha and modo != "smoke":
        rel.setdefault("sha_insumos", {})[ef["rf_data_file"]] = conferir_sha_insumo(ef["rf_data_file"])
    tf = ef["transfer_from"]
    esperado_tf = transfer_esperado(cidade, seed, q, evid, suf)
    if q == 1:
        if tf:
            raise GuardaErro("B1_PRE", f"Q1 nao pode ter transfer_from (veio {tf!r})")
    else:
        if tf != esperado_tf:
            raise GuardaErro("B1_PRE", f"transfer_from {tf!r} != esperado {esperado_tf!r}")
        p = Path(tf)
        if not p.is_file():
            raise GuardaErro("B1_PRE", f"transfer_from nao e arquivo existente: {tf}")
        if p.parent.parent.name != rotulo(cidade, seed, q - 1, suf):
            raise GuardaErro("B1_PRE", f"rotulo do checkpoint {p.parent.parent.name!r} nao e o do mesmo braco/Q anterior")
        ok, mot = concluida_por_conteudo(cidade, seed, q - 1, modo, evid)
        if not ok:
            raise GuardaErro("S2_PRE", f"Q{q-1} do BA1 NAO esta concluida por conteudo: {mot}")
        rel["transfer_from"] = tf
        rel["transfer_sha256"] = sha256_arquivo(p)
    return rel


def checar_pos(cidade, seed, q, modo="lote", evid_dir=None, log_path=None, valores=None) -> dict:
    validar_celula(cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    label = rotulo(cidade, seed, q, suf)
    rel = {"fase": "pos", "label": label, "modo": modo}
    p = run_json_path(evid, label)
    if not p.is_file():
        raise GuardaErro("POS_RUN_JSON", f"run JSON nao foi gravado: {p}")
    rj = json.loads(p.read_text())
    if rj.get("script_sha256") != TREINADOR_SHA256:
        raise GuardaErro("SHA_TREINADOR", f"run JSON registra script_sha256 {rj.get('script_sha256')}")
    exigir_config(rj.get("config", {}), cidade, seed, q, modo, evid, valores)
    rel["config_diff_ok"] = True
    # [A3 gate ia-bug-silencioso 18/10] amp 'on' sem CUDA cai em use_amp=False calado no treinador: recusar
    if (rj.get("config") or {}).get("amp") == "on":
        amp_ef = (rj.get("otimizacao_ba1") or {}).get("amp_efetivo")
        if amp_ef is not True:
            raise GuardaErro("AMP_POS", f"config.amp='on' mas otimizacao_ba1.amp_efetivo={amp_ef!r} (corrida sem AMP/GPU)")
        rel["amp_efetivo"] = True
    ok, mot = concluida_por_conteudo(cidade, seed, q, modo, evid)
    if not ok:
        raise GuardaErro("S2_POS", f"corrida nao concluida por conteudo: {mot}")
    rel["concluida_por_conteudo"] = True
    esperado_tf = transfer_esperado(cidade, seed, q, evid, suf)
    if q == 1:
        if rj.get("transfer_from"):
            raise GuardaErro("B1_POS", f"Q1 registra transfer_from de topo {rj.get('transfer_from')!r}")
    else:
        if rj.get("transfer_from") != esperado_tf:
            raise GuardaErro("B1_POS", f"run_json['transfer_from'] = {rj.get('transfer_from')!r} != {esperado_tf!r}")
    if log_path is not None:
        txt = Path(log_path).read_text(errors="replace") if Path(log_path).is_file() else ""
        linhas = [ln for ln in txt.splitlines() if "[TRANSFER]" in ln]
        if q == 1 and linhas:
            raise GuardaErro("B1_POS", "Q1 com linha [TRANSFER] no log")
        if q > 1 and not any(esperado_tf in ln for ln in linhas):
            raise GuardaErro("B1_POS", f"sem linha [TRANSFER] com {esperado_tf!r} no log {log_path}")
        rel["linha_transfer_no_log"] = (linhas[0] if linhas else None)
    if modo != "smoke":
        sha_rf = (rj.get("dataset") or {}).get("rf_data_sha256")
        if not sha_rf:
            raise GuardaErro("SHA_INSUMO", "run JSON sem dataset.rf_data_sha256")
        ofic = hashes_oficiais().get(rj["config"]["rf_data_file"])
        if sha_rf != ofic:
            raise GuardaErro("SHA_INSUMO", f"rf_data_sha256 {sha_rf} != HASHES {ofic}")
        rel["rf_data_sha256_ok"] = True
        # guarda de particao (§4): mesmos nos do C0 da celula
        c0p = carregar_c0(cidade, seed, q)["particoes"]
        for k in ("train", "val", "test"):
            a, b = rj["particoes"][k]["idx_sha256_global"], c0p[k]["idx_sha256_global"]
            if a != b:
                raise GuardaErro("PARTICAO", f"{k}: idx_sha256_global {a} != C0 {b}")
        rel["particoes_iguais_ao_c0"] = True
    else:
        rel["particoes_iguais_ao_c0"] = "nao_aplicavel_smoke (janela --max-nodes)"
    mo = rj.get("modelo") or {}
    if mo.get("classe") != "RadioUNetStyleRF" or mo.get("passo_terreno_terreno") is not False \
            or mo.get("input_dim") != INPUT_DIM_BA1:
        raise GuardaErro("MODELO_BA1", f"modelo gravado fora do desenho: {mo}")
    cg = rj.get("censo_gradiente_1o_lote") or {}
    if not cg or cg.get("n_params_com_gradiente", 0) <= 0:
        raise GuardaErro("GRADIENTE", f"censo de gradiente ausente ou vazio: {cg}")
    aud = [e.get("auditoria_sementes_ba1") or {} for e in rj.get("epocas", [])]
    if not aud or not all(a.get("igual_particao_treino") is True for a in aud):
        raise GuardaErro("SEMENTES", "alguma epoca sem auditoria de sementes == particao de treino")
    rel["modelo_ba1_ok"] = True
    rel["n_params_treinaveis"] = mo.get("n_params_treinaveis")
    rel["censo_gradiente"] = {k: cg.get(k) for k in ("n_params_nominais", "n_params_com_gradiente",
                                                     "parametros_grad_none", "parametros_grad_zero")}
    return rel


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acao", choices=["concluida", "pre", "pos", "abertas"])
    ap.add_argument("--cidade", choices=list(CIDADES), default="bauru")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mmap", action="store_true")
    a = ap.parse_args()
    modo = "smoke" if a.smoke else "lote"
    try:
        if a.acao == "abertas":
            print(json.dumps({"abertas": abertas(), "config_ba1_sha256": config_ba1_sha()}, ensure_ascii=False))
            sys.exit(0 if not abertas() else 1)
        if a.acao == "concluida":
            ok, mot = concluida_por_conteudo(a.cidade, a.seed, a.q, modo)
            print(json.dumps({"concluida": ok, "motivo": mot}, ensure_ascii=False))
            sys.exit(0 if ok else 1)
        if a.acao == "pre":
            argv = montar_argv(a.cidade, a.seed, a.q, modo, a.mmap)
            print(json.dumps(checar_pre(a.cidade, a.seed, a.q, argv, modo), indent=1, ensure_ascii=False))
        else:
            lp = evid_do_modo(modo) / rotulo(a.cidade, a.seed, a.q, sufixo_do_modo(modo)) / "stdout_wrapper.log"
            print(json.dumps(checar_pos(a.cidade, a.seed, a.q, modo, log_path=lp), indent=1, ensure_ascii=False))
    except GuardaErro as e:
        print(f"GUARDA ABORTA {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
