# -*- coding: utf-8 -*-
"""Guarda de configuracao da fase C (C1 sem regularizadores, C2 sem monotonicidade) - R4 IEEE Access.

Protocolo: _revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md (aprovado pelo dono em 14/10/2026) + Emenda 1
(achados B1, S2, S3, S4, A1 do gate ia-bug-silencioso).

O que esta guarda faz (e SO isto):
  * monta o argv de uma corrida a partir do run JSON C0 correspondente (mesma config, exceto os pesos ablados e os
    campos que podem diferir);
  * compara a config efetiva com a do C0: so podem diferir os pesos ablados, caminhos (graph_dir, base_dir, evid_dir),
    run_label, evid_dir, transfer_from e mmap. Qualquer outra diferenca aborta (GuardaErro);
  * B1 (transferencia silenciosa), ANTES: transfer_from e arquivo existente, do MESMO braco, cidade, seed e Q(k-1),
    e a corrida Q(k-1) esta concluida por conteudo; DEPOIS: run_json["transfer_from"] (campo de topo) == caminho
    esperado e linha [TRANSFER] no log da corrida;
  * S2 (conclusao por conteudo): selecao.test_no_melhor_ckpt presente, n_epocas_rodadas == epocas do C0 de
    referencia e checkpoint_best.pt existente (o campo "status" do treinador vale sempre "provisorio" e nao e usado).

Uso de linha de comando (depuracao):  python guarda_config.py concluida|pre|pos --braco c1noreg --cidade bauru --seed 42 --q 1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
from pathlib import Path

sys.dont_write_bytecode = True

FASE_C = Path(__file__).resolve().parent
REV = FASE_C.parent
EVID_C0 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO")
TREINOS_C0 = EVID_C0 / "treinos"
TREINADOR = EVID_C0 / "dados/scripts_congelados/train_gnn_c0_spatial.py"
TREINADOR_SHA256 = "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445"
GRAPH_DIR = "/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3"
BASE_DIR = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2"
HASHES_TXT = Path(GRAPH_DIR) / "HASHES_SHA256.txt"
EVID_DIR = FASE_C / "treinos"
EVID_DIR_SMOKE = FASE_C / "smoke_treinos"
CACHE_SHA = FASE_C / "insumos_sha_cache.json"

CIDADES = ("bauru", "campinas", "lins", "sorocaba")
SEEDS = (42, 43, 44, 45, 46)
QS = (1, 2, 3, 4)

# braco -> prefixo do rotulo, pesos ablados (flags do treinador) e se o peso FSPL da PhysicsRFLoss e zerado pelo wrapper
BRACOS = {
    "c1noreg": {"pesos_ablados": ("variance_weight", "ndvi_weight", "dist_grad_weight"), "fspl_zero": True},
    "c2nomono": {"pesos_ablados": ("dist_grad_weight",), "fspl_zero": False},
}
FSPL_PADRAO_C0 = 0.05  # default de PhysicsRFLoss.__init__ (physics_loss.py:40); o treinador nao o altera no C0

# chaves da config que PODEM diferir do C0 (alem dos pesos ablados do braco)
CHAVES_LIVRES = ("graph_dir", "base_dir", "evid_dir", "run_label", "transfer_from", "mmap")
# modo smoke: valores exatos esperados (fora do lote; o lote nunca roda em smoke)
SMOKE_VALORES = {"epochs": 1, "smoke": True, "max_nodes": 60000, "batch_size": 2048, "num_workers": 0,
                 "hash": False, "no_baselines": True}
SUFIXO_SMOKE = "_smoke"
BOOL_FLAGS = ("smoke", "selftest", "hash", "mmap", "no_baselines")


class GuardaErro(RuntimeError):
    """Violacao do protocolo: a corrida NAO pode seguir."""

    def __init__(self, codigo: str, msg: str):
        super().__init__(f"[{codigo}] {msg}")
        self.codigo = codigo
        self.msg = msg


# ----------------------------------------------------------------------------------------------------------------
# utilitarios
# ----------------------------------------------------------------------------------------------------------------
def sha256_arquivo(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_TREINADOR_MOD = None


def carregar_treinador():
    """Importa o treinador congelado (sem editar) apos conferir o sha. Retorna o modulo (cacheado)."""
    global _TREINADOR_MOD
    if _TREINADOR_MOD is None:
        conferir_sha_treinador()
        spec = importlib.util.spec_from_file_location("treino_congelado", str(TREINADOR))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _TREINADOR_MOD = m
    return _TREINADOR_MOD


def conferir_sha_treinador() -> str:
    s = sha256_arquivo(TREINADOR)
    if s != TREINADOR_SHA256:
        raise GuardaErro("SHA_TREINADOR", f"sha do treinador {s} != esperado {TREINADOR_SHA256}")
    return s


def rotulo(braco: str, cidade: str, seed: int, q: int, sufixo: str = "") -> str:
    return f"{braco}_{cidade}_s{seed}_Q{q}_g10b2{sufixo}"


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


def validar_celula(braco: str, cidade: str, seed: int, q: int) -> None:
    if braco not in BRACOS:
        raise GuardaErro("ARGUMENTO", f"braco invalido: {braco}")
    if cidade not in CIDADES or seed not in SEEDS or q not in QS:
        raise GuardaErro("ARGUMENTO", f"celula fora do desenho: {cidade} s{seed} Q{q}")


def evid_do_modo(modo: str) -> Path:
    return EVID_DIR_SMOKE if modo == "smoke" else EVID_DIR


def sufixo_do_modo(modo: str) -> str:
    return SUFIXO_SMOKE if modo == "smoke" else ""


def transfer_esperado(braco, cidade, seed, q, evid_dir, sufixo) -> str:
    """Caminho esperado do checkpoint de Q(k-1) do MESMO braco/cidade/seed ('' para Q1)."""
    if q == 1:
        return ""
    return str(ckpt_path(evid_dir, rotulo(braco, cidade, seed, q - 1, sufixo)))


# ----------------------------------------------------------------------------------------------------------------
# argv a partir do C0
# ----------------------------------------------------------------------------------------------------------------
def montar_argv(braco: str, cidade: str, seed: int, q: int, modo: str = "lote", mmap: bool = False,
                evid_dir=None) -> list[str]:
    """argv do treinador: config do run JSON C0 correspondente, com pesos ablados, caminhos Linux, rotulo do braco,
    transfer_from do mesmo braco (Q>=2). Em smoke aplica SMOKE_VALORES."""
    validar_celula(braco, cidade, seed, q)
    c0 = carregar_c0(cidade, seed, q)["config"]
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    cfg = dict(c0)
    cfg["graph_dir"] = GRAPH_DIR
    cfg["base_dir"] = BASE_DIR
    cfg["evid_dir"] = str(evid)
    cfg["run_label"] = rotulo(braco, cidade, seed, q, suf)
    cfg["transfer_from"] = transfer_esperado(braco, cidade, seed, q, evid, suf)
    cfg["mmap"] = bool(mmap) or modo == "smoke"
    for w in BRACOS[braco]["pesos_ablados"]:
        cfg[w] = 0.0
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
    """Config efetiva = o que o proprio parser do treinador congelado produz (vars(args))."""
    m = carregar_treinador()
    return vars(m.parse_args(list(argv)))


# ----------------------------------------------------------------------------------------------------------------
# diff config efetiva x C0
# ----------------------------------------------------------------------------------------------------------------
def diff_config(efetiva: dict, braco: str, cidade: str, seed: int, q: int, modo: str, evid_dir=None) -> list[dict]:
    """Lista de violacoes (vazia = ok). So podem diferir: pesos ablados do braco, graph_dir/base_dir/evid_dir,
    run_label, transfer_from, mmap. Cada chave livre tem VALOR ESPERADO exato (exceto mmap, que so precisa ser bool)."""
    validar_celula(braco, cidade, seed, q)
    c0 = carregar_c0(cidade, seed, q)["config"]
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    esperado = {
        "graph_dir": GRAPH_DIR, "base_dir": BASE_DIR, "evid_dir": str(evid),
        "run_label": rotulo(braco, cidade, seed, q, suf),
        "transfer_from": transfer_esperado(braco, cidade, seed, q, evid, suf),
    }
    for w in BRACOS[braco]["pesos_ablados"]:
        esperado[w] = 0.0
    if modo == "smoke":
        esperado.update(SMOKE_VALORES)
    viol: list[dict] = []
    faltam = sorted(set(c0) - set(efetiva))
    sobram = sorted(set(efetiva) - set(c0))
    if faltam or sobram:
        viol.append({"chave": "<conjunto de chaves>", "faltam": faltam, "sobram": sobram})
    for k in sorted(set(c0) & set(efetiva)):
        ef, ref = efetiva[k], c0[k]
        if k == "mmap":
            if not isinstance(ef, bool):
                viol.append({"chave": k, "efetivo": ef, "esperado": "bool"})
            continue
        if k in esperado:
            if ef != esperado[k] or isinstance(ef, bool) != isinstance(esperado[k], bool):
                viol.append({"chave": k, "efetivo": ef, "esperado": esperado[k], "c0": ref})
            continue
        if ef != ref:
            viol.append({"chave": k, "efetivo": ef, "c0": ref, "motivo": "diverge do C0 e nao e chave livre"})
    # pesos ablados: o C0 precisa ter peso > 0 (senao a ablacao nao teria efeito)
    for w in BRACOS[braco]["pesos_ablados"]:
        if not (isinstance(c0.get(w), (int, float)) and c0[w] > 0):
            viol.append({"chave": w, "c0": c0.get(w), "motivo": "peso do C0 nao e > 0: ablacao sem sentido"})
    return viol


def exigir_config(efetiva, braco, cidade, seed, q, modo, evid_dir=None) -> None:
    v = diff_config(efetiva, braco, cidade, seed, q, modo, evid_dir)
    if v:
        raise GuardaErro("CONFIG_DIVERGE", json.dumps(v, ensure_ascii=False))


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


def concluida_por_conteudo(braco, cidade, seed, q, modo="lote", evid_dir=None) -> tuple[bool, str]:
    """(concluida, motivo). Nao usa o campo status (sempre 'provisorio' no treinador)."""
    validar_celula(braco, cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    label = rotulo(braco, cidade, seed, q, sufixo_do_modo(modo))
    p = run_json_path(evid, label)
    if not p.is_file():
        return False, "run JSON ausente"
    try:
        rj = json.loads(p.read_text())
    except Exception as e:  # JSON truncado = corrida interrompida durante a gravacao
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
# sha dos insumos (HASHES_SHA256.txt), com cache por tamanho+mtime
# ----------------------------------------------------------------------------------------------------------------
_LOCK_SHA = threading.Lock()


def hashes_oficiais() -> dict:
    d = {}
    for ln in HASHES_TXT.read_text().splitlines():
        p = ln.split()
        if len(p) >= 2:
            d[p[1].lstrip("*")] = p[0].lower()
    return d


def conferir_sha_insumo(nome: str) -> str:
    """Confere o sha256 de um insumo de graph_data_v3 contra HASHES_SHA256.txt (cache por size+mtime_ns)."""
    ofic = hashes_oficiais()
    if nome not in ofic:
        raise GuardaErro("SHA_INSUMO", f"{nome} nao consta em HASHES_SHA256.txt")
    f = Path(GRAPH_DIR) / nome
    if not f.is_file():
        raise GuardaErro("SHA_INSUMO", f"insumo ausente: {f}")
    st = f.stat()
    with _LOCK_SHA:
        cache = json.loads(CACHE_SHA.read_text()) if CACHE_SHA.is_file() else {}
        c = cache.get(nome)
        if c and c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns and c["sha256"] == ofic[nome]:
            return c["sha256"]
    s = sha256_arquivo(f)
    if s != ofic[nome]:
        raise GuardaErro("SHA_INSUMO", f"{nome}: sha {s} != HASHES_SHA256.txt {ofic[nome]}")
    with _LOCK_SHA:
        cache = json.loads(CACHE_SHA.read_text()) if CACHE_SHA.is_file() else {}
        cache[nome] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": s}
        tmp = CACHE_SHA.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1))
        os.replace(tmp, CACHE_SHA)
    return s


# ----------------------------------------------------------------------------------------------------------------
# checagens ANTES e DEPOIS
# ----------------------------------------------------------------------------------------------------------------
def checar_pre(braco, cidade, seed, q, argv, modo="lote", evid_dir=None, verificar_sha=True) -> dict:
    """Aborta (GuardaErro) se a corrida nao pode comecar. Retorna relatorio."""
    validar_celula(braco, cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    rel = {"fase": "pre", "braco": braco, "cidade": cidade, "seed": seed, "q": q, "modo": modo}
    rel["sha_treinador"] = conferir_sha_treinador()
    ef = config_do_argv(argv)
    exigir_config(ef, braco, cidade, seed, q, modo, evid)
    rel["config_diff_ok"] = True
    # insumos: existencia + tamanho igual ao do C0; sha contra HASHES (so no lote)
    c0 = carregar_c0(cidade, seed, q)
    ds = c0.get("dataset", {})
    for nome, chave in ((ef["graph_file"], "graph_bytes"), (ef["rf_data_file"], "rf_data_bytes")):
        f = Path(ef["graph_dir"]) / nome
        if not f.is_file():
            raise GuardaErro("INSUMO", f"insumo ausente: {f}")
        if ds.get(chave) is not None and f.stat().st_size != ds[chave]:
            raise GuardaErro("INSUMO", f"{f}: {f.stat().st_size} B != {ds[chave]} B do C0")
        if verificar_sha and modo != "smoke":
            rel.setdefault("sha_insumos", {})[nome] = conferir_sha_insumo(nome)
    # B1 antes
    tf = ef["transfer_from"]
    esperado_tf = transfer_esperado(braco, cidade, seed, q, evid, suf)
    if q == 1:
        if tf:
            raise GuardaErro("B1_PRE", f"Q1 nao pode ter transfer_from (veio {tf!r})")
    else:
        if tf != esperado_tf:
            raise GuardaErro("B1_PRE", f"transfer_from {tf!r} != esperado {esperado_tf!r} (mesmo braco/cidade/seed, Q{q-1})")
        p = Path(tf)
        if not p.is_file():
            raise GuardaErro("B1_PRE", f"transfer_from nao e arquivo existente: {tf}")
        if p.parent.parent.name != rotulo(braco, cidade, seed, q - 1, suf):
            raise GuardaErro("B1_PRE", f"rotulo do checkpoint {p.parent.parent.name!r} nao e o do mesmo braco/Q anterior")
        ok, mot = concluida_por_conteudo(braco, cidade, seed, q - 1, modo, evid)
        if not ok:
            raise GuardaErro("S2_PRE", f"Q{q-1} do mesmo braco NAO esta concluida por conteudo: {mot}")
        rel["transfer_from"] = tf
        rel["transfer_sha256"] = sha256_arquivo(p)
    return rel


def _componentes(rj: dict) -> list[dict]:
    return [(e.get("loss_componentes") or {}) for e in rj.get("epocas", [])]


def checar_pos(braco, cidade, seed, q, modo="lote", evid_dir=None, log_path=None, info_patch=None) -> dict:
    """Aborta (GuardaErro) se a corrida terminada nao atende ao protocolo. info_patch = estado da instancia da loss
    lido pelo wrapper ao FIM do treino ({'fspl_weight':..., 'n_instancias':...})."""
    validar_celula(braco, cidade, seed, q)
    evid = Path(evid_dir) if evid_dir else evid_do_modo(modo)
    suf = sufixo_do_modo(modo)
    label = rotulo(braco, cidade, seed, q, suf)
    rel = {"fase": "pos", "label": label, "modo": modo}
    p = run_json_path(evid, label)
    if not p.is_file():
        raise GuardaErro("POS_RUN_JSON", f"run JSON nao foi gravado: {p}")
    rj = json.loads(p.read_text())
    if rj.get("script_sha256") != TREINADOR_SHA256:
        raise GuardaErro("SHA_TREINADOR", f"run JSON registra script_sha256 {rj.get('script_sha256')}")
    # config gravada no run JSON x C0 (segundo diff, agora sobre o que o treinador efetivamente registrou)
    exigir_config(rj.get("config", {}), braco, cidade, seed, q, modo, evid)
    rel["config_diff_ok"] = True
    # S2
    ok, mot = concluida_por_conteudo(braco, cidade, seed, q, modo, evid)
    if not ok:
        raise GuardaErro("S2_POS", f"corrida nao concluida por conteudo: {mot}")
    rel["concluida_por_conteudo"] = True
    # B1 depois (campo de TOPO, nao config)
    esperado_tf = transfer_esperado(braco, cidade, seed, q, evid, suf)
    if q == 1:
        if rj.get("transfer_from"):
            raise GuardaErro("B1_POS", f"Q1 registra transfer_from de topo {rj.get('transfer_from')!r}")
    else:
        if rj.get("transfer_from") != esperado_tf:
            raise GuardaErro("B1_POS", f"run_json['transfer_from'] (topo) = {rj.get('transfer_from')!r} != esperado {esperado_tf!r}")
    if log_path is not None:
        txt = Path(log_path).read_text(errors="replace") if Path(log_path).is_file() else ""
        linhas = [ln for ln in txt.splitlines() if "[TRANSFER]" in ln]
        if q == 1 and linhas:
            raise GuardaErro("B1_POS", "Q1 com linha [TRANSFER] no log")
        if q > 1 and not any(esperado_tf in ln for ln in linhas):
            raise GuardaErro("B1_POS", f"sem linha [TRANSFER] com {esperado_tf!r} no log {log_path}")
        rel["linha_transfer_no_log"] = (linhas[0] if linhas else None)
    # insumo: sha do rf_data registrado pelo treinador (--hash) == HASHES
    sha_rf = (rj.get("dataset") or {}).get("rf_data_sha256")
    if modo != "smoke":
        if not sha_rf:
            raise GuardaErro("SHA_INSUMO", "run JSON sem dataset.rf_data_sha256")
        ofic = hashes_oficiais().get(rj["config"]["rf_data_file"])
        if sha_rf != ofic:
            raise GuardaErro("SHA_INSUMO", f"rf_data_sha256 {sha_rf} != HASHES {ofic}")
        rel["rf_data_sha256_ok"] = True
    # ablacao efetiva: pesos registrados e termos zerados em TODAS as epocas
    lo = rj.get("loss", {})
    ab = BRACOS[braco]["pesos_ablados"]
    comp_zero = {"dist_grad_weight": ["dist_gradient_penalty"], "variance_weight": ["variance_penalty"],
                 "ndvi_weight": ["shadowing_ndvi_penalty"]}
    for w, k in (("dist_grad_weight", "distance_gradient_weight"), ("variance_weight", "variance_weight"),
                 ("ndvi_weight", "shadowing_ndvi_weight")):
        if w in ab and lo.get(k) != 0.0:
            raise GuardaErro("ABLACAO", f"loss.{k} = {lo.get(k)} (esperado 0.0)")
        if w not in ab and lo.get(k) != carregar_c0(cidade, seed, q)["config"][w]:
            raise GuardaErro("ABLACAO", f"loss.{k} = {lo.get(k)} diverge do C0 em termo NAO ablado")
    cs = _componentes(rj)
    termos_zero = [t for w in ab for t in comp_zero.get(w, [])]
    if BRACOS[braco]["fspl_zero"]:
        termos_zero.append("constraint_loss")
    for i, c in enumerate(cs):
        for t in termos_zero:
            if t not in c:
                raise GuardaErro("ABLACAO", f"epocas[{i}].loss_componentes sem a chave {t}")
            if c[t] != 0.0:
                raise GuardaErro("ABLACAO", f"epocas[{i}].loss_componentes.{t} = {c[t]} (esperado 0 no braco {braco})")
    rel["termos_zerados_verificados"] = termos_zero
    if BRACOS[braco]["fspl_zero"]:
        # constraint_loss = vegetation_weight*veg + fspl_weight*fspl; sem dossel so ha o FSPL -> 0 prova fspl_weight = 0
        pass
    if info_patch is not None:
        esp_fspl = 0.0 if BRACOS[braco]["fspl_zero"] else FSPL_PADRAO_C0
        if info_patch.get("n_instancias") != 1:
            raise GuardaErro("ABLACAO", f"n de instancias da loss = {info_patch.get('n_instancias')} (esperado 1)")
        if info_patch.get("fspl_weight_fim") != esp_fspl or info_patch.get("fspl_weight_construcao") != esp_fspl:
            raise GuardaErro("ABLACAO", f"fspl_weight da instancia {info_patch} != {esp_fspl}")
        rel["info_patch"] = info_patch
    return rel


# ----------------------------------------------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acao", choices=["concluida", "pre", "pos"])
    ap.add_argument("--braco", required=True, choices=list(BRACOS))
    ap.add_argument("--cidade", required=True, choices=list(CIDADES))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--q", required=True, type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mmap", action="store_true")
    a = ap.parse_args()
    modo = "smoke" if a.smoke else "lote"
    try:
        if a.acao == "concluida":
            ok, mot = concluida_por_conteudo(a.braco, a.cidade, a.seed, a.q, modo)
            print(json.dumps({"concluida": ok, "motivo": mot}, ensure_ascii=False))
            sys.exit(0 if ok else 1)
        if a.acao == "pre":
            argv = montar_argv(a.braco, a.cidade, a.seed, a.q, modo, a.mmap)
            print(json.dumps(checar_pre(a.braco, a.cidade, a.seed, a.q, argv, modo), indent=1, ensure_ascii=False))
        else:
            lp = evid_do_modo(modo) / rotulo(a.braco, a.cidade, a.seed, a.q, sufixo_do_modo(modo)) / "stdout_wrapper.log"
            print(json.dumps(checar_pos(a.braco, a.cidade, a.seed, a.q, modo, log_path=lp), indent=1, ensure_ascii=False))
    except GuardaErro as e:
        print(f"GUARDA ABORTA {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
