# -*- coding: utf-8 -*-
"""Wrapper do braco BA1 (RadioUNet-style in-house adaptation, R2-3): uma corrida por processo.

Padrao de fase_C/wrapper_ablacao.py, sem monkeypatch (o treinador do BA1 e arquivo novo, conferido por sha):
  python wrapper_ba1.py --cidade bauru --seed 42 --q 1 [--mmap] [--smoke] [--so-pos] [--dry-run]

Fluxo: sha do treinador -> argv do C0 + config_ba1.json -> guarda PRE (diff de config, ABERTAS, B1 antes, S2 da Q
anterior, sha do insumo) -> corrida (stdout duplicado em <run>/stdout_wrapper.log) -> guarda POS (config gravada, S2,
B1 depois, sha do rf_data, particoes == C0, modelo/gradiente/sementes do BA1) -> sidecar <run>/sidecar_<run>.json.
Codigos de saida: 0 ok | 2 guarda PRE | 3 ja concluida (nao sobrescreve) | 4 erro no treino | 5 guarda POS | 6 sem GPU
(GPU exigida no smoke e no lote desde a A3 do gate de 18/10; a guarda POS tambem recusa amp 'on' sem amp_efetivo).
Corrida interrompida anterior NAO e apagada: a pasta vai para treinos/_incompletas/<run>_<carimbo>.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import guarda_ba1 as G  # noqa: E402

WRAPPER = Path(__file__).resolve()


class Tee:
    def __init__(self, orig, arq):
        self._o, self._f = orig, arq

    def write(self, s):
        self._o.write(s)
        self._f.write(s)
        return len(s)

    def flush(self):
        self._o.flush()
        self._f.flush()

    def __getattr__(self, k):
        return getattr(self._o, k)


def versoes() -> dict:
    d = {"python": sys.version.split()[0]}
    for rotulo, pacote in (("torch", "torch"), ("torch_geometric", "torch_geometric"), ("pyg_lib", "pyg-lib"),
                           ("torch_sparse", "torch-sparse"), ("torch_scatter", "torch-scatter"),
                           ("torch_cluster", "torch-cluster"), ("numpy", "numpy")):
        try:
            d[rotulo] = md.version(pacote)
        except Exception:
            d[rotulo] = None
    try:
        import torch
        d["cuda_torch"] = torch.version.cuda
        d["cudnn"] = torch.backends.cudnn.version()
        d["cuda_disponivel"] = torch.cuda.is_available()
        d["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        d["torch_erro"] = repr(e)
    return d


def plataforma() -> dict:
    u = platform.uname()
    return {"platform": platform.platform(), "sistema": u.system, "kernel": u.release, "maquina": u.machine,
            "host": socket.gethostname(), "executavel": sys.executable, "pid": os.getpid()}


def gravar_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, p)


def mover_incompleta(run_dir: Path, evid: Path, label: str):
    if not run_dir.exists():
        return None
    dest_dir = evid / "_incompletas"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(run_dir), str(dest))
    return str(dest)


def executar(cidade, seed, q, modo="lote", mmap=False, so_pos=False, dry_run=False, exigir_gpu=None) -> int:
    t0 = time.time()
    evid = G.evid_do_modo(modo)
    suf = G.sufixo_do_modo(modo)
    label = G.rotulo(cidade, seed, q, suf)
    run_dir = evid / label
    log_path = run_dir / "stdout_wrapper.log"
    sidecar_path = run_dir / f"sidecar_{label}.json"
    exigir_gpu = True if exigir_gpu is None else exigir_gpu   # [A3 gate 18/10] GPU exigida no smoke E no lote
    sidecar = {"label": label, "referencia_c0": G.rotulo_c0(cidade, seed, q), "braco": G.BRACO, "cidade": cidade,
               "seed": seed, "q": q, "modo": modo, "inicio_utc": datetime.now(timezone.utc).isoformat(),
               "wrapper": str(WRAPPER), "wrapper_sha256": G.sha256_arquivo(WRAPPER),
               "guarda_sha256": G.sha256_arquivo(Path(G.__file__)),
               "treinador": str(G.TREINADOR), "treinador_sha256_esperado": G.TREINADOR_SHA256,
               "config_ba1": str(G.CONFIG_BA1), "config_ba1_sha256": G.config_ba1_sha(),
               "versoes": versoes(), "plataforma": plataforma()}
    try:
        sidecar["treinador_sha256_conferido"] = G.conferir_sha_treinador()
        argv = G.montar_argv(cidade, seed, q, modo, mmap, evid)
        sidecar["argv"] = argv
        sidecar["valores_ba1"] = G.valores_ba1(modo)
        if exigir_gpu and not sidecar["versoes"].get("cuda_disponivel"):
            print("[WRAPPER] GPU exigida e indisponivel", file=sys.stderr)
            return 6
        if so_pos:
            if sidecar_path.is_file():
                antigo = json.loads(sidecar_path.read_text())
                antigo.setdefault("reexecucoes_pos", []).append({"em": sidecar["inicio_utc"],
                                                                  "wrapper_sha256": sidecar["wrapper_sha256"]})
                sidecar = antigo
        else:
            ok, mot = G.concluida_por_conteudo(cidade, seed, q, modo, evid)
            if ok and not dry_run:
                print(f"[WRAPPER] {label} ja concluida por conteudo; nada a fazer (nao sobrescreve)")
                return 3
            sidecar["guarda_pre"] = G.checar_pre(cidade, seed, q, argv, modo, evid)
        if dry_run:
            print(json.dumps({"dry_run": True, "label": label, "argv": argv,
                              "guarda_pre": sidecar.get("guarda_pre")}, indent=1, ensure_ascii=False, default=str))
            return 0
        if not so_pos:
            moved = mover_incompleta(run_dir, evid, label) if run_dir.exists() else None
            sidecar["pasta_incompleta_movida_para"] = moved
            run_dir.mkdir(parents=True, exist_ok=True)
            sidecar["status"] = "iniciado"
            gravar_json(sidecar_path, sidecar)
            f_log = open(log_path, "a", buffering=1)
            f_log.write(f"# wrapper {datetime.now(timezone.utc).isoformat()} {label}\n")
            so, se = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(so, f_log), Tee(se, f_log)
            try:
                rc = G.carregar_treinador().main(argv)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
            except BaseException:
                traceback.print_exc()
                rc = 1
            finally:
                sys.stdout, sys.stderr = so, se
                f_log.close()
            sidecar["rc_treinador"] = rc
            if rc not in (0, None):
                sidecar["status"] = "erro_treino"
                sidecar["fim_utc"] = datetime.now(timezone.utc).isoformat()
                gravar_json(sidecar_path, sidecar)
                return 4
        try:
            sidecar["guarda_pos"] = G.checar_pos(cidade, seed, q, modo, evid,
                                                 log_path=(log_path if log_path.is_file() else None))
        except G.GuardaErro as e:
            sidecar["status"] = "guarda_pos_falhou"
            sidecar["guarda_pos_erro"] = str(e)
            sidecar["fim_utc"] = datetime.now(timezone.utc).isoformat()
            gravar_json(sidecar_path, sidecar)
            print(f"[WRAPPER] GUARDA POS ABORTA {e}", file=sys.stderr)
            return 5
        sidecar["status"] = "concluida_guarda_ok"
        sidecar["fim_utc"] = datetime.now(timezone.utc).isoformat()
        sidecar["duracao_wrapper_s"] = round(time.time() - t0, 1)
        gravar_json(sidecar_path, sidecar)
        print(f"[WRAPPER] {label}: concluida, guarda POS ok; sidecar {sidecar_path}")
        return 0
    except G.GuardaErro as e:
        sidecar["status"] = "guarda_pre_falhou"
        sidecar["guarda_pre_erro"] = str(e)
        print(f"[WRAPPER] GUARDA PRE ABORTA {e}", file=sys.stderr)
        try:
            if run_dir.exists():
                sidecar["fim_utc"] = datetime.now(timezone.utc).isoformat()
                gravar_json(run_dir / f"sidecar_{label}.pre_falhou.json", sidecar)
        except Exception:
            pass
        return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade", required=True, choices=list(G.CIDADES))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--q", required=True, type=int)
    ap.add_argument("--smoke", action="store_true", help="2 epocas, 200k nos, GPU exigida; saida em fase_BA1/smoke_treinos")
    ap.add_argument("--mmap", action="store_true", help="torch.load(mmap=True) (diferenca permitida pelo protocolo)")
    ap.add_argument("--so-pos", action="store_true", help="nao treina: so roda a guarda POS e regrava o sidecar")
    ap.add_argument("--dry-run", action="store_true", help="monta argv e roda a guarda PRE; sem treino")
    a = ap.parse_args(argv)
    return executar(a.cidade, a.seed, a.q, "smoke" if a.smoke else "lote", a.mmap, a.so_pos, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
