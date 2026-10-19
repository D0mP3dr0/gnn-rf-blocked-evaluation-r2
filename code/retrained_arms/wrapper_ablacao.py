# -*- coding: utf-8 -*-
"""Wrapper da ablacao (fase C do R4 IEEE Access): chama o treinador CONGELADO sem edita-lo.

Treinador: FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/train_gnn_c0_spatial.py
(sha256 6f955629... conferido antes de tudo). Mecanismo: import (importlib) + monkeypatch de
physics_loss.CurriculumRFLoss.__init__ (zera fspl_weight da instancia DEPOIS da construcao no braco c1noreg); os pesos
dist/variance/ndvi vao por CLI do proprio treinador. Config = a do run JSON C0 correspondente (guarda_config.montar_argv).

Uma corrida por processo:
  python wrapper_ablacao.py --braco c1noreg|c2nomono --cidade bauru --seed 42 --q 1 [--mmap] [--smoke] [--so-pos] [--dry-run]

Fluxo: sha do treinador -> argv do C0 -> guarda PRE (diff de config, B1 antes, S2 da Q anterior, sha dos insumos) ->
corrida (stdout duplicado em <run>/stdout_wrapper.log) -> guarda POS (diff de config gravada, B1 depois, S2, ablacao
efetiva, sha do rf_data) -> sidecar <run>/sidecar_<run_label>.json (versoes torch/torch_geometric/pyg-lib/torch-sparse,
plataforma, sha do wrapper/guarda/treinador, patch, resultados das guardas).
Codigos de saida: 0 ok | 2 guarda PRE | 3 ja concluida (nao sobrescreve) | 4 erro no treino | 5 guarda POS | 6 sem GPU.
Corrida interrompida anterior NAO e apagada: a pasta e movida para treinos/_incompletas/<run>_<carimbo>.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import guarda_config as G  # noqa: E402

WRAPPER = Path(__file__).resolve()


class Tee:
    """Duplica escrita para o terminal/log da fila e para um arquivo da corrida (usado pelo POS para ler [TRANSFER])."""

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


def instalar_patch(braco: str, info: dict):
    """Monkeypatch de CurriculumRFLoss.__init__: apos a construcao zera fspl_weight (so c1noreg) e registra a instancia.
    physics_loss e importado de base_dir/02_models ANTES do treinador, que reutiliza o modulo via sys.modules."""
    for sub in ("02_models", "03_training"):
        sp = str(Path(G.BASE_DIR) / sub)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    import physics_loss
    orig = physics_loss.CurriculumRFLoss.__init__
    info["instancias"] = []
    zerar = G.BRACOS[braco]["fspl_zero"]

    def _init(self, *a, **k):
        orig(self, *a, **k)
        if zerar:
            self.fspl_weight = 0.0   # peso FSPL da PhysicsRFLoss = 0 (atributo da instancia, apos construcao)
        info["instancias"].append(self)
        info["fspl_weight_construcao"] = float(self.fspl_weight)
        print("[PATCH] braco=%s fspl_weight=%s dist_grad=%s variance=%s ndvi=%s vegetation=%s" % (
            braco, self.fspl_weight, self.distance_gradient_weight, self.variance_weight,
            self.shadowing_ndvi_weight, self.vegetation_weight), flush=True)

    physics_loss.CurriculumRFLoss.__init__ = _init
    return physics_loss, orig


def mover_incompleta(run_dir: Path, evid: Path, label: str):
    if not run_dir.exists():
        return None
    dest_dir = evid / "_incompletas"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(run_dir), str(dest))
    return str(dest)


def executar(braco, cidade, seed, q, modo="lote", mmap=False, so_pos=False, dry_run=False, treinador_main=None,
             exigir_gpu=None, evid_dir=None) -> int:
    """treinador_main: so para teste com treinador SIMULADO (teste_mutacao_guarda.py); em producao e None."""
    t0 = time.time()
    evid = Path(evid_dir) if evid_dir else G.evid_do_modo(modo)
    suf = G.sufixo_do_modo(modo)
    label = G.rotulo(braco, cidade, seed, q, suf)
    run_dir = evid / label
    log_path = run_dir / "stdout_wrapper.log"
    sidecar_path = run_dir / f"sidecar_{label}.json"
    exigir_gpu = (modo == "smoke") if exigir_gpu is None else exigir_gpu
    sidecar = {"label": label, "referencia_c0": G.rotulo_c0(cidade, seed, q), "braco": braco, "cidade": cidade,
               "seed": seed, "q": q, "modo": modo, "inicio_utc": datetime.now(timezone.utc).isoformat(),
               "wrapper": str(WRAPPER), "wrapper_sha256": G.sha256_arquivo(WRAPPER),
               "guarda_sha256": G.sha256_arquivo(Path(G.__file__)),
               "treinador": str(G.TREINADOR), "treinador_sha256_esperado": G.TREINADOR_SHA256,
               "versoes": versoes(), "plataforma": plataforma()}
    try:
        # ---------- ANTES ----------
        sidecar["treinador_sha256_conferido"] = G.conferir_sha_treinador()
        argv = G.montar_argv(braco, cidade, seed, q, modo, mmap, evid)
        sidecar["argv"] = argv
        if exigir_gpu and not sidecar["versoes"].get("cuda_disponivel"):
            print("[WRAPPER] GPU exigida e indisponivel", file=sys.stderr)
            return 6
        if so_pos:
            if sidecar_path.is_file():   # preserva o sidecar da corrida; a reexecucao do POS vai em campo proprio
                antigo = json.loads(sidecar_path.read_text())
                antigo.setdefault("reexecucoes_pos", []).append({"em": sidecar["inicio_utc"],
                                                                  "wrapper_sha256": sidecar["wrapper_sha256"]})
                sidecar = antigo
        else:
            ok, mot = G.concluida_por_conteudo(braco, cidade, seed, q, modo, evid)
            if ok and not dry_run:
                print(f"[WRAPPER] {label} ja concluida por conteudo; nada a fazer (nao sobrescreve)")
                return 3
            sidecar["guarda_pre"] = G.checar_pre(braco, cidade, seed, q, argv, modo, evid,
                                                 verificar_sha=(treinador_main is None))
        if dry_run:
            info = {}
            fisica, orig = instalar_patch(braco, info)
            try:
                G.carregar_treinador()
                # constroi a loss como o treinador (frequency_mhz=900, pesos do argv) so para ver o patch agir
                ef = G.config_do_argv(argv)
                fisica.CurriculumRFLoss(frequency_mhz=ef["freq_mhz"], distance_gradient_weight=ef["dist_grad_weight"],
                                        distance_gradient_n_pairs=512, variance_weight=ef["variance_weight"],
                                        shadowing_ndvi_weight=ef["ndvi_weight"], shadowing_ndvi_min_corr=0.15)
            finally:
                fisica.CurriculumRFLoss.__init__ = orig
            print(json.dumps({"dry_run": True, "label": label, "argv": argv,
                              "fspl_weight_instancia": info.get("fspl_weight_construcao")}, indent=1))
            return 0
        info_patch: dict = {}
        if not so_pos:
            moved = mover_incompleta(run_dir, evid, label) if run_dir.exists() else None
            sidecar["pasta_incompleta_movida_para"] = moved
            run_dir.mkdir(parents=True, exist_ok=True)
            sidecar["status"] = "iniciado"
            gravar_json(sidecar_path, sidecar)
            # ---------- CORRIDA ----------
            f_log = open(log_path, "a", buffering=1)
            f_log.write(f"# wrapper {datetime.now(timezone.utc).isoformat()} {label}\n")
            so, se = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(so, f_log), Tee(se, f_log)
            fisica = orig = None
            try:
                fisica, orig = instalar_patch(braco, info_patch)
                if treinador_main is None:
                    main = G.carregar_treinador().main
                else:
                    main = treinador_main
                rc = main(argv)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
            except BaseException:
                traceback.print_exc()
                rc = 1
            finally:
                sys.stdout, sys.stderr = so, se
                f_log.close()
                if fisica is not None:
                    fisica.CurriculumRFLoss.__init__ = orig
            sidecar["rc_treinador"] = rc
            insts = info_patch.pop("instancias", [])
            info_patch["n_instancias"] = len(insts)
            info_patch["fspl_weight_fim"] = float(insts[0].fspl_weight) if insts else None
            sidecar["patch"] = info_patch
            if rc not in (0, None):
                sidecar["status"] = "erro_treino"
                sidecar["fim_utc"] = datetime.now(timezone.utc).isoformat()
                gravar_json(sidecar_path, sidecar)
                return 4
        else:
            info_patch = None
        # ---------- DEPOIS ----------
        try:
            sidecar["guarda_pos"] = G.checar_pos(braco, cidade, seed, q, modo, evid,
                                                 log_path=(log_path if log_path.is_file() else None), info_patch=info_patch)
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
    ap.add_argument("--braco", required=True, choices=list(G.BRACOS))
    ap.add_argument("--cidade", required=True, choices=list(G.CIDADES))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--q", required=True, type=int)
    ap.add_argument("--smoke", action="store_true", help="1 epoca, 60k nos, GPU exigida; saida em fase_C/smoke_treinos")
    ap.add_argument("--mmap", action="store_true", help="torch.load(mmap=True) (diferenca permitida pelo protocolo)")
    ap.add_argument("--so-pos", action="store_true", help="nao treina: so roda a guarda POS e regrava o sidecar")
    ap.add_argument("--dry-run", action="store_true", help="monta argv, roda a guarda PRE e mostra o patch; sem treino")
    a = ap.parse_args(argv)
    return executar(a.braco, a.cidade, a.seed, a.q, "smoke" if a.smoke else "lote", a.mmap, a.so_pos, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
