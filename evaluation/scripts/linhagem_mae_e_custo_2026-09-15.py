"""Cross-checks which checkpoint lineage produced each reported MAE, plus wall-clock/VRAM cost per run."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_EVID = Path(__file__).resolve().parents[1]
SAIDA = BASE_EVID / "dados" / "consolidacao"
PREREG = BASE_EVID / "dados" / "planos" / "preregistro_admissibilidade_e_params_2026-09-15.json"
FASE0_Q2 = BASE_EVID / "dados" / "fase0" / "fase0_metrics_bauru_s42_Q2.json"
MAIN_TEX = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\REVISAO_R3\main.tex")
BASE_COLAB = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\baseline_colab_\logs")

CONJUNTOS = {
    "version_20": BASE_COLAB / "version_20",
    "version_20.1": BASE_COLAB / "version_20.1" / "logs" / "version_20",
    "version_20.2": BASE_COLAB / "version_20.2",
    "version_20.3": BASE_COLAB / "version_20.3" / "version_20",
}
MODELOS = ["radiounet", "rem_gnn", "radiogat", "rme_gan"]
PUBLICADO_MAE = {"rem_gnn": 0.190, "radiounet": 0.251, "rme_gan": 0.218,
                 "radiogat": 0.961, "gnn": 0.349}
TOL_MAE = 5e-4


def morre(msg: str) -> None:
    raise SystemExit(f"[ABORTA] {msg}")


def sha256(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def carimbo(p: Path) -> dict:
    st = p.stat()
    return {"caminho": str(p), "bytes": st.st_size, "sha256": sha256(p),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def exigir(d: dict, caminho: list[str], onde: str):
    cur = d
    for k in caminho:
        if not isinstance(cur, dict) or k not in cur:
            morre(f"chave ausente {'.'.join(caminho)} em {onde}")
        cur = cur[k]
    return cur


def linhagem_dos_mae() -> dict:
    import torch
    d = json.loads(FASE0_Q2.read_text(encoding="utf-8"))
    out = {}
    for nome, pub in PUBLICADO_MAE.items():
        m = exigir(d, ["modelos", nome], str(FASE0_Q2))
        mae = exigir(m, ["todos_os_nos", "mae_rssi_db"], str(FASE0_Q2))
        if abs(round(float(mae), 3) - pub) > TOL_MAE:
            morre(f"{nome}: MAE do artefato {mae} nao arredonda no publicado {pub}")
        ck = Path(exigir(m, ["checkpoint"], str(FASE0_Q2)))
        if not ck.exists():
            morre(f"{nome}: checkpoint do artefato ausente em disco: {ck}")
        h = sha256(ck)
        reg = exigir(m, ["checkpoint_sha256"], str(FASE0_Q2))
        if h != reg:
            morre(f"{nome}: sha do checkpoint em disco {h[:12]} != registrado {reg[:12]}; "
                  "o arquivo mudou depois da avaliacao")
        o = torch.load(ck, map_location="cpu", weights_only=False)
        comp = {k: sum(t.numel() for t in v.values())
                for k, v in o.items()
                if k.endswith("_state_dict") and "optimizer" not in k
                and "scheduler" not in k and "scaler" not in k
                and isinstance(v, dict) and all(hasattr(x, "numel") for x in v.values())}
        tt = [k2 for k, v in o.items()
              if k.endswith("_state_dict") and isinstance(v, dict)
              and all(hasattr(x, "numel") for x in v.values())
              for k2 in v if str(k2).startswith("tt_msg")]
        ts = str(o.get("timestamp", ""))[:19]
        del o


        casam = sorted(((c, str(raiz)) for c, raiz in CONJUNTOS.items()
                        if str(raiz).lower() in str(ck).lower()),
                       key=lambda x: -len(x[1]))
        conj = casam[0][0] if casam else "GNN_RF_V2/logs/version_20 (GNN-RF, campanha v20)"
        if len(casam) > 1:
            print(f"    [nota] {nome}: prefixo ambiguo, escolhido o mais longo "
                  f"({conj}) entre {[c for c, _ in casam]}")
        out[nome] = {
            "mae_rssi_db_publicado": pub,
            "mae_rssi_db_no_artefato": float(mae),
            "artefato_do_mae": str(FASE0_Q2),
            "campo_do_mae": f"modelos.{nome}.todos_os_nos.mae_rssi_db",
            "checkpoint": str(ck),
            "checkpoint_sha256": h,
            "checkpoint_data_treino": ts,
            "conjunto_de_treinos": conj,
            "tem_tt_msg": bool(tt),
            "linhagem": "COM tt_msg" if tt else "SEM tt_msg",
            "params_no_state_dict": comp,
            "n_params_declarado_pelo_fase0": m.get("n_params_treinaveis"),
            "nota_divergencia_fase0": (
                "fase0_eval.py:271 grava a contagem da CLASSE INSTANCIADA (com tt_msg), "
                "nao a do checkpoint carregado com strict=False (fase0_eval.py:168-182)"),
        }
    return out


def matriz_custo() -> dict:
    linhas = []
    for conj, raiz in CONJUNTOS.items():
        for m in MODELOS:
            for q in ("Q1", "Q4"):
                f = raiz / f"{m}_bauru_s42_{q}" / "computational_cost.json"
                if not f.exists():
                    linhas.append({"conjunto": conj, "modelo": m, "quadrante": q,
                                   "ausente": str(f)})
                    continue
                d = json.loads(f.read_text(encoding="utf-8"))
                linhas.append({
                    "conjunto": conj, "modelo": m, "quadrante": q,
                    "n_params": exigir(d, ["n_params"], str(f)),
                    "total_training_s": exigir(d, ["total_training_s"], str(f)),
                    "model_size_mb": exigir(d, ["model_size_mb"], str(f)),
                    "best_mae_rssi": exigir(d, ["best_mae_rssi"], str(f)),
                    "arquivo": str(f), "sha256": sha256(f),
                })
    return {"linhas": linhas,
            "nota": ("cada computational_cost.json carrega o n_params da PROPRIA linhagem: "
                     "o par publicado (parametro da linhagem sem tt_msg com tempo da "
                     "linhagem com tt_msg) nao existe em nenhum artefato isolado")}


TOKENS = ["265602", "72450", "301826", "72194", "397186", "73554", "375938",
          "73206", "1812515", "72k", "302k", "397k", "Params", "Size"]


def inventario_manuscrito() -> dict:
    if not MAIN_TEX.exists():
        morre(f"main.tex ausente: {MAIN_TEX}")
    linhas = MAIN_TEX.read_text(encoding="utf-8").splitlines()
    achados = []
    for i, ln in enumerate(linhas, 1):
        hits = [t for t in TOKENS if t in ln]
        if hits:
            achados.append({"linha": i, "tokens": hits, "trecho": ln.strip()[:600]})
    return {"arquivo": carimbo(MAIN_TEX), "ocorrencias": achados,
            "nota": "leitura apenas; main.tex nao foi escrito por este script"}


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    mae = linhagem_dos_mae()
    custo = matriz_custo()
    inv = inventario_manuscrito()

    linhagens_mae = sorted({v["linhagem"] for k, v in mae.items() if k != "gnn"})
    tempos_publicados = {"radiounet": (444, 740), "rem_gnn": (383, 1080),
                         "radiogat": (2347, 1511), "rme_gan": (879, 762)}
    conj_do_tempo = {}
    for m, (t1, t4) in tempos_publicados.items():
        casa = [l for l in custo["linhas"] if l.get("total_training_s") is not None
                and l["modelo"] == m
                and abs(l["total_training_s"] - (t1 if l["quadrante"] == "Q1" else t4)) <= 0.5]
        conj_do_tempo[m] = sorted({l["conjunto"] for l in casa})

    opcao_A = {}
    opcao_B = {}
    for m in MODELOS:
        l203 = {l["quadrante"]: l for l in custo["linhas"]
                if l["conjunto"] == "version_20.3" and l["modelo"] == m
                and "ausente" not in l}
        l201 = {l["quadrante"]: l for l in custo["linhas"]
                if l["conjunto"] == "version_20.1" and l["modelo"] == m
                and "ausente" not in l}
        if l203:
            opcao_A[m] = {"params": l203["Q1"]["n_params"],
                          "size_mb": l203["Q1"]["model_size_mb"],
                          "Q1_s": l203["Q1"]["total_training_s"],
                          "Q4_s": l203["Q4"]["total_training_s"],
                          "linhagem": "COM tt_msg (version_20.3)"}
        if l201:
            opcao_B[m] = {"params": l201["Q1"]["n_params"],
                          "size_mb": l201["Q1"]["model_size_mb"],
                          "Q1_s": l201["Q1"]["total_training_s"],
                          "Q4_s": l201["Q4"]["total_training_s"],
                          "linhagem": "SEM tt_msg (version_20.1)"}

    res = {
        "artefato_tipo": "linhagem_mae_e_opcoes_de_correcao_custo",
        "status": "provisorio",
        "citavel": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "linhagem_dos_mae_publicados": mae,
        "linhagem_unica_dos_baselines": linhagens_mae,
        "conjunto_de_onde_vem_cada_tempo_publicado": conj_do_tempo,
        "matriz_custo_por_conjunto": custo,
        "opcao_A_manter_tempos_trocar_parametros": opcao_A,
        "opcao_B_manter_parametros_trocar_tempos": opcao_B,
        "criterio_de_decisao": (
            "a linha da tabela tem de descrever o modelo que produziu o numero de ERRO "
            "publicado. Os MAE de Bauru Q2 (0,251 / 0,218 / 0,190 / 0,961) saem dos "
            "checkpoints version_20.1, linhagem SEM tt_msg, cujos parametros sao os "
            "publicados. Logo a opcao B (manter parametros, trocar tempos pelos da mesma "
            "arvore) e a unica que mantem tabela de custo, tabela de arquitetura e secao "
            "de resultados descrevendo o MESMO modelo"),
        "inventario_no_manuscrito": inv,
        "preregistro": carimbo(PREREG),
        "script": carimbo(Path(__file__).resolve()),
        "ambiente": {"python": sys.version.split()[0], "executavel": sys.executable,
                     "cuda_usada": False},
    }
    out = SAIDA / "linhagem_mae_e_custo_2026-09-15.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado {out}")
    for k, v in mae.items():
        print(f"  {k:10s} mae={v['mae_rssi_db_no_artefato']:.7f} {v['linhagem']:11s} "
              f"{v['conjunto_de_treinos']:24s} {v['checkpoint_data_treino']} "
              f"sha={v['checkpoint_sha256'][:12]} params={v['params_no_state_dict']}")
    print("  tempos publicados vem de:", conj_do_tempo)
    print("  opcao A:", json.dumps(opcao_A, ensure_ascii=False))
    print("  opcao B:", json.dumps(opcao_B, ensure_ascii=False))


if __name__ == "__main__":
    main()
