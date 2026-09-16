"""Computes SHA-256, mtime, and size of the 16 enriched reference-field files."""
import hashlib, os, json, time, sys

BASE = "F:/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3"
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADS = ["Q1", "Q2", "Q3", "Q4"]

out = {}
t_ini = time.time()
for cidade in CIDADES:
    for q in QUADS:
        nome = f"transfer_dataset_{cidade}_v19_{q}_enriched_cftudo.pt"
        caminho = os.path.join(BASE, nome)
        st = os.stat(caminho)
        h = hashlib.sha256()
        t0 = time.time()
        with open(caminho, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
        dt = time.time() - t0
        out[nome] = {
            "sha256": h.hexdigest(),
            "tamanho_bytes": st.st_size,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
            "mtime_epoch": st.st_mtime,
            "tempo_hash_s": round(dt, 2),
        }
        print(nome, out[nome]["sha256"][:16], out[nome]["mtime_iso"], f"{dt:.1f}s", flush=True)

out["_meta"] = {
    "tempo_total_s": round(time.time() - t_ini, 2),
    "base": BASE,
    "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}

destino = "D:/_ARQUIVO_SSD_F/TOPO_RF/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/_campanha_2026-09-13/t1_sorocaba/hash_16_alvos.json"
with open(destino, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("gravado em", destino)
