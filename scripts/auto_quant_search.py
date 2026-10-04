#!/usr/bin/env python3
# Block-grouped per-layer quant sensitivity search for LLM mnn models,
# based on tools/converter/tools/auto_quant.py (MNNConvert --testdir diff rate).
#
# Subcommands:
#   prepare  : build testdir (input.json + <input>.txt) and golden <output>.txt
#              from the fp16 llm.mnn (run once in pymnn).
#   search   : greedy block-grouped W8->W4 bit search under dynamic-A8 activations,
#              driven by MNNConvert convert+test error rate vs target.
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np


def log(msg):
    print(f"[autoq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- prepare ----
def cmd_prepare(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    testdir = args.testdir
    os.makedirs(testdir, exist_ok=True)
    L = args.n_tokens
    hidden = args.hidden_size

    # 1. token ids from calib text (first line)
    texts = [l.strip() for l in open(args.calib, encoding="utf-8") if l.strip()][:1]
    assert texts, "empty calib file"
    tok = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    ids = tok(texts[0], add_special_tokens=False)["input_ids"][:L]
    ids = (ids + [(tok.eos_token_id or 0)] * L)[:L]
    log(f"token ids ({len(ids)}): {ids[:12]}...")

    # 2. fp model (bf16) for embeddings + golden logits
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        hs = model.get_input_embeddings()(torch.tensor([ids])).to(torch.float32)  # [1,L,H]
        pos = torch.arange(L, dtype=torch.int64).unsqueeze(0)  # [1, L] for HF rope
        out = model(inputs_embeds=hs.to(torch.bfloat16),
                    attention_mask=None, position_ids=pos)
        golden = out.logits[:, -1, :].to(torch.float32)  # [1, V], logits_index=-1 semantics
    log(f"embeddings {tuple(hs.shape)} std={float(hs.std()):.4f}; golden logits {tuple(golden.shape)}")
    hs = hs.reshape(L, 1, hidden).numpy()

    # 3. additive causal mask per exported-graph convention (model.py:full_attention_mask)
    cfg = json.load(open(os.path.join(args.hf_model_dir, "config.json")))
    att_type = cfg.get("attention_type", "full")
    fmin = np.float32(np.finfo(np.float32).min)
    if att_type == "sliding":
        win = int(cfg.get("sliding_window", 4096))
        q = np.arange(L).reshape(-1, 1)
        k = np.arange(L).reshape(1, -1)
        keep = (k <= q) & (k > q - win)
    else:
        keep = np.tril(np.ones((L, L), dtype=bool))
    masks = np.where(keep, np.float32(0.0), fmin).reshape(1, 1, L, L)

    def dump(name, arr):
        with open(os.path.join(testdir, f"{name}.txt"), "w") as f:
            f.write("\n".join(f"{v:.9e}" for v in arr.astype(np.float64).reshape(-1)))
        log(f"wrote {name}.txt ({arr.size} vals)")

    dump("input_ids", hs)
    dump("attention_mask", masks)
    dump("position_ids", np.arange(L, dtype=np.int32).reshape(1, L))
    dump("logits_index", np.array([-1], dtype=np.int32))
    with open(os.path.join(testdir, "input.json"), "w") as f:
        json.dump({"inputs": [
            {"name": "input_ids", "shape": [L, 1, hidden]},
            {"name": "attention_mask", "shape": [1, 1, L, L]},
            {"name": "position_ids", "shape": [1, L]},
            {"name": "logits_index", "shape": [1]},
        ], "outputs": ["logits"]}, f, indent=2)
    with open(os.path.join(testdir, "logits.txt"), "w") as f:
        for v in golden.reshape(-1):
            f.write(f"{v:.9e}\n")
    log("golden logits.txt written; prepare done")


# ----------------------------------------------------------------- search ----
def run_mnnconvert(mnnconvert, model_dir, dstmodel, dstjson, testdir, fwdjson, hqq):
    cmd = (f"{mnnconvert} -f ONNX --modelFile llm.onnx --MNNModel {dstmodel} "
           f"--allowCustomOp --transformerFuse --saveExternalData "
           f"--transformerFuseC4=1 --transformerFuseQkvProj=1 "
           f"--transformerFuseGateUpProj=1 --transformerFuseLnProj=1 "
           f"--weightQuantBits=8 --weightQuantAsymmetric=0 "
           f"--compressionParamsFile {dstjson} --testdir {testdir} "
           f"--thredhold 0.001 --testconfig {fwdjson} --alignDenormalizedValue 0 ")
    if hqq:
        cmd += "--hqq "
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=model_dir, capture_output=True, text=True)
    dt = time.time() - t0
    info = p.stdout + p.stderr
    if not os.path.exists(dstmodel):
        log(f"MNNConvert FAILED ({dt:.0f}s): tail of log:\n{info[-1500:]}")
        raise RuntimeError("convert failed")
    return get_rate(info), dt, info


def get_rate(loginfo):
    rate = 0.0
    for line in loginfo.split("\n"):
        if "TESTERROR" not in line or "absMaxV" not in line:
            continue
        try:
            content = line.split("absMaxV:")[1].replace(" ", "")
            maxv, diffmax = content.split("-DiffMax")
            maxv, diffmax = float(maxv), float(diffmax)
            if maxv > 0.01 and diffmax / maxv > rate:
                rate = diffmax / maxv
        except Exception:
            pass
    return rate


class QuantInfo:
    def __init__(self, json_path):
        with open(json_path) as f:
            self.compress = json.load(f)
        self.dstjson = json_path
        self.layers = []  # (block_idx_or_None, special_name_or_None, layer_ref)
        for algo in self.compress.get("algo", []):
            if algo.get("type") != "QUANTIZE":
                continue
            qp = algo.get("quantParams", {})
            for layer in qp.get("layer", []):
                conv = layer.get("conv")
                if conv is None or conv["kernelSize"][0] * conv["kernelSize"][1] != 1:
                    continue
                name = layer.get("opName", "")
                m = re.search(r"/layers\.(\d+)/", name) or re.search(r"/blocks\.(\d+)/", name)
                blk = int(m.group(1)) if m else None
                special = None
                low = name.lower()
                if "lm_head" in low:
                    special = "lm_head"
                elif blk is None and ("embed" in low or "tok_embeddings" in low):
                    special = "embed"
                self.layers.append((blk, special, layer))
            break

    def groups(self):
        blocks = sorted({b for b, s, _ in self.layers if b is not None})
        specials = sorted({s for b, s, _ in self.layers if s})
        return blocks, specials

    def set_group(self, blk, bits):
        for b, s, layer in self.layers:
            if b == blk:
                layer["weight"][0]["bits"] = bits

    def set_special(self, special, bits):
        for b, s, layer in self.layers:
            if s == special:
                layer["weight"][0]["bits"] = bits

    def set_blocksize(self, blk_size):
        for _, _, layer in self.layers:
            layer["weight"][0]["blockSize"] = blk_size

    def update(self):
        with open(self.dstjson, "w") as f:
            json.dump(self.compress, f, indent=4)


def cmd_search(args):
    model_dir = args.model_dir
    mnnconvert = args.mnnconvert
    dstmodel = os.path.abspath(args.dstmodel)
    dstjson = dstmodel + ".json"
    fwdjson = os.path.abspath(args.fwdjson)
    testdir = os.path.abspath(args.testdir)
    hqq = args.hqq

    n_layers_total = 0
    # baseline conversion to obtain compression json
    log("baseline conversion (all W8, dynamic A8)...")
    rate0, dt, info = run_mnnconvert(mnnconvert, model_dir, dstmodel, dstjson,
                                     testdir, fwdjson, hqq)
    log(f"baseline rate={rate0:.5f} ({dt:.0f}s); convert output head: {info[:400]!r}")
    qi = QuantInfo(dstjson)
    if not qi.layers:
        head = open(dstjson).read()[:500]
        raise RuntimeError(f"compression json has 0 conv1x1 layers; json head: {head}")
    qi = QuantInfo(dstjson)
    blocks, specials = qi.groups()
    n_layers_total = len(qi.layers)
    log(f"groups: {len(blocks)} blocks, specials={specials}, conv1x1 layers={n_layers_total}")
    with open(args.report, "w") as rep:
        rep.write(f"baseline_rate {rate0:.6f}\n")
        rep.write(f"groups blocks={blocks} specials={specials} layers={n_layers_total}\n")

    if rate0 > args.rate:
        log(f"baseline W8 rate {rate0:.5f} already > target {args.rate}; "
            f"search will find best-effort mix")

    decisions = {}
    for b in blocks:
        qi.set_group(b, 4)
        qi.update()
        rate, dt, _ = run_mnnconvert(mnnconvert, model_dir, dstmodel, dstjson,
                                     testdir, fwdjson, hqq)
        ok = rate <= args.rate
        if not ok:
            qi.set_group(b, 8)
            qi.update()
        decisions[f"blocks.{b}"] = (4 if ok else 8, rate)
        log(f"block {b}: W4 -> rate={rate:.5f} {'KEEP W4' if ok else 'rollback W8'} ({dt:.0f}s)")
        with open(args.report, "a") as rep:
            rep.write(f"blocks.{b} {'4' if ok else '8'} rate={rate:.6f}\n")

    for sp in specials:
        qi.set_special(sp, 4)
        qi.update()
        rate, dt, _ = run_mnnconvert(mnnconvert, model_dir, dstmodel, dstjson,
                                     testdir, fwdjson, hqq)
        ok = rate <= args.rate
        if not ok:
            qi.set_special(sp, 8)
            qi.update()
        decisions[sp] = (4 if ok else 8, rate)
        log(f"special {sp}: W4 -> rate={rate:.5f} {'KEEP W4' if ok else 'rollback W8'}")
        with open(args.report, "a") as rep:
            rep.write(f"{sp} {'4' if ok else '8'} rate={rate:.6f}\n")

    # final verify with the settled json
    qi.update()
    rate_final, dt, _ = run_mnnconvert(mnnconvert, model_dir, dstmodel, dstjson,
                                       testdir, fwdjson, hqq)
    log(f"FINAL rate={rate_final:.5f} (target {args.rate})")
    with open(args.report, "a") as rep:
        rep.write(f"final rate={rate_final:.6f}\n")
        for k, (bits, r) in decisions.items():
            rep.write(f"# {k}: W{bits} (probe rate {r:.6f})\n")
    log("search done; dst model: " + dstmodel)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare")
    q.add_argument("--model_dir", default=None)
    q.add_argument("--hf_model_dir", required=True)
    q.add_argument("--calib", required=True)
    q.add_argument("--testdir", required=True)
    q.add_argument("--embed_file", default="embeddings_bf16.bin")
    q.add_argument("--hidden_size", type=int, default=2048)
    q.add_argument("--n_tokens", type=int, default=48)
    q.set_defaults(func=cmd_prepare)

    s = sub.add_parser("search")
    s.add_argument("--model_dir", required=True)
    s.add_argument("--mnnconvert", required=True)
    s.add_argument("--dstmodel", required=True)
    s.add_argument("--testdir", required=True)
    s.add_argument("--fwdjson", required=True)
    s.add_argument("--rate", type=float, default=0.05)
    s.add_argument("--hqq", type=int, default=1)
    s.add_argument("--report", required=True)
    s.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
