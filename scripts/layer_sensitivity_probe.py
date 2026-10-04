#!/usr/bin/env python3
# Per-layer quantization sensitivity probe (torch-based).
#
# Answers: which transformer layers tolerate W4 weights / A8 (dynamic int8)
# activations with acceptable end-to-end logits error, for precision/speed
# balancing. Replaces the MNNConvert auto_quant flow, which silently skips
# weight quantization for external-weight LLM graphs.
#
# Method:
#   1. baseline forward on N calib texts -> golden last-token logits + per-layer
#      input activations cache.
#   2. for each transformer block: fake-quantize that block's Linear weights
#      (W8-int8 / W4-blk64) and/or its inputs (A8 dynamic per-token), re-run
#      the block on cached inputs, propagate through the unquantized tail,
#      measure logits MSE (relative) vs golden.
#   3. rank blocks/linear groups by error -> report which tolerate W4/A8.
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F


def log(msg):
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fake_quant_weight(w, bits, block=64):
    """Symmetric block-wise weight fake quant (MNN blk64 style)."""
    wf = w.float()
    oc = wf.shape[0]
    flat = wf.reshape(oc, -1)
    if block and block > 0:
        n = flat.shape[1]
        pad = (-n) % block
        if pad:
            flat = F.pad(flat, (0, pad))
        blocks = flat.reshape(oc, -1, block)
        scale = blocks.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
        scale = scale.clamp(min=1e-12)
        q = (blocks / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        out = (q * scale).reshape(oc, -1)[:, :n]
    else:
        scale = flat.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
        scale = scale.clamp(min=1e-12)
        out = (flat / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * scale
    return out.reshape_as(wf).to(w.dtype)


def fake_quant_act_dynamic(x, bits=8):
    """Symmetric per-token dynamic activation fake quant (MNN DynamicQuant style)."""
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min=1e-12)
    q = (xf / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (q * scale).to(x.dtype)


class LayerQuantizer:
    """Patches Linears of one module with fake-quant forward."""

    def __init__(self, module, w_bits, w_block, a_bits):
        self.handles = []
        self.w_bits, self.w_block, self.a_bits = w_bits, w_block, a_bits
        for name, mod in module.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(mod, inp, out):
            w = fake_quant_weight(mod.weight.data, self.w_bits, self.w_block)
            x = inp[0]
            if self.a_bits:
                x = fake_quant_act_dynamic(x, self.a_bits)
            with torch.no_grad():
                return F.linear(x, w, mod.bias)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def cmd_probe(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    blocks = model.model.layers
    n_blocks = len(blocks)
    log(f"model loaded: {n_blocks} blocks, dtype={next(model.parameters()).dtype}")

    # calib inputs
    texts = [l.strip() for l in open(args.calib, encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(texts), size=min(args.n_samples, len(texts)), replace=False)
    texts = [texts[i] for i in sorted(idx)]
    input_ids = [tok(t, add_special_tokens=False)["input_ids"][:args.seq_len] for t in texts]
    input_ids = [torch.tensor(x) if x else torch.tensor([tok.eos_token_id or 0]) for x in input_ids]
    # group by length for efficiency
    L = max(len(x) for x in input_ids)
    ids = torch.stack([F.pad(x, (0, L - len(x)), value=tok.eos_token_id or 0) for x in input_ids])
    attn = (ids != (tok.eos_token_id or 0)).long()
    log(f"calib: {ids.shape[0]} samples, len={L}")

    # 1. baseline full forward: golden logits + per-block input capture
    emb = model.model.embed_tokens(ids)
    cache = {}

    def mk_capture(i):
        def hook(mod, cargs):
            cache[i] = cargs[0].detach()
        return hook

    handles = [blk.register_forward_pre_hook(mk_capture(i)) for i, blk in enumerate(blocks)]
    golden = model(inputs_embeds=emb, attention_mask=None).logits[:, -1, :].float()
    for h in handles:
        h.remove()
    log(f"baseline forward done, cached {len(cache)} block inputs")

    mse = lambda a, b: float(((a - b) ** 2).mean() / ((b ** 2).mean() + 1e-9))

    def full_forward_with_block_patched(b, wb, wa):
        def mk_swap(i):
            def hook(mod, cargs):
                return (cache[i],) + cargs[1:]
            return hook
        sw = blocks[b].register_forward_pre_hook(mk_swap(b))
        q = LayerQuantizer(blocks[b], wb, args.block, wa)
        out = model(inputs_embeds=emb, attention_mask=None).logits[:, -1, :].float()
        q.remove()
        sw.remove()
        return out

    cfg = {k: v for k, v in vars(args).items() if k != "func"}
    report = {"config": cfg, "blocks": {}}
    # 2. per-block quantized re-run (model recomputes the clean prefix,
    #    block b's input is swapped to the cached baseline input)
    for b in range(n_blocks):
        for tag, wb, wa in (("w4a8", 4, 8), ("w4a16", 4, 0), ("w8a8", 8, 8)):
            if args.modes and tag not in args.modes.split(","):
                continue
            err = mse(full_forward_with_block_patched(b, wb, wa), golden)
            report["blocks"].setdefault(f"block{b}", {})[tag] = err
        log(f"block {b}: " + ", ".join(
            f"{t}={report['blocks'][f'block{b}'][t]:.5f}" for t in report["blocks"][f"block{b}"]))

    # 3. whole-model references
    for tag, wb, wa in (("all_w4a8", 4, 8), ("all_w8a8", 8, 8)):
        if args.modes and tag not in args.modes.split(","):
            continue
        q = LayerQuantizer(model.model, wb, args.block, wa)
        out = model(inputs_embeds=cache[0], attention_mask=None)["logits"]
        q.remove()
        err = mse(out[:, -1, :].float(), golden)
        report.setdefault("whole", {})[tag] = err
        log(f"{tag}: err={err:.5f}")

    report["note"] = ("error = relative MSE of last-token logits vs bf16 baseline, "
                      "averaged over calib samples; w4a8 = W4 blk weights + A8 dynamic "
                      "acts on this block only, others fp")
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
    log(f"report written: {args.out}")

    # ranking summary
    for tag in next(iter(report["blocks"].values())).keys():
        errs = [(b, d[tag]) for b, d in report["blocks"].items()]
        errs.sort(key=lambda x: x[1])
        log(f"[{tag}] most tolerant 5: {[(b, round(e,5)) for b, e in errs[:5]]}")
        log(f"[{tag}] least tolerant 5: {[(b, round(e,5)) for b, e in errs[-5:]]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("probe")
    q.add_argument("--model", required=True, help="HF model dir (bf16 weights)")
    q.add_argument("--calib", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--n_samples", type=int, default=16)
    q.add_argument("--seq_len", type=int, default=96)
    q.add_argument("--block", type=int, default=64)
    q.add_argument("--modes", default="w4a8,w4a16,w8a8,all_w4a8,all_w8a8")
    q.set_defaults(func=cmd_probe)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
