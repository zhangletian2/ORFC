#!/usr/bin/env python
"""Merge results from JSON files and log files into a unified table."""
import json, glob, os, re, math

V34 = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(V34, 'results', 'soft_pq', 'dinov2_vitl14')
LOG_DIRS = [os.path.join(V34, 'logs_multilayer'),
            os.path.join(V34, 'logs_finetune'),
            os.path.join(V34, 'logs_lowrate')]

rows = {}  # key -> dict

def make_key(layer, K, emb, lmbda, lr, ep, mse, fzR):
    return (layer, K, emb, lmbda, lr, ep, mse, fzR)

# --- 1. Read JSON files ---
for f in sorted(glob.glob(os.path.join(JSON_DIR, '*.json'))):
    try:
        with open(f) as fh:
            r = json.load(fh)
    except:
        continue
    c = r.get('config', {})
    bt = c.get('bottleneck_dim', 0)
    if bt != 1024:
        continue
    layer = c.get('layer', '?')
    K = c.get('K', 0)
    emb = c.get('embedding_dim', 0)
    lmbda = c.get('lmbda', 0.0)
    lr = c.get('lr', 0)
    ep = c.get('epochs', 100)
    mse = c.get('mse_loss', False)
    fzR = c.get('freeze_transform', False)
    tau = c.get('tau_start', 0)

    acc = r.get('soft_pq_acc', 0)
    miou = r.get('soft_pq_miou', 0)
    dl = r.get('soft_pq_delta_l', 0)
    rate = r.get('rate_info', {}).get('rans_bpt', 0)

    key = make_key(layer, K, emb, lmbda, lr, ep, mse, fzR)
    rows[key] = dict(layer=layer, K=K, emb=emb, lmbda=lmbda, lr=lr,
                     ep=ep, mse=mse, fzR=fzR, tau=tau,
                     acc=acc, miou=miou, dl=dl, rate=rate,
                     source='json')

# --- 2. Read log files (fill missing) ---
for log_dir in LOG_DIRS:
    if not os.path.isdir(log_dir):
        continue
    for f in sorted(glob.glob(os.path.join(log_dir, '*.log'))):
        with open(f) as fh:
            text = fh.read()
        if 'Summary:' not in text:
            continue
        fname = os.path.basename(f).replace('.log', '')

        m_layer = re.search(r'Summary:\s*(blk\d+)', text)
        m_K = re.search(r'K=(\d+)', text)
        m_emb = re.search(r'emb=(\d+)', text)
        if not (m_layer and m_K and m_emb):
            continue
        layer = m_layer.group(1)
        K = int(m_K.group(1))
        emb = int(m_emb.group(1))

        # parse config from log header (more reliable than filename)
        m_lmbda_hdr = re.search(r'λ=([\d.]+)', text)
        if m_lmbda_hdr:
            lmbda = float(m_lmbda_hdr.group(1))
        else:
            lmbda = 0.5
            m_l = re.search(r'_l(\d+)', fname)
            if m_l:
                lval = m_l.group(1)
                lmbda = {'00': 0.0, '01': 0.1, '02': 0.2, '03': 0.3,
                         '05': 0.5, '10': 1.0}.get(lval, float(lval) / 10)

        m_lr_hdr = re.search(r'lr=([\d.eE\-]+)', text)
        if m_lr_hdr:
            lr = float(m_lr_hdr.group(1))
        else:
            lr = 0.0003
            m_lr = re.search(r'lr(\d+e\d+)', fname)
            if m_lr:
                lr_str = m_lr.group(1)
                lr = {'3e4': 0.0003, '5e4': 0.0005, '1e4': 0.0001}.get(lr_str, 0.0003)

        m_ep_hdr = re.search(r'epochs=(\d+)', text)
        if m_ep_hdr:
            ep = int(m_ep_hdr.group(1))
        else:
            ep = 100
            m_ep = re.search(r'ep(\d+)', fname)
            if m_ep:
                ep = int(m_ep.group(1))

        m_tau_hdr = re.search(r'τ=([\d.]+)', text)
        tau = float(m_tau_hdr.group(1)) if m_tau_hdr else 0.5

        m_acc = re.findall(r'Codec\s+Acc=([\d.]+)', text)
        m_miou = re.findall(r'Codec\s+mIoU=([\d.]+)', text)
        m_dl = re.findall(r'ΔL_ref=([\d.]+)', text)
        m_rate = re.findall(r'rANS=([\d.]+)', text)

        acc = float(m_acc[-1]) if m_acc else 0
        miou = float(m_miou[-1]) if m_miou else 0
        dl = float(m_dl[-1]) if m_dl else 0
        rate = float(m_rate[-1]) if m_rate else 0

        key = make_key(layer, K, emb, lmbda, lr, ep, False, False)
        if key not in rows:
            rows[key] = dict(layer=layer, K=K, emb=emb, lmbda=lmbda, lr=lr,
                             ep=ep, mse=False, fzR=False, tau=tau,
                             acc=acc, miou=miou, dl=dl, rate=rate,
                             source='log')

# --- 3. Build table ---
all_rows = sorted(rows.values(),
                  key=lambda r: (r['layer'], r['mse'], r['fzR'],
                                 r['K'], r['emb'], r['lmbda'], r['lr']))

uncompressed = {
    'blk05': (0.984, 0.820),
    'blk10': (0.980, 0.819),
    'blk15': (0.982, 0.818),
    'blk20': (0.956, 0.794),
}

cur_layer = ''
for r in all_rows:
    if r['mse'] or r['fzR']:
        continue  # skip ablation for main table
    layer = r['layer']
    if layer != cur_layer:
        if cur_layer:
            print()
        ua, um = uncompressed.get(layer, (1, 1))
        print(f"=== {layer} (uncompressed: Acc={ua*100:.1f}%, mIoU={um*100:.1f}%) ===")
        print(f"{'Config':<28} {'BPFP':>6} {'Acc':>7} {'mIoU':>7} {'ΔL_ref':>9} {'src':>4}")
        print('-' * 68)
        cur_layer = layer

    K, emb, lmbda, lr, ep = r['K'], r['emb'], r['lmbda'], r['lr'], r['ep']
    bpfp = r['rate'] / 1024 if r['rate'] > 0 else 0

    parts = [f"K={K}", f"e{emb}"]
    if lmbda != 0.5:
        parts.append(f"λ={lmbda}")
    if lr != 0.0003:
        parts.append(f"lr={lr}")
    if ep != 100:
        parts.append(f"ep={ep}")
    cfg = ', '.join(parts)

    src = r.get('source', '?')[0]  # j or l
    print(f"{cfg:<28} {bpfp:>6.4f} {r['acc']*100:>6.1f}% {r['miou']*100:>6.1f}% {r['dl']:>9.0f} {src:>4}")

# --- 4. Ablation table ---
print("\n\n=== blk20 Ablation ===")
print(f"{'Config':<40} {'BPFP':>6} {'Acc':>7} {'mIoU':>7}")
print('-' * 65)
for r in all_rows:
    if r['layer'] != 'blk20':
        continue
    if not (r['mse'] or r['fzR']):
        continue
    K, emb, lmbda, lr = r['K'], r['emb'], r['lmbda'], r['lr']
    bpfp = r['rate'] / 1024 if r['rate'] > 0 else 0
    tags = []
    if r['mse']:
        tags.append('MSE')
    if r['fzR']:
        tags.append('fzR')
    tau_val = r.get('tau', 0)
    if tau_val == 0:
        tags.append('τ=0')
    cfg = f"K={K}, e{emb}, λ={lmbda}, lr={lr} [{'/'.join(tags)}]"
    print(f"{cfg:<40} {bpfp:>6.4f} {r['acc']*100:>6.1f}% {r['miou']*100:>6.1f}%")

# --- 5. Best per layer ---
print("\n\n=== Best per layer (main experiments) ===")
print(f"{'Layer':<6} {'Best Acc config':<35} {'Best mIoU config':<35}")
print('-' * 78)
for layer in ['blk05', 'blk10', 'blk15', 'blk20']:
    lr = [r for r in all_rows if r['layer'] == layer and not r['mse'] and not r['fzR']]
    if not lr:
        continue
    ba = max(lr, key=lambda x: x['acc'])
    bm = max(lr, key=lambda x: x['miou'])
    a_cfg = f"K={ba['K']},e{ba['emb']},λ={ba['lmbda']}→{ba['acc']*100:.1f}%"
    m_cfg = f"K={bm['K']},e{bm['emb']},λ={bm['lmbda']}→{bm['miou']*100:.1f}%"
    print(f"{layer:<6} {a_cfg:<35} {m_cfg:<35}")
