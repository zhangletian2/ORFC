import sys, os, json, numpy as np
_ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_ABLATION_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "utils"))
from cal_bd_rate import BD_RATE

D = 1024.0

# =================== VTM Anchor (blk20) ===================
vtm_cls_R = [0.003171, 0.003849, 0.005213, 0.026109, 0.046051,
             0.110066, 0.221285, 0.616955, 1.028143]
vtm_cls_D = [0.0, 0.0, 0.0, 7.40, 19.40, 48.40, 73.60, 91.80, 95.60]

vtm_seg_R = [0.000427, 0.000532, 0.001061, 0.013066, 0.027212,
             0.091487, 0.180139, 0.470134, 0.795831]
vtm_seg_D = [0.03, 0.23, 2.80, 15.46, 28.83, 55.50, 66.77, 75.98, 79.36]

# =================== DOPQ Variants ===================
# Format: (BPFP_cls, Acc%, BPFP_seg, mIoU%)
#   BPFP_cls = rans_bpt / 1024  (ImageNet, learned prior)
#   BPFP_seg = seg_rans / 1024  (VOC, learned prior)
# Also: BPFP_tr_cls = rans_train_bpt / 1024  (ImageNet histogram)

variants = {}

# (f) Full DOPQ
variants['Full DOPQ'] = {
    'cls_R': [95.08/D, 126.75/D, 382.82/D, 500.18/D],
    'cls_D': [87.40,   92.00,    95.40,    96.20],
    'seg_R': [95.54/D, 127.62/D, 383.25/D, 501.72/D],
    'seg_D': [67.36,   70.74,    74.90,    75.68],
    # histogram-based rate
    'cls_R_tr': [94.94/D, 126.43/D, 381.50/D, 494.40/D],
}

# (b) L_ref -> MSE
variants['L_ref->MSE'] = {
    'cls_R': [92.50/D, 120.98/D, 316.45/D, 397.53/D],
    'cls_D': [33.60,   52.80,    85.00,    88.20],
    'seg_R': [93.49/D, 122.92/D, 319.32/D, 399.60/D],
    'seg_D': [28.72,   40.64,    62.81,    67.35],
    'cls_R_tr': [92.12/D, 120.00/D, 308.21/D, 386.21/D],
}

# (c) - Train R (fzR)
variants['-Train R'] = {
    'cls_R': [95.42/D, 127.28/D, 383.46/D, 504.98/D],
    'cls_D': [73.20,   81.40,    94.00,    96.40],
    'seg_R': [95.46/D, 127.64/D, 384.36/D, 506.69/D],
    'seg_D': [57.38,   59.49,    72.67,    73.53],
    'cls_R_tr': [94.80/D, 126.45/D, 381.09/D, 498.31/D],
}

# (d) - Train C (fzC)
# NOTE: seg_R uses ImageNet rans_bpt as proxy (VOC seg_rate computed from wrong ckpt)
variants['-Train C'] = {
    'cls_R': [94.83/D, 125.67/D, 371.54/D, 482.01/D],
    'cls_D': [87.20,   91.80,    94.40,    96.40],
    'seg_R': [94.83/D, 125.67/D, 371.54/D, 482.01/D],  # ImageNet BPFP as proxy
    'seg_D': [67.07,   69.70,    75.34,    75.40],
    'cls_R_tr': [94.49/D, 125.08/D, 349.64/D, 411.32/D],
}

# (e) - Soft PQ, tau=0, lmbda=0.5 (Hard PQ + ECVQ, mode collapse)
variants['-SoftPQ(tau0,lm0.5)'] = {
    'cls_R': [12.79/D, 21.01/D, 68.90/D, 110.55/D],
    'cls_D': [3.20,    6.20,    44.80,   60.20],
    'seg_R': [18.61/D, 27.23/D, 73.65/D, 115.06/D],
    'seg_D': [3.37,    21.58,   12.51,   23.35],
    'cls_R_tr': [10.78/D, 16.80/D, 36.18/D, 49.00/D],
}

# (e') - Soft PQ, tau=0, lmbda=0 (Hard PQ, no ECVQ)
variants['-SoftPQ(tau0,lm0)'] = {
    'cls_R': [73.00/D, 99.25/D, 327.26/D, 449.94/D],
    'cls_D': [79.80,   87.20,    94.40,    94.60],
    'seg_R': [79.67/D, 109.36/D, 351.58/D, 476.89/D],
    'seg_D': [64.46,   69.56,    73.08,    73.96],
    'cls_R_tr': [73.00/D, 99.25/D, 327.26/D, 449.94/D],
}

# =================== Compute BD-Rate ===================
print('=' * 80)
print(f'{\"Variant\":>25} | {\"Cls BD-Rate(%)\":>14} | {\"Seg BD-Rate(%)\":>14} | {\"Cls overlap\":>14} | {\"Seg overlap\":>14}')
print('-' * 80)

for name, v in variants.items():
    # Classification BD-rate (Acc, higher_better)
    bd_cls = BD_RATE(vtm_cls_R, vtm_cls_D, v['cls_R'], v['cls_D'],
                     piecewise=1, higher_better=True)
    
    # Segmentation BD-rate (mIoU, higher_better)
    bd_seg = BD_RATE(vtm_seg_R, vtm_seg_D, v['seg_R'], v['seg_D'],
                     piecewise=1, higher_better=True)
    
    # Overlap ranges
    cls_D_arr = np.array(v['cls_D'])
    vtm_cls_D_arr = np.array([x for x in vtm_cls_D if x > 0])  # exclude 0%
    cls_lo = max(cls_D_arr.min(), min(vtm_cls_D))
    cls_hi = min(cls_D_arr.max(), max(vtm_cls_D))
    cls_overlap = f'{cls_lo:.1f}-{cls_hi:.1f}' if cls_hi > cls_lo else 'NONE'
    
    seg_D_arr = np.array(v['seg_D'])
    seg_lo = max(seg_D_arr.min(), min(vtm_seg_D))
    seg_hi = min(seg_D_arr.max(), max(vtm_seg_D))
    seg_overlap = f'{seg_lo:.1f}-{seg_hi:.1f}' if seg_hi > seg_lo else 'NONE'
    
    bd_cls_s = f'{bd_cls:+.2f}' if not np.isnan(bd_cls) else 'N/A'
    bd_seg_s = f'{bd_seg:+.2f}' if not np.isnan(bd_seg) else 'N/A'
    
    print(f'{name:>25} | {bd_cls_s:>14} | {bd_seg_s:>14} | {cls_overlap:>14} | {seg_overlap:>14}')

print('=' * 80)
print()
print('NOTE: Negative BD-Rate = bitrate savings vs VTM (better)')
print('      Positive BD-Rate = bitrate overhead vs VTM (worse)')
print()

# =================== Also compute with histogram-based rate ===================
print('=' * 80)
print('BD-Rate using rans_train_bpt (histogram encoding, fairer for fzC):')
print(f'{\"Variant\":>25} | {\"Cls BD-Rate(%)\":>14}')
print('-' * 50)
for name, v in variants.items():
    bd_cls_tr = BD_RATE(vtm_cls_R, vtm_cls_D, v['cls_R_tr'], v['cls_D'],
                        piecewise=1, higher_better=True)
    bd_cls_tr_s = f'{bd_cls_tr:+.2f}' if not np.isnan(bd_cls_tr) else 'N/A'
    print(f'{name:>25} | {bd_cls_tr_s:>14}')
print('=' * 80)

import sys, json, numpy as np
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "utils"))
from cal_bd_rate import BD_RATE

D = 1024.0

# =================== VTM Anchor (blk20) ===================
vtm_cls_R = [0.003171, 0.003849, 0.005213, 0.026109, 0.046051,
             0.110066, 0.221285, 0.616955, 1.028143]
vtm_cls_D = [0.0, 0.0, 0.0, 7.40, 19.40, 48.40, 73.60, 91.80, 95.60]

vtm_seg_R = [0.000427, 0.000532, 0.001061, 0.013066, 0.027212,
             0.091487, 0.180139, 0.470134, 0.795831]
vtm_seg_D = [0.03, 0.23, 2.80, 15.46, 28.83, 55.50, 66.77, 75.98, 79.36]

# =================== Data by variant ===================
# Each: (rans_bpt_list, rans_train_bpt_list, seg_rans_list, acc_list, miou_list)
data = {
    'Full DOPQ': {
        'r':  [95.08, 126.75, 382.82, 500.18],
        'rt': [94.94, 126.43, 381.50, 494.40],
        'sr': [95.54, 127.62, 383.25, 501.72],
        'acc': [87.40, 92.00, 95.40, 96.20],
        'miou': [67.36, 70.74, 74.90, 75.68],
    },
    'Lref->MSE': {
        'r':  [92.50, 120.98, 316.45, 397.53],
        'rt': [92.12, 120.00, 308.21, 386.21],
        'sr': [93.49, 122.92, 319.32, 399.60],
        'acc': [33.60, 52.80, 85.00, 88.20],
        'miou': [28.72, 40.64, 62.81, 67.35],
    },
    '-Train R': {
        'r':  [95.42, 127.28, 383.46, 504.98],
        'rt': [94.80, 126.45, 381.09, 498.31],
        'sr': [95.46, 127.64, 384.36, 506.69],
        'acc': [73.20, 81.40, 94.00, 96.40],
        'miou': [57.38, 59.49, 72.67, 73.53],
    },
    '-Train C': {
        'r':  [94.83, 125.67, 371.54, 482.01],
        'rt': [94.49, 125.08, 349.64, 411.32],
        'sr_proxy': [94.83, 125.67, 371.54, 482.01],  # ImageNet as proxy (VOC seg_rate wrong)
        'acc': [87.20, 91.80, 94.40, 96.40],
        'miou': [67.07, 69.70, 75.34, 75.40],
    },
    'HardPQ(lm0.5)': {
        'r':  [12.79, 21.01, 68.90, 110.55],
        'rt': [10.78, 16.80, 36.18, 49.00],
        'sr': [18.61, 27.23, 73.65, 115.06],
        'acc': [3.20, 6.20, 44.80, 60.20],
        'miou': [3.37, 21.58, 12.51, 23.35],
    },
    'HardPQ(lm0)': {
        'r':  [73.00, 99.25, 327.26, 449.94],
        'rt': [73.00, 99.25, 327.26, 449.94],
        'sr': [79.67, 109.36, 351.58, 476.89],
        'acc': [79.80, 87.20, 94.40, 94.60],
        'miou': [64.46, 69.56, 73.08, 73.96],
    },
}

# =================== COMPUTE ===================
print('='*90)
print('                      DOPQ blk20 Ablation BD-Rate vs VTM (all in %)')
print('='*90)
print()
print('Rate metric choices:')
print('  cls_rans  : rans_bpt/1024     (learned prior, ImageNet features)')
print('  cls_hist  : rans_train_bpt/1024 (histogram, ImageNet features)')  
print('  seg_voc   : seg_rans/1024     (learned prior, VOC features)')
print('  seg_hist_p: rans_train_bpt/1024 (histogram, ImageNet, proxy for VOC)')
print()

header = f'{\"Variant\":>18} | {\"cls_rans\":>9} | {\"cls_hist\":>9} | {\"seg_voc\":>9} | {\"seg_hist_p\":>10} | {\"Acc range\":>12} | {\"mIoU range\":>12}'
print(header)
print('-'*len(header))

for name, v in data.items():
    acc = v['acc']
    miou = v['miou']
    
    # Classification with rans_bpt
    cls_r = [x/D for x in v['r']]
    bd_cls_r = BD_RATE(vtm_cls_R, vtm_cls_D, cls_r, acc, piecewise=1, higher_better=True)
    
    # Classification with rans_train_bpt
    cls_rt = [x/D for x in v['rt']]
    bd_cls_rt = BD_RATE(vtm_cls_R, vtm_cls_D, cls_rt, acc, piecewise=1, higher_better=True)
    
    # Segmentation with seg_rans (VOC)
    if 'sr' in v:
        seg_r = [x/D for x in v['sr']]
    else:
        seg_r = [x/D for x in v['sr_proxy']]
    bd_seg_r = BD_RATE(vtm_seg_R, vtm_seg_D, seg_r, miou, piecewise=1, higher_better=True)
    
    # Segmentation with histogram proxy (ImageNet rans_train_bpt)
    seg_rt = [x/D for x in v['rt']]
    bd_seg_rt = BD_RATE(vtm_seg_R, vtm_seg_D, seg_rt, miou, piecewise=1, higher_better=True)
    
    acc_range = f'{min(acc):.0f}-{max(acc):.0f}'
    miou_range = f'{min(miou):.0f}-{max(miou):.0f}'
    
    fmt = lambda x: f'{x:+.1f}' if not np.isnan(x) else 'N/A'
    print(f'{name:>18} | {fmt(bd_cls_r):>9} | {fmt(bd_cls_rt):>9} | {fmt(bd_seg_r):>9} | {fmt(bd_seg_rt):>10} | {acc_range:>12} | {miou_range:>12}')

print()
print('='*90)
print()

# =================== Delta vs Full DOPQ ===================
print('Contribution of each component (Δ BD-Rate vs Full DOPQ, using cls_rans & seg_voc):')
print()
# Full DOPQ reference
ref_cls = BD_RATE(vtm_cls_R, vtm_cls_D, [x/D for x in data['Full DOPQ']['r']], data['Full DOPQ']['acc'], piecewise=1, higher_better=True)
ref_seg_r = data['Full DOPQ'].get('sr', data['Full DOPQ'].get('sr_proxy'))
ref_seg = BD_RATE(vtm_seg_R, vtm_seg_D, [x/D for x in ref_seg_r], data['Full DOPQ']['miou'], piecewise=1, higher_better=True)

print(f'{\"Variant\":>18} | {\"Δcls_BD\":>9} | {\"Δseg_BD\":>9} | Interpretation')
print('-'*85)
for name, v in data.items():
    if name == 'Full DOPQ':
        print(f'{name:>18} |   (ref)   |   (ref)   | Anchor = {ref_cls:+.1f}% cls, {ref_seg:+.1f}% seg')
        continue
    cls_r = [x/D for x in v['r']]
    bd_c = BD_RATE(vtm_cls_R, vtm_cls_D, cls_r, v['acc'], piecewise=1, higher_better=True)
    seg_r_list = v.get('sr', v.get('sr_proxy'))
    seg_r = [x/D for x in seg_r_list]
    bd_s = BD_RATE(vtm_seg_R, vtm_seg_D, seg_r, v['miou'], piecewise=1, higher_better=True)
    
    dc = bd_c - ref_cls if not np.isnan(bd_c) else float('nan')
    ds = bd_s - ref_seg if not np.isnan(bd_s) else float('nan')
    
    fmt = lambda x: f'{x:+.1f}pp' if not np.isnan(x) else 'N/A'
    
    interp = ''
    if name == 'Lref->MSE':
        interp = 'L_ref is CRITICAL (largest degradation)'
    elif name == '-Train R':
        interp = 'Trainable R important, esp. for seg'
    elif name == '-Train C':
        interp = 'Trainable C marginal for cls, n/a for seg'
    elif name == 'HardPQ(lm0.5)':
        interp = 'MODE COLLAPSE — tau=0 + ECVQ breaks training'
    elif name == 'HardPQ(lm0)':
        interp = 'Soft gradient adds minor benefit'
    
    print(f'{name:>18} | {fmt(dc):>9} | {fmt(ds):>9} | {interp}')
