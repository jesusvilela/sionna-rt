"""One frozen held-out synthetic ablation of Bargmann B vs anchored pairwise features.

The preregistration at research/PREREG_BARGMANN_ABLATION_2026-11-05.md
was committed BEFORE model fitting. No Sionna/GPU/physical channels are used.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

FROZEN_SHA256 = '16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6'
PHIS = (0., 90., 180.)
TRAIN_THETAS = (0., 45., 90.)
TEST_THETAS = (22.5, 67.5)
RATIOS = (.01, .03, .10, .35)
REF_VECTORS = (
    np.array([1, 0], dtype=complex),
    np.array([1, 1], dtype=complex)/np.sqrt(2),
    np.array([1, 1j], dtype=complex)/np.sqrt(2),
)


def projector(v):
    v = np.asarray(v, dtype=complex)
    return np.outer(v, v.conj()) / np.vdot(v, v).real


REFS = tuple(projector(v) for v in REF_VECTORS)


def from_case(case):
    g = case['channel']
    re, im = g['cross_correlation_real'], g['cross_correlation_imag']
    G = np.array([[g['port_gain_split'][0], re+1j*im],
                  [re-1j*im, g['port_gain_split'][1]]], dtype=complex)
    w = np.linalg.eigvalsh(G)
    if w[1] <= 0 or w[0] < -1e-12:
        raise ValueError('invalid Gram')
    if (w[1]-w[0]) / sum(w) < 1e-7:
        return None
    P = (G-w[0]*np.eye(2))/(w[1]-w[0])
    P = (P+P.conj().T)/2
    np.testing.assert_allclose(P@P, P, atol=5e-12)
    np.testing.assert_allclose(P.conj().T, P, atol=5e-12)
    return P


def overlap(P, Q):
    z = np.trace(P@Q)
    if abs(z.imag) > 1e-10 or z.real < -1e-10 or z.real > 1+1e-10:
        raise ValueError('nonphysical overlap')
    return float(np.clip(z.real,0,1))


def bargmann(P,Q,R):
    return complex(np.trace(P@Q@R))


def source_rows(path):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FROZEN_SHA256:
        raise ValueError('frozen 75-case source SHA mismatch')
    dataset=json.loads(raw)
    if len(dataset['cases']) != 75 or dataset['n_cases'] != 75:
        raise ValueError('not the frozen 75-case sweep')
    groups={}
    isotropic=0
    for c in dataset['cases']:
        r,theta,phi = (float(c['grid'][k]) for k in
                       ('eigenvalue_ratio','theta_deg','phi_deg'))
        if r == 1.:
            isotropic+=1
            if from_case(c) is not None:
                raise AssertionError('isotropic Gram produced a spurious projector')
            continue
        if r not in RATIOS or theta not in (*TRAIN_THETAS, *TEST_THETAS) or phi not in PHIS:
            raise AssertionError('unregistered synthetic grid entry')
        p=from_case(c)
        if p is None:
            raise AssertionError('nongenerate r has degenerate principal projector')
        group=(r,theta)
        groups.setdefault(group,{})[phi] = {'P':p,'target':int(c['rank_one_vs']['circular_16psk']=='A'),
                                            'archived_decision':c['rank_one_vs']['circular_16psk']}
    if isotropic!=15 or len(groups)!=20:
        raise AssertionError(f'isotropic/groups count mismatch: {isotropic}/{len(groups)}')
    for grp,records in groups.items():
        if sorted(records) != list(PHIS):
            raise AssertionError('missing phi in group '+str(grp))
    return groups, sha


def make_features(groups):
    rows=[]
    for (r,theta), nodes in sorted(groups.items()):
        B=bargmann(*(nodes[phi]['P'] for phi in PHIS))
        for phi in PHIS:
            local=nodes[phi]['P']
            other=[nodes[j]['P'] for j in PHIS if j !=phi]
            baseline = [r] + [overlap(local, anchor) for anchor in REFS]
            baseline += [overlap(local, x) for x in other]
            rows.append({'group':(r,theta),'r':r,'theta':theta,'phi':phi,
                         'x_base':baseline,'x_plus':baseline+[B.real,B.imag],
                         'y':nodes[phi]['target'],
                         'decision':nodes[phi]['archived_decision'],
                         'B_re':B.real,'B_im':B.imag})
    return rows


def model():
    return make_pipeline(StandardScaler(),
         LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,class_weight=None))


def run(path):
    groups,sha=source_rows(path)
    rows=make_features(groups)
    train=[i for i,x in enumerate(rows) if x['theta'] in TRAIN_THETAS]
    test=[i for i,x in enumerate(rows) if x['theta'] in TEST_THETAS]
    if len(rows)!=60 or len(train)!=36 or len(test)!=24:
        raise AssertionError('pre-registered split size drift')
    train_groups={rows[i]['group'] for i in train}
    test_groups={rows[i]['group'] for i in test}
    if len(train_groups)!=12 or len(test_groups)!=8 or train_groups & test_groups:
        raise AssertionError('train/test group leakage')
    Y=np.array([x['y'] for x in rows],dtype=int)
    if len(set(Y[train])) != 2:
        raise AssertionError('training has only one label')
    predictions={}
    for key,attr in [('pairwise','x_base'),('plus_B','x_plus')]:
        X=np.array([x[attr] for x in rows],dtype=float)
        if not np.all(np.isfinite(X)):
            raise AssertionError('nonfinite features')
        pipe=model()
        pipe.fit(X[train],Y[train])
        predictions[key]=pipe.predict_proba(X[test])[:,1]
    y=Y[test]
    lp=predictions['pairwise'];lb=predictions['plus_B']
    score_pairwise=float(brier_score_loss(y,lp))
    score_B=float(brier_score_loss(y,lb))
    delta=score_B-score_pairwise
    group_deltas={str(grp):float(np.mean([(lb[k]-y[k])**2-(lp[k]-y[k])**2
                      for k,i in enumerate(test) if rows[i]['group']==grp])) for grp in sorted(test_groups)}
    values=np.array(list(group_deltas.values()))
    rng=np.random.default_rng(6105)
    draws=rng.integers(0,8,size=(5000,8))
    CI=np.percentile(np.mean(values[draws],axis=1),[2.5,97.5])
    promoted=bool(delta <= -.02 and CI[1] < 0.)
    status='provisional_synthetic_B_usefulness_only' if promoted else 'retire_B_engineering_prediction_claim'
    return {'date':'2026-11-05','source_sha256':sha,'synthetic_data_only':True,
            'n_grid':75,'n_excluded_isotropic':15,
            'train_n':len(train),'test_n':len(test),
            'train_groups':sorted([list(g) for g in train_groups]),
            'test_groups':sorted([list(g) for g in test_groups]),
            'train_labels':dict(Counter(Y[train].tolist())),
            'test_labels':dict(Counter(y.tolist())),
            'primary_metric':'held_out_Brier_loss_lower_better',
            'baseline_Brier':score_pairwise,'plus_B_Brier':score_B,
            'delta_plus_B_minus_pairwise':float(delta),
            'group_bootstrap_delta_CI95':[float(x) for x in CI],
            'promotion_threshold':{'delta_max':-.02,'bootstrap_upper_below':0},
            'B_useful':promoted,'decision':status,
            'group_deltas':group_deltas,
            'heldout_predictions':[{'r':rows[i]['r'],'theta':rows[i]['theta'],
                'phi':rows[i]['phi'],'archive_label':rows[i]['decision'],
                'target':int(y[k]),'p_pairwise':float(lp[k]),'p_plus_B':float(lb[k])}
                for k,i in enumerate(test)],
            'environment':{'numpy':np.__version__,'sklearn':sklearn.__version__},
            'limitations':[
             'Synthetically defined (r,theta,phi) triples are not spatial or measured channel triples',
             'Only 8 held-out groups, not independent wireless propagation scenes',
             'Baseline anchors capture full local projector, so B has no information-theoretic increment for per-site target',
             'Bootstrap quantifies synthetic-grid group variability, not physical-world uncertainty',
             'One frozen classifier and split, no hyperparameter search']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError('refuse to overwrite archived trial')
    result=run(a.source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in
        ('baseline_Brier','plus_B_Brier','delta_plus_B_minus_pairwise',
         'group_bootstrap_delta_CI95','B_useful','decision',
         'train_n','test_n','test_labels')},indent=2))


if __name__=='__main__':main()
