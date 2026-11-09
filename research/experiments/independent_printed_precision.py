"""Independent 80-digit, direct-H check of one printed Sionna channel.

This uses only mpmath; it never calls prior project code, uses ray tracing,
or treats the other 15 Munich receiver channels as available.
"""
import json
import sys
from pathlib import Path
from mpmath import mp

mp.dps = 80
SOURCE_NOTEBOOK_SHA = '10499baff91d408e5c64be080e0b0872fe81010e'
REFERENCE_BLOB_SHA = 'f691364eed61adf928b772c3f6255d1c1f36270f'
N0 = mp.mpf('0.01')
I = mp.j
PRINTED = (
    (mp.mpc('0.92982936','0.33584663'), mp.mpc('0.01006189','0.00676795')),
    (mp.mpc('-0.06026739','-0.00641686'), mp.mpc('-0.13674566','-0.01022593')),
)
BALANCED = ((1/mp.sqrt(2),mp.mpf('0')),(mp.mpf('0'),1/mp.sqrt(2)))
EXPECTED = {
    'printed': {
        'circle':('0.025844113326724615','0.05182352183597337'),
        'rank':('0.06808739425621839','0.10646272262942524'),
        'qpsk':('0.16517190571537954','0.41465776263146537'),
    },
    'balanced': {
        'circle':('0.025534279763417538','0.05119838899317461'),
        'rank':('0.00006490090751482276','0.0001298018227169444'),
        'qpsk':('0.00000028665157187919645','0.0000011466108999055833'),
    },
}


def books():
    root2 = mp.sqrt(2)
    g = [mp.mpc(a,b)/root2 for a,b in ((1,1),(1,-1),(-1,1),(-1,-1))]
    qpsk=[(a/root2,b/root2) for a in g for b in g]
    rank=[]
    for m in range(8):
        v=(mp.cos(mp.pi*m/4),mp.sin(mp.pi*m/4))
        for sign in (1,-1):
            rank.append((v[0]*(1+I*sign)/root2,v[1]*(1+I*sign)/root2))
    circle=[(mp.exp(2*I*mp.pi*k/16)/root2,I*mp.exp(2*I*mp.pi*k/16)/root2)
            for k in range(16)]
    return {'circle':circle,'rank':rank,'qpsk':qpsk}


def norm2(z):
    return z.real*z.real + z.imag*z.imag


def projection(H,x):
    return (H[0][0]*x[0]+H[0][1]*x[1], H[1][0]*x[0]+H[1][1]*x[1])


def independent_bounds(H,C):
    if len(C)!=16: raise AssertionError('not 16 symbols')
    y=[projection(H,x) for x in C]
    lowers=[];uppers=[];maxdist=mp.mpf(0);mindist=mp.inf
    for i in range(16):
        values=[]
        for j in range(16):
            if i==j: continue
            d2=norm2(y[i][0]-y[j][0])+norm2(y[i][1]-y[j][1])
            if d2 < mindist:mindist=d2
            if d2 > maxdist:maxdist=d2
            values.append(mp.erfc(mp.sqrt(d2)/(2*mp.sqrt(N0)))/2)
        lowers.append(max(values))
        uppers.append(mp.fsum(values))
    return {'lower':mp.fsum(lowers)/16,'upper':mp.fsum(uppers)/16,
            'min_d2':mindist,'max_d2':maxdist}


def assert_unit_energy(C):
    if len(C)!=16:raise AssertionError('wrong 16-symbol cardinality')
    power=[mp.fsum(norm2(x[port]) for x in C)/16 for port in (0,1)]
    for x in C:
        assert abs(norm2(x[0])+norm2(x[1])-1)<mp.mpf('1e-70')
    for p in power:assert abs(p-mp.mpf('0.5'))<mp.mpf('1e-70')
    return power


def test_q_special_function():
    # Independent Gaussian integral for a few Q values, versus erfc.
    for t in ('0','0.5','2.0','4.0'):
        x=mp.mpf(t)
        via_integral=mp.quad(lambda z:mp.exp(-z*z/2)/mp.sqrt(2*mp.pi),[x,mp.inf])
        via_erfc=mp.erfc(x/mp.sqrt(2))/2
        if abs(via_erfc-via_integral)>mp.mpf('1e-55'):
            raise AssertionError('Q integration disagreement')


def report():
    test_q_special_function()
    C=books()
    for c in C.values(): assert_unit_energy(c)
    for x in C['circle']:
        assert abs((x[0].conjugate()*x[1]).imag-mp.mpf('.5'))<mp.mpf('1e-70')
    data={};max_abs_drift=mp.mpf(0)
    for location,H in [('printed',PRINTED),('balanced',BALANCED)]:
        data[location]={}
        for design,c in C.items():
            entry=independent_bounds(H,c)
            for j,k in enumerate(('lower','upper')):
                drift=abs(entry[k]-mp.mpf(EXPECTED[location][design][j]))
                max_abs_drift=max(max_abs_drift,drift)
                if drift > mp.mpf('5e-11'):
                    raise AssertionError(('historical printed result drift',location,design,k,mp.nstr(drift,20)))
            data[location][design]=entry
    order=[('printed','circle','rank'),('printed','rank','qpsk'),
           ('balanced','qpsk','rank'),('balanced','rank','circle')]
    comparisons=[]
    Emax=mp.sqrt(8)*mp.mpf('5e-9')
    pep_error=2*Emax/(mp.sqrt(2*N0)*mp.sqrt(2*mp.pi))
    for location,a,b in order:
        margin=data[location][b]['lower']-data[location][a]['upper']
        allowance=16*pep_error if location=='printed' else mp.mpf(0)
        if margin<=allowance:raise AssertionError(('strict interval gap failed',location,a,b))
        comparisons.append({'channel':location,'lower_SER_design':a,
                            'higher_SER_design':b,'gap':mp.nstr(margin,60),
                            'max_printed_rounding_gap_loss':mp.nstr(allowance,45),
                            'guaranteed_remaining_gap':mp.nstr(margin-allowance,60)})
    def serialize(entry):return {k:mp.nstr(v,65) for k,v in entry.items()}
    return {'scope':'one eight-decimal printed H; one synthetic balanced negative control',
            'source_notebook_blob':SOURCE_NOTEBOOK_SHA,
            'previous_result_blob':REFERENCE_BLOB_SHA,
            'precision_decimal_digits':mp.dps,
            'N0':'0.01','all_comparisons_passed':True,
            'reference_absolute_drift_max':mp.nstr(max_abs_drift,60),
            'printed_H_fro_rounding_bound':mp.nstr(Emax,60),
            'printed_single_PEP_perturbation_bound':mp.nstr(pep_error,60),
            'printed_full_union_perturbation_bound':mp.nstr(15*pep_error,60),
            'comparisons':comparisons,
            'results':{loc:{name:serialize(entry) for name,entry in rec.items()} for loc,rec in data.items()},
            'limitations': 'high-precision numerical evaluation, not directed-rounding proof; printed H only; no full Munich sample; codebook definition is source-derived'}


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError('no silent overwrite of results')
    output=json.dumps(report(),indent=2,sort_keys=True)+'\n'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(output)
    print(json.dumps({k:v for k,v in json.loads(output).items() if k not in ('results',)},indent=2))


if __name__=='__main__':main(sys.argv[1:])
