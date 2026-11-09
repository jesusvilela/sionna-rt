"""CPU tests for source-separated high-precision printed-channel verification."""
import json
import tempfile
import unittest
from pathlib import Path
from mpmath import mp
import independent_printed_precision as v


class HighPrecisionPrintedTests(unittest.TestCase):
    def test_source_codeword_structure(self):
        code = v.books()
        self.assertEqual(set(code), {'circle','rank','qpsk'})
        for design in code.values():
            self.assertEqual(len(design),16)
            power = v.assert_unit_energy(design)
            for p in power:self.assertLess(abs(p-mp.mpf('.5')),mp.mpf('1e-70'))
        for x in code['rank']:
            self.assertLess(abs((x[0].conjugate()*x[1]).imag),mp.mpf('1e-70'))
        for x in code['circle']:
            self.assertLess(abs((x[0].conjugate()*x[1]).imag-mp.mpf('.5')),mp.mpf('1e-70'))

    def test_gaussian_analytic_integral_control(self):
        v.test_q_special_function()

    def test_no_project_bound_dependencies(self):
        # The direct-channel verifier uses only mpmath and stdlib.
        # This is a code-path separation check, not external peer review.
        text=Path(v.__file__).read_text()
        for prohibited in ('from printed_channel_control import','import scipy','import numpy','from gram_decision_sweep import'):
            self.assertNotIn(prohibited,text)

    def test_six_80_digit_intervals_reproduce_historical_controls(self):
        books=v.books()
        for label,H in (('printed',v.PRINTED),('balanced',v.BALANCED)):
            for name,C in books.items():
                b=v.independent_bounds(H,C)
                for k,n in enumerate(('lower','upper')):
                    self.assertLess(abs(b[n]-mp.mpf(v.EXPECTED[label][name][k])),mp.mpf('5e-11'))
                self.assertGreater(b['min_d2'],0)
                self.assertGreaterEqual(b['upper'],b['lower'])

    def test_strict_rankings_and_print_roundoff_allowance(self):
        result=v.report()
        self.assertTrue(result['all_comparisons_passed'])
        self.assertEqual(len(result['comparisons']),4)
        allowed=mp.mpf(result['printed_single_PEP_perturbation_bound'])*16
        for c in result['comparisons']:
            self.assertGreater(mp.mpf(c['guaranteed_remaining_gap']),0)
            if c['channel']=='printed':
                self.assertAlmostEqual(float(c['max_printed_rounding_gap_loss']),float(allowed),places=11)
        self.assertLess(mp.mpf(result['reference_absolute_drift_max']),mp.mpf('5e-11'))

    def test_existing_result_file_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            outfile=Path(d)/'report.json'
            outfile.write_text('sentinel')
            with self.assertRaises(FileExistsError):v.main(['--out',str(outfile)])
            self.assertEqual(outfile.read_text(),'sentinel')


if __name__=='__main__': unittest.main()
