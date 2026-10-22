import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
SPEC = importlib.util.spec_from_file_location('release',
    Path(__file__).resolve().parents[1]/'tools/two_node_release.py')
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)

class ReleaseTests(unittest.TestCase):
    def test_promotion_requires_all_gates(self):
        replay = {str(d):{'passed':True} for d in (128,144)}
        baseline = {'elapsed_seconds':100,'peak_host_gib':[420,421]}
        optimized = {'elapsed_seconds':90,'peak_host_gib':[421,422]}
        self.assertEqual(RELEASE.choose_profile(replay,baseline,optimized)[0],'optimized')
        optimized['elapsed_seconds'] = 106
        self.assertEqual(RELEASE.choose_profile(replay,baseline,optimized)[0],'baseline')
        optimized['elapsed_seconds'] = 90; optimized['peak_host_gib'][1] = 424
        self.assertEqual(RELEASE.choose_profile(replay,baseline,optimized)[0],'baseline')
        optimized['peak_host_gib'][1] = 422; replay['144']['passed'] = False
        self.assertEqual(RELEASE.choose_profile(replay,baseline,optimized)[0],'baseline')

if __name__ == '__main__': unittest.main()
