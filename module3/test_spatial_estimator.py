import unittest
import numpy as np
import spatial_estimator as s

class TestStatisticalPlacement(unittest.TestCase):
    def test_targets_and_sizes_used(self):
        branches=[{'id':1,'points':[[0,0.4,0],[0.3,0.9,0]]},
                  {'id':2,'points':[[0.4,0.3,0],[0.6,0.8,0]]}]
        fruit={'mean':.05,'std':.005,'lower':.04,'upper':.06}
        tip=s.fit_beta([.2,.3,.5,.6,.7])
        height=s.fit_beta([.55,.6,.65,.7,.75])
        placed,failed,logs=s.place_fruits(branches,3,fruit,tip,height,0,1,.2,
            np.random.default_rng(42),attempts=40,min_anchor_distance=.01)
        self.assertEqual(len(placed)+len(failed),3)
        self.assertTrue(all(.04<=x['diameter']<=.06 for x in placed))
        self.assertTrue(all(x['center'][1]-x['radius']>=.2 for x in placed))
        self.assertTrue(all(x['target_height_fraction'] is not None for x in placed))
        self.assertTrue(all('sampled_tip_fraction' in x for x in logs))
    def test_empirical_height_changes_branch_choice(self):
        branches=s.prepare_branches([
            {'id':1,'points':[[0,.2,0],[.3,.4,0]]},
            {'id':2,'points':[[0,.7,0],[.3,.9,0]]}])
        rng=np.random.default_rng(1)
        low=s.candidate_order(branches,.5,.3,0,1,rng)
        high=s.candidate_order(branches,.5,.8,0,1,rng)
        self.assertEqual(low[0]['branch_id'],1)
        self.assertEqual(high[0]['branch_id'],2)

if __name__=='__main__':unittest.main()
