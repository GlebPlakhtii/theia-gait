"""Synthetic imbalance/leakage checks; no frozen or held-out data are opened."""
from dataclasses import replace
import unittest

import numpy as np

from gait_from_scratch_sampling import WindowIdentity, conditional_balancing_weights, sample_training_epoch


def identity(person='P1',trial='P1:T1',source='ljmu360',method='sapiens2_1b'):
    return WindowIdentity(person,trial,source,method,'train',True)


class SamplingTests(unittest.TestCase):
    def test_unequal_windows_and_trials_do_not_change_person_mass(self):
        records=[identity()]*900+[identity(trial='P1:T2')]*100+[identity('P2','P2:T1')]*10
        w=conditional_balancing_weights(records)
        self.assertAlmostEqual(w[:900].sum(),.25)
        self.assertAlmostEqual(w[900:1000].sum(),.25)
        self.assertAlmostEqual(w[1000:].sum(),.5)
        indices,audit=sample_training_epoch(records,0)
        counts=audit['methods']['sapiens2_1b']
        self.assertTrue(.4<len([i for i in indices if i<1000])/len(indices)<.6)
        self.assertTrue(.4<counts['physical_trial_draw_counts']['P1:T1']/counts['participant_draw_counts']['P1']<.6)

    def test_publisher_cannot_exceed_half_even_with_many_more_windows(self):
        records=[identity()]*4+[identity('PG1','PG1:T1','progait','publisher')]*1000
        indices,audit=sample_training_epoch(records,0)
        self.assertEqual(len(indices),8)
        self.assertEqual(audit['methods']['publisher']['samples'],4)
        self.assertEqual(audit['methods']['sapiens2_1b']['samples'],4)

    def test_sparse_or_absent_auxiliary_stays_below_cap(self):
        records=[identity()]*10+[identity('PG1','PG1:T1','progait','publisher')]*3
        self.assertEqual(sample_training_epoch(records,0)[1]['methods']['publisher']['samples'],3)
        self.assertEqual(sample_training_epoch(records[:10],0)[1]['methods']['publisher']['samples'],0)
        with self.assertRaises(ValueError):sample_training_epoch(records[10:],0)

    def test_epoch_seed_is_reproducible_without_freezing_all_epochs(self):
        records=[identity()]*100
        a=sample_training_epoch(records,0)[0];b=sample_training_epoch(records,0)[0]
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,sample_training_epoch(records,1)[0]))

    def test_holdout_unknown_and_provisional_rows_are_rejected(self):
        for bad in [replace(identity(),split='validation'),replace(identity(),split='test'),
                    replace(identity(),known_participant=False),replace(identity(),participant_id=''),
                    replace(identity(),source='additional_input'),replace(identity(),source='movi'),
                    replace(identity(),source='mpi3dhp')]:
            with self.assertRaises(ValueError):sample_training_epoch([identity(),bad],0)
            with self.assertRaises(ValueError):conditional_balancing_weights([identity(),bad])

    def test_methods_have_separate_statistical_mass_and_trial_identity_is_shared(self):
        records=[identity('PG1','PG1:T1','progait')]*10+[identity('PG1','PG1:T1','progait','publisher')]*30
        weights=conditional_balancing_weights(records)
        self.assertAlmostEqual(weights[:10].sum(),1.)
        self.assertAlmostEqual(weights[10:].sum(),1.)
        sample_training_epoch(records,0)
        with self.assertRaises(ValueError):
            sample_training_epoch([identity(),identity('P2','P1:T1')],0)
        with self.assertRaises(ValueError):
            sample_training_epoch([identity(method='publisher')],0)


if __name__=='__main__':unittest.main()
