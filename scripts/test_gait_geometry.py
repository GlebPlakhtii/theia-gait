import unittest
import numpy as np
from gait_geometry import NAMES, bend, unsigned_bend_3d, measure_segment, filtered, stats


class GeometryTests(unittest.TestCase):
    def test_signed_bend_and_mirror(self):
        h=np.array([[0.,2.]]);k=np.array([[0.,1.]]);a=np.array([[0.,0.]])
        self.assertAlmostEqual(bend(h,k,a,1)[0],0)
        a=np.array([[-1.,0.]])
        self.assertAlmostEqual(bend(h,k,a,1)[0],45)
        mirror=np.array([-1.,1.])
        np.testing.assert_allclose(bend(h,k,a,1),bend(h*mirror,k*mirror,a*mirror,-1))
        np.testing.assert_allclose(bend(h,k,a,1),bend(h+83,k+83,a+83,1))

    def test_3d_rotation_and_depth_ambiguity(self):
        a=np.array([[0.,-1.,0.]]);b=np.array([[1.,-1.,0.]])
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(unsigned_bend_3d(a,b),unsigned_bend_3d(a@R,b@R))
        alternate=b.copy();alternate[:,2]=5
        np.testing.assert_array_equal(b[:,:2],alternate[:,:2])
        self.assertGreater(abs(unsigned_bend_3d(a,b)[0]-unsigned_bend_3d(a,alternate)[0]),20)

    def skeleton(self):
        x=np.ones((60,len(NAMES),2));t=np.arange(60)/30
        def put(n,xy):x[:,NAMES.index(n)]=xy
        for s,sign in [('left',1),('right',-1)]:
            for j,y in [('shoulder',4),('hip',2),('knee',1),('ankle',0),('heel',0),('big_toe',0),('small_toe',0),('elbow',3),('wrist',2),('ear',5)]:
                put(s+'_'+j,[sign*.25,y])
            put(s+'_big_toe',[sign*.25+.25,0]);put(s+'_small_toe',[sign*.25+.2,0])
        return x,t

    def test_missing_and_view_gates(self):
        x,t=self.skeleton();c=dict(source='progait',view='front/rear',activity='walking',environment='overground',cohort='with_amputation')
        result=measure_segment(x,t,c,2)
        m=result['metrics'][4]
        self.assertEqual(m['measurement_status'],'not_evaluable');self.assertIsNone(m['value'])
        c['view']='side';x[:,NAMES.index('left_knee')]=np.nan
        result=measure_segment(x,t,c,2)
        self.assertIsNone(stats(result['series']['left_unsigned_projected_knee_bend_deg']))
        self.assertNotIn('left_unsigned_projected_knee_bend_deg',result['metrics'][4]['observations'])

    def test_scale_translation_and_phase_abstention(self):
        x,t=self.skeleton();c=dict(source='progait',view='side',activity='walking',environment='overground',cohort='with_amputation')
        a=measure_segment(x,t,c,2);b=measure_segment(x*3+17,t,c,6)
        for key in a['series']:np.testing.assert_allclose(a['series'][key],b['series'][key],atol=1e-10,equal_nan=True)
        # A heel rise, including terminal stance, cannot become a vaulting diagnosis without validated phases.
        x[40:,NAMES.index('left_heel'),1]=.5
        r=measure_segment(x,t,c,2)
        self.assertIsNone(r['metrics'][6]['value']);self.assertEqual(r['metrics'][6]['detection_status'],'not_evaluable')
        c['view']='front/rear';r=measure_segment(x,t,c,2)
        self.assertIsNone(r['metrics'][0]['value'])
        self.assertIn('gait_phase_unvalidated',r['metrics'][0]['reason_codes'])

    def test_filter_preserves_gaps_and_missing_is_not_zero(self):
        x,t=self.skeleton();x[20:30,0]=np.nan
        y=filtered(x,t)
        self.assertTrue(np.isnan(y[20:30,0]).all())
        self.assertIsNone(stats(np.full(30,np.nan)))


if __name__=='__main__':unittest.main()
