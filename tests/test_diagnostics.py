import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics import common, stability_sweep, summarize_tuning, tune_hyperparameters
from solver import OptimalControlSolver
from solver_settings import DEFAULT_PROFILE, PROFILE_PATH, load_profile
import run_continuation


def case(name='case', profile='test', **options):
    return dict(id=name,geometry='test',profile=dict(id=profile,corrector={}),
                phase='full',seed='auto',snap_phase=True,**{'lambda':.23,'kappa':.83},**options)


def completed_continuation(s,**kwargs):
    y=np.zeros((12,2))
    y[:3,0],y[3:6,0]=s.r0,s.v0
    y[:3,-1],y[3:6,-1]=s.rf,s.vf
    tr=SimpleNamespace(success=True,t=np.array([0.,s.T]),y=y)
    s.step_log=[(1.,.05,True)]
    s.continuation_status=dict(state='completed',target_alpha=1.,reached_alpha=1.)
    return [dict(alpha=1.,a_max=.2,p0=np.zeros(6),trajectory=tr)]


class ConfigurationTests(unittest.TestCase):
    def test_profile_reference_and_instance_settings(self):
        alias=PROFILE_PATH.parent/'diagnostics/tuned_corrector_profile.json'
        self.assertEqual(load_profile(alias),load_profile(PROFILE_PATH))
        s=OptimalControlSolver(1.,.01,0.,'mixed')
        s.corrector_options['diff_step']=2e-5
        self.assertEqual(s.corrector_configuration()['profile'],'custom')
        self.assertEqual(s.corrector_configuration()['diff_step'],2e-5)
        self.assertEqual(OptimalControlSolver(1.,.01,0.,'mixed').corrector_options,
                         DEFAULT_PROFILE['corrector'])
        plan=common.read_plan(PROFILE_PATH.parent/'diagnostics/tuned_cases_plan.json')
        self.assertTrue(all(c['profile']==DEFAULT_PROFILE for c in plan))

    def test_runner_records_actual_settings(self):
        original=OptimalControlSolver.__init__
        def initialize(s,*args,**kwargs):
            original(s,*args,**kwargs)
            s.corrector_options['diff_step']=2e-5
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
                OptimalControlSolver,'__init__',initialize), patch.object(
                OptimalControlSolver,'solve',return_value=SimpleNamespace(success=False)):
            folder=Path(tmp)/'run'
            run_continuation.run(folder)
            record=common.read_json(folder/'summary.json')
        self.assertEqual(record['corrector']['profile'],'custom')
        self.assertEqual(record['corrector']['diff_step'],2e-5)
        self.assertEqual(record['model']['a_max_constructor'],.01)
        self.assertNotIn('initial_a_max',record['model'])

    def test_custom_settings_survive_save_load(self):
        s=common.make_solver({'lambda':.5,'kappa':1.5,'snap_phase':True})
        s.p0=np.zeros(6)
        s.corrector_options['diff_step']=2e-5
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                OptimalControlSolver,'integrate_trajectory',return_value=SimpleNamespace(
                    success=True,t=np.array([s.T]),y=np.zeros((12,1)))):
            path=Path(tmp)/'solution.pkl'
            s.save_solution(path)
            loaded,_=OptimalControlSolver.load_and_integrate(path)
        self.assertEqual(loaded.corrector_options,s.corrector_options)
        self.assertEqual(loaded.initial_corrector_options,s.initial_corrector_options)


class DiagnosticFailureTests(unittest.TestCase):
    def test_verification_exception_is_a_final_error_with_saved_solutions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                OptimalControlSolver,'continuation_method',completed_continuation), patch(
                'diagnostics.stability_sweep.verify_solutions',side_effect=RuntimeError('injected')):
            folder=Path(tmp)
            result=stability_sweep.worker(case(),folder,verify_all=True)
            self.assertEqual(result,common.read_json(folder/'result.json'))
            self.assertTrue((folder/'solutions.npz').exists())
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['solve_status'],'completed')
        self.assertEqual(result['stage'],'verification')
        self.assertEqual(result['verification_status'],'error')
        self.assertEqual(result['solutions_count'],1)

    def test_failed_forward_check_is_not_completed(self):
        checks=[dict(alpha=1.,success=False,residual=float('inf'),points=1)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                OptimalControlSolver,'continuation_method',completed_continuation), patch(
                'diagnostics.stability_sweep.verify_solutions',return_value=checks):
            folder=Path(tmp)
            result=stability_sweep.worker(case(),folder,verify_all=True)
            saved=common.read_json(folder/'result.json')
        self.assertEqual(result['status'],'verification_failed')
        self.assertEqual(saved['verification_status'],'failed')
        self.assertIsNone(saved['maximum_independent_residual'])

    def test_budget_before_first_solution_has_a_complete_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case.json'
            common.write_json(path,case(evaluation_budget=0))
            result=tune_hyperparameters.worker(path)
            self.assertEqual(result,common.read_json(path.parent/'result.json'))
        self.assertEqual(result['status'],'evaluation_budget')
        self.assertEqual(result['tuning_evaluations'],0)
        self.assertIsNone(result['last_accepted'])
        self.assertEqual(result['verification_status'],'not_run')

    def test_process_failure_overwrites_a_premature_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            script=folder/'crash.py'
            script.write_text(
                "import json,sys\nfrom pathlib import Path\n"
                "p=Path(sys.argv[2])\nc=json.loads(p.read_text())\n"
                "(p.parent/'result.json').write_text(json.dumps(dict(case=c,status='completed',"
                "verification_status='passed')))\nraise SystemExit(7)\n",encoding='utf-8')
            result=common.run_case(case(),folder,script)
            self.assertEqual(result,common.read_json(folder/'case/result.json'))
        self.assertEqual(result['status'],'process_error')
        self.assertEqual(result['exit_code'],7)
        self.assertEqual(result['verification_status'],'interrupted')

    def test_zero_exit_without_final_result_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            script=folder/'empty.py'
            script.write_text('pass\n',encoding='utf-8')
            result=common.run_case(case(),folder,script)
        self.assertEqual(result['status'],'process_error')

    def test_zero_exit_without_verified_completion_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            script=folder/'premature.py'
            script.write_text("import json,sys\nfrom pathlib import Path\n"
                "p=Path(sys.argv[2]).parent/'result.json'\nr=json.loads(p.read_text())\n"
                "r['status']='completed'\np.write_text(json.dumps(r))\n",encoding='utf-8')
            result=common.run_case(case(),folder,script)
        self.assertEqual(result['status'],'process_error')


class SummaryTests(unittest.TestCase):
    def test_incomplete_records_and_paired_tasks(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            definitions=[('a','left',10,.01,'completed','passed'),
                         ('repeat','left',30,.01,'completed','passed'),
                         ('b','right',20,.01,'completed','passed'),
                         ('strict','left',900,1e-5,'completed','passed'),
                         ('crashed','error_profile',None,.01,'process_error','not_run'),
                         ('premature','unverified_profile',None,.01,'completed','not_run')]
            for name,profile,evaluations,tol,status,verification in definitions:
                folder=root/'input'/name
                folder.mkdir(parents=True)
                common.write_json(folder/'result.json',dict(case=case(name,profile,tol=tol),
                    status=status,verification_status=verification,evaluations=evaluations,
                    last_accepted=None))
            with patch('sys.argv',['summary',str(root/'input'),'--output',str(root/'report')]):
                summarize_tuning.main()
            result=common.read_json(root/'report/tables.json')
        self.assertEqual(next(r for r in result['trials'] if r['id']=='premature')['status'],'unverified')
        pair=next(r for r in result['paired_completed_comparisons']
                  if r['left']=='left' and r['right']=='right')
        self.assertEqual(pair['paired_completed_tasks'],1)
        self.assertEqual(pair['left_evaluations'],20)
        self.assertEqual(pair['right_evaluations'],20)


if __name__=='__main__':
    unittest.main()
