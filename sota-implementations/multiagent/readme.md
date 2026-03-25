baseline mlp:
/home/yons/Graduation/pyquaticus/env-full/bin/python /home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py /home/yons/Graduation/rl_occt/outputs/2026-03-25/13-23-39/run-20260325_132346-iqskyxx5o5cogro6e1053/rollouts/rollout_iter_0_frames_0_paths_0_5.pt /home/yons/Graduation/VMAS_occt/vmas/scenarios/occt_scenario_test_result --report-dir /home/yons/Graduation/rl_occt/occt_metrics_reports --plot-dir /home/yons/Graduation/rl_occt/occt_metrics_plots --plot-transition-comparison --comparison-road-id 0 --comparison-agent-id 1 --comparison-methods marl pid mppi

adaptive weight mlp:
/home/yons/Graduation/pyquaticus/env-full/bin/python /home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py /home/yons/Graduation/rl_occt/outputs/2026-03-25/13-49-11_adaptive_mlp_eval/run-20260325_134914-kuzsppocdp8pnogh65eqf/rollouts/rollout_iter_0_frames_0_paths_0_5.pt /home/yons/Graduation/VMAS_occt/vmas/scenarios/occt_scenario_test_result --report-dir /home/yons/Graduation/rl_occt/occt_metrics_reports --plot-dir /home/yons/Graduation/rl_occt/occt_metrics_plots --plot-transition-comparison --comparison-road-id 0 --comparison-agent-id 1 --comparison-methods marl pid mppi

plot curves for specific method and road:
/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
/home/yons/Graduation/rl_occt/outputs/2026-03-25/13-23-39_baseline_mlp_eval/run-20260325_132346-iqskyxx5o5cogro6e1053/rollouts/rollout_iter_0_frames_0_paths_0_5.pt \
--plot-method marl \
--representative-roads 0

plot comparision:
/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
/home/yons/Graduation/rl_occt/outputs/2026-03-25/13-23-39_baseline_mlp_eval/run-20260325_132346-iqskyxx5o5cogro6e1053/rollouts/rollout_iter_0_frames_0_paths_0_5.pt \
/home/yons/Graduation/VMAS_occt/vmas/scenarios/occt_scenario_test_result \
--plot-overall-bars

/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
/home/yons/Graduation/rl_occt/outputs/2026-03-25/22-03-36_full_obse_mlp_eval/run-20260325_220339-on4kowagb39yhqlat8onv/rollouts/rollout_iter_0_frames_0_paths_0_5.pt \
--plot-method marl \
--representative-roads 0
