plot specific method figures and tables
/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
--data-dir outputs/occt_comparision/action_smooth_comparision \
--representative-roads 0 \
--plot-method-fill-stype false

/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
--data-dir outputs/occt_comparision/full_observation_comparison \
--representative-roads 0 \
--plot-method-fill-stype false

join exist tables and compare:
/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
--data-dir outputs/occt_comparision/action_smooth_comparision \
--compare-methods pid mppi marl_mlp_switch_obse_his5 marl_mlp_full_obse_his5 marl_mlp_continuous_w_cp_his10 marl_lipsnet_continuous_his10 marl_lipsnet_phase_blending_his10 marl_mlp_continuous_w_more_cp_his10 marl_mlp_continuous_s_curve_reward_his10

/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
--data-dir outputs/occt_comparision/full_observation_comparison \
--compare-methods pid mppi marl_mlp_full_obse marl_mlp_full_obse_no_hinge_status

/home/yons/Graduation/pyquaticus/env-full/bin/python \
/home/yons/Graduation/rl_occt/sota-implementations/multiagent/occt_metrics_evaluation.py \
--data-dir outputs/occt_comparision/action_smooth_comparision \
--compare-methods pid mppi marl_mlp_switch_obse_his5 marl_mlp_full_obse_his5 marl_mlp_continuous_w_cp_his10 marl_lipsnet_continuous_his10 marl_lipsnet_phase_blending_his10 marl_mlp_continuous_w_more_cp_his10 marl_mlp_continuous_s_curve_reward_his10