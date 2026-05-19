#!/bin/bash
# TTT v3: full Criteo 1TB dataset + canonical MLPerf LR schedule.
#
# v2 problem: constant LR=0.004 with no warmup/decay caused AUC to peak at
# 0.78 (epoch ~8%) then degrade to ~0.67 by end of epoch -- classic
# adagrad-without-warmup overshoot. v3 adds the standard MLPerf schedule:
#   linear warmup 0 -> 0.004 over 2750 steps
#   constant 0.004 for the middle
#   polynomial decay (power=2.0) 0.004 -> 0.0 from step 49315 over 27772
#   so end of decay at step 77087 (just before MAX_ITER=80000)
#
# These warmup_steps / decay_start / decay_steps values are the canonical
# MLPerf-training DLRM-DCNv2 schedule for bs=55296.
source $(dirname ${BASH_SOURCE[0]})/config_b200_1x8_rr_bs1x_auto_mi350_quick.sh

# Convergence overrides
export MAX_ITER=30000                # ~40% epoch; with proper LR decay we expect
                                     # AUC to saturate above 0.80 well before this
export EVAL_INTERVAL=500             # tighter eval cadence (60 evals over 30k iters)
export AUC_THRESHOLD=0.95            # disable auto-stop -> see full curve

# Canonical MLPerf LR schedule for bs=55296. With proper warmup the model
# typically converges to >0.80275 AUC within ~3000-6000 iters (a few percent
# of the full dataset), so DECAY_START is set EARLY (at iter 6000) so we
# see the decay phase complete inside our 30000-iter window. Decay finishes
# at iter 6000 + 18000 = 24000, leaving 6000 iters of post-decay at end_lr.
export WARMUP_STEPS=2750
export DECAY_START=6000
export DECAY_STEPS=18000
