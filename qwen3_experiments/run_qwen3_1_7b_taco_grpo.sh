#!/usr/bin/env bash
# The prepared config is derived from the pinned Polaris GRPO config.
# The persistent queue invokes this launcher only after predecessor audits pass.
set -euo pipefail
: "${TACO_GRPO_ROOT:?Set the prepared TACO GRPO run directory}"
exec python -u -W ignore -m verl.trainer.main_ppo \
    --config-path "${TACO_GRPO_ROOT}" --config-name resolved_config "$@"
