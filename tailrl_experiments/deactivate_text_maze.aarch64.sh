# Restore variables changed by activate.aarch64.sh.

_maze_deactivate_aarch64() {
    local name saved was_set entry result="" removed=0
    local -a entries=()
    IFS=: read -r -a entries <<<"${PATH}"
    for entry in "${entries[@]}"; do
        if [[ "${removed}" == 0 && "${entry}" == "${_MAZE_CUDA_BIN_ADDED:-}" ]]; then
            removed=1
            continue
        fi
        result="${result}${result:+:}${entry}"
    done
    export PATH="${result}"
    for name in ${_MAZE_ACTIVATION_VARS:-}; do
        saved="_MAZE_SAVED_${name}"
        was_set="_MAZE_WAS_SET_${name}"
        if [[ "${!was_set}" == 1 ]]; then
            printf -v "${name}" '%s' "${!saved}"
            export "${name}"
        else
            unset "${name}"
        fi
        unset "${saved}" "${was_set}"
    done
    unset _MAZE_ACTIVATION_VARS _MAZE_CUDA_BIN_ADDED
}

_maze_deactivate_aarch64
unset -f _maze_deactivate_aarch64
