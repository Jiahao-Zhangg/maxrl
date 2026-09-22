# Conda activation hook for the independent ARM/GH200 environment.
# Install as $CONDA_PREFIX/etc/conda/activate.d/tailrl_rb.sh.

_maze_filter_cuda_paths() {
    local root=$1 value=$2 entry result=""
    local -a entries=()
    IFS=: read -r -a entries <<<"${value}"
    for entry in "${entries[@]}"; do
        [[ -n "${entry}" ]] || continue
        if [[ "${entry}" != "${root}"* ]] &&
            [[ "${entry}" == */cuda/* || "${entry}" == */cuda-*/* || "${entry}" == */math_libs/* ]]; then
            continue
        fi
        result="${result}${result:+:}${entry}"
    done
    printf '%s' "${result}"
}

_maze_activate_aarch64() {
    local cuda_root=${MAZE_CUDA_HOME:-/sw/user/cudatoolkits/installs/cuda-12.6.1}
    local host_cc=${MAZE_HOST_CC:-/opt/cray/pe/gcc-native/13/bin/gcc}
    local host_cxx=${MAZE_HOST_CXX:-/opt/cray/pe/gcc-native/13/bin/g++}
    local compiler_version name saved was_set filtered cache_root
    [[ -n "${CONDA_PREFIX:-}" ]] || { echo "Activate the Conda environment first." >&2; return 1; }
    [[ -x "${cuda_root}/bin/nvcc" && -x "${host_cc}" && -x "${host_cxx}" ]] || {
        echo "CUDA 12.6 and GCC 13 are required; set MAZE_CUDA_HOME / MAZE_HOST_CC / MAZE_HOST_CXX if relocated." >&2
        return 1
    }
    compiler_version=$("${cuda_root}/bin/nvcc" --version)
    [[ "${compiler_version}" == *"release 12.6"* ]] || {
        echo "This ARM environment requires a CUDA 12.6 toolkit." >&2
        return 1
    }

    export _MAZE_ACTIVATION_VARS="CUDA_HOME CUDA_PATH CUDACXX CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH PYTHONNOUSERSITE TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR HF_HOME"
    for name in ${_MAZE_ACTIVATION_VARS}; do
        saved="_MAZE_SAVED_${name}"
        was_set="_MAZE_WAS_SET_${name}"
        if [[ -v "${name}" ]]; then
            printf -v "${saved}" '%s' "${!name}"
            printf -v "${was_set}" '%s' 1
        else
            printf -v "${saved}" '%s' ""
            printf -v "${was_set}" '%s' 0
        fi
        export "${saved}" "${was_set}"
    done

    export CUDA_HOME="${cuda_root}" CUDA_PATH="${cuda_root}" CUDACXX="${cuda_root}/bin/nvcc"
    export CC="${host_cc}" CXX="${host_cxx}"
    # Only prepend PATH; deactivation removes this one entry after Conda updates its own PATH.
    export _MAZE_CUDA_BIN_ADDED="${cuda_root}/bin"
    export PATH="${_MAZE_CUDA_BIN_ADDED}:${PATH}"
    filtered=$(_maze_filter_cuda_paths "${cuda_root}" "${CPATH:-}")
    export CPATH="${cuda_root}/include${filtered:+:${filtered}}"
    filtered=$(_maze_filter_cuda_paths "${cuda_root}" "${LIBRARY_PATH:-}")
    export LIBRARY_PATH="${cuda_root}/lib64${filtered:+:${filtered}}"
    filtered=$(_maze_filter_cuda_paths "${cuda_root}" "${LD_LIBRARY_PATH:-}")
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${cuda_root}/lib64${filtered:+:${filtered}}"
    export PYTHONNOUSERSITE=1 TORCH_CUDA_ARCH_LIST=9.0
    cache_root=${XDG_CACHE_HOME:-${HOME}/.cache}
    export TORCH_EXTENSIONS_DIR="${cache_root}/tailrl_rb/torch_extensions/py310-cu126"
    export HF_HOME=${HF_HOME:-${cache_root}/huggingface}
}

_maze_activate_aarch64
unset -f _maze_activate_aarch64 _maze_filter_cuda_paths
