# AGENTS.md

## 1. Overview

HCCL is the open-source collective communication layer in CANN for Ascend clusters. This repository implements public HCCL APIs, internal operator orchestration, topology-aware algorithm selection, and packaging logic for host and device-side communication components.

## 2. Folder Structure

- `include`: public headers exported to consumers of the HCCL library.
  - `hccl.h`: primary public API surface for collective and point-to-point communication.
  - `hccl_mc2.h`: additional public entry points for MC2-related capabilities.
- `src`: production source code.
  - `common`: shared infrastructure used across operators.
    - `hcomm_dlsym`: dynamic symbol wrappers that isolate optional or versioned `hcomm` dependencies behind local adapter functions.
    - other files here provide logging, parameter validation, environment/config parsing, utility helpers, error reporting, and shared types.
  - `ops`: operator implementations and the shared orchestration framework.
    - `op_common`: cross-operator runtime pieces.
      - `selector`: registry-driven algorithm selection based on `OpParam`, topology, backend mode, and device traits.
      - `executor`: execution base classes, channel/resource handling, and executor registries.
      - `template`: reusable algorithm-template bases plus backend-specific template implementations and registries.
      - `topo`: topology discovery and topology-shape matching utilities consumed by selectors and executors.
    - `all_gather`, `all_gather_v`, `all_reduce`, `all_to_all_v`, `batch_send_recv`, `broadcast`, `recv`, `reduce`, `reduce_scatter`, `reduce_scatter_v`, `scatter`, `send`: per-operator modules. Most follow the same split of API entrypoint, selector, executor, and backend-specific templates.
    - `interface_graph_mode`: graph-mode integration points that complement the default op-based execution path.
- `cmake`: shared CMake modules for configuration, packaging, third-party setup, helper functions, and signing/package assembly flows.
- `scripts`: release, signing, packaging, and version-update utilities used by build and packaging flows.
  - `custom`, `package`, `sign`, `signtool`, `update_version_info`: specialized scripts grouped by delivery concern.
- `docs`: user-facing documentation, API references, figures, and build guidance.
- `examples`: runnable examples grouped by communication scenario and extension style.
- `test`: validation code.
  - `ut`: unit-test code and test environment preparation assets.
  - `st`: system/integration-style algorithm tests and supporting scripts.
- `build.sh`, `build_third_party.sh`: top-level build entry scripts for normal builds, tests, packaging, and third-party setup.
- `CMakeLists.txt`, `version.cmake`: root build and version orchestration.

## 3. Core Behaviors & Patterns

- Operator entrypoints are thin orchestration layers. A typical flow is: validate pointers/counts/tags, obtain communicator and device context, populate `OpParam`, run selector logic, optionally fall back to an inner/legacy implementation, then dispatch through a shared execution path such as `HcclExecOp`.
- Compatibility and fallback are first-class. Operators commonly short-circuit to older inner implementations when AICPU/CCU/AIV paths are unavailable, when the `hcomm` version is too old, or when the detected device type cannot support the newer execution path.
- Algorithm choice is topology-driven rather than hardcoded per API. Selectors inspect `TopoInfoWithNetLayerDetails`, rank layout, backend capability, data size, and shape markers like `MESH_1D`, `CLOS`, or multi-level topologies to produce a concrete algorithm name.
- Registration is static and distributed. Selectors and templates register themselves via macros at load time, and shared registries resolve them later by op type, priority, or template type. Adding a new algorithm usually means wiring a new selector/template/executor into the existing registry system rather than editing a central switch.
- Backend specialization is layered. Shared bases live under `src/ops/op_common`, while backend-specific implementations live under `template/aicpu`, `template/aiv`, `template/ccu`, or `template/dpu`. The selected algorithm name decides which specialized path runs.
- Resource restoration and execution are separated. Executors reconstruct channel/resource maps from serialized context, then run kernel orchestration loops using template objects and per-loop slices rather than embedding transport details directly in the API layer.
- Graph mode extends, rather than replaces, the standard flow. Graph-mode entrypoints still perform the same validation and setup work, but additionally assemble a `ResPackGraphMode` with tag, stream list, and scratch-memory metadata before calling the common execution routine.
- Error propagation is macro-based and intentionally shallow. `CHK_RET`, `CHK_PRT_RET`, `CHK_PTR_NULL`, and related macros log at the failure site and return immediately, so most call chains are written as linear sequences of guarded steps instead of nested recovery branches.
- Input validation also reports structured diagnostics. Many front-door checks pair local guards with `RPT_INPUT_ERR` so invalid user inputs are both rejected locally and surfaced through the external error-reporting channel.
- Shared infrastructure absorbs external-library drift at the boundary. The `hcomm_dlsym` layer wraps optional symbols and unsupported calls so most of `src/ops` can code against stable local adapters instead of scattering version checks everywhere.

## 4. Conventions

- Namespace and type naming follow C++ project conventions: namespaces use lowercase (`ops_hccl`), classes use `PascalCase`, enums/constants/macros use `UPPER_SNAKE_CASE`, and many local aliases use short integral names such as `u32` and `u64`.
- Operator modules are named by communication primitive, and file names usually mirror the exported class or responsibility exactly, for example `all_gather_op.cc`, `all_gather_auto_selector.cc`, or `ins_v2_all_gather_parallel_executor.cc`.
- Public API functions use `Hccl*` prefixes and return `HcclResult`. Helper functions inside a primitive keep the primitive name in the symbol, such as `AllGatherInitAndCheck` or `AllGatherOutPlaceCommon`, which makes call chains easy to follow during debugging.
- Logging and guard style are standardized. Use `HCCL_INFO`, `HCCL_WARNING`, `HCCL_ERROR`, and the `CHK_*` macros instead of ad hoc `if`/`return` logging blocks so failures carry consistent file, line, thread, and error-code context.
- Selection and registration code relies on macros rather than manual bootstrapping. New selectors should use `REGISTER_SELECTOR` or `REGISTER_SELECTOR_BY_OPTYPE`; new templates should use `REGISTER_TEMPLATE`; registry ownership stays in `op_common` instead of individual operators.
- Directory splits reflect runtime responsibilities: `selector` decides algorithm names, `executor` coordinates concrete execution/resource handling, and `template` holds reusable backend algorithm bodies. Keep new code in the layer that matches its responsibility rather than mixing policy and execution.
- Backend-specific code should stay confined to backend folders (`aicpu`, `aiv`, `ccu`, `dpu`) with shared abstractions promoted upward only when they are used by multiple backends.
- Build and packaging logic stays declarative in CMake where possible, while shell scripts orchestrate environment setup and end-to-end build/package flows. Do not move product packaging rules into arbitrary source directories.
- Public headers belong under `include`; internal helpers should remain under `src`. Avoid leaking internal execution or topology abstractions into the exported API surface unless the API contract explicitly changes.

## 5. Working Agreements

- Respond in the user's preferred language; if unspecified, infer it from the repository context. Keep domain terms, APIs, and code identifiers in English, and never translate fenced code blocks.
- Build context before editing: search for the same operator, selector, executor, template, or helper in sibling modules and preserve established patterns.
- Prefer small, local changes. This codebase is highly pattern-driven, so extend existing registries, helpers, and directory splits instead of introducing parallel abstractions.
- Do not add tests, lint tasks, or formatting-only changes unless the user explicitly asks for them.
- Preserve public HCCL APIs and observable behavior unless the user requests a behavior change; call out any compatibility impact explicitly.
- Keep new functions and classes single-purpose and colocated with the relevant operator/backend module.
- Avoid new external dependencies unless strictly necessary, and explain why if one is required.
- There is no standalone type-check target in this repository. After C++ changes, use compile verification through the existing build system, with `cmake -S . -B build` and `cmake --build build` as the default validation path, or the equivalent `build.sh` mode already used by the touched area.
