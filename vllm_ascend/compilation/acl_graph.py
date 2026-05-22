# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import weak_ref_tensors

# 记录仍然存活的 ACLGraphWrapper，sleep 时不需要区分主模型、草稿模型
# 或上游 CUDA graph manager 的归属，也能统一失效所有已捕获图。
_acl_graph_wrappers = weakref.WeakSet()
# HCCL 地址日志比较长，每个 rank 的每一轮 capture 只打印一次。
# sleep 失效图缓存时会重置该标记并递增 generation。
_hccl_group_addresses_logged_for_capture = False
_hccl_group_address_log_generation = 0


def _collect_hccl_group_debug_info() -> list[str]:
    """返回 vLLM 已注册 HCCL ProcessGroup 的本进程地址信息。"""
    # 从 vLLM 的 parallel_state 中读取 group 注册表，避免依赖
    # vllm_ascend.patch.worker.patch_distributed 的具体导出版本。
    from vllm.distributed import parallel_state

    # _groups 中保存的是 unique_name -> weakref(GroupCoordinator)。
    groups = getattr(parallel_state, "_groups", {})
    # debug_info 保存本 rank 内可用于对比的地址字符串。
    debug_info: list[str] = []
    # 同一个 GroupCoordinator 可能被多个引用路径命中，seen 用于去重。
    seen: set[int] = set()
    for group_ref in list(groups.values()):
        # 注册表存的是 weakref，先取出真实 GroupCoordinator 对象。
        group = group_ref()
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))

        # device_group 对应 HCCL ProcessGroup；被 sleep 销毁后这里会是 None。
        device_group = getattr(group, "device_group", None)
        if device_group is None:
            continue
        # device_communicator 是基于 device_group 构造的 NPU 通信封装。
        device_communicator = getattr(group, "device_communicator", None)
        # Python id/repr 只能作为本进程内的地址指纹；不同 rank 属于不同
        # 进程地址空间，不能跨 rank 比较，只能比较同一 rank 的 sleep 前后。
        debug_info.append(
            f"{getattr(group, 'unique_name', '<unknown>')}"
            f"(group_name={getattr(group, 'group_name', '<unknown>')}, "
            f"device_group_id=0x{id(device_group):x}, "
            f"device_group={device_group!r}, "
            f"device_communicator_id="
            f"{'None' if device_communicator is None else hex(id(device_communicator))})"
        )
    return debug_info


def _get_hccl_group_log_rank_info() -> tuple[str, str]:
    """尽力获取 rank 信息，方便关联同一 rank 的地址日志。"""
    # 分布式未初始化时不能调用 get_rank，保留字符串用于日志说明。
    rank = "uninitialized"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())

    # local_rank 不在 torch ProcessGroup 里，只能从 vLLM world group 上取。
    local_rank = "unknown"
    try:
        from vllm.distributed import parallel_state

        # 遍历已注册 group，找到 world group 的 local_rank。
        groups = getattr(parallel_state, "_groups", {})
        for group_ref in list(groups.values()):
            group = group_ref()
            if group is None or getattr(group, "group_name", None) != "world":
                continue
            local_rank = str(getattr(group, "local_rank", "unknown"))
            break
    except Exception:
        pass
    return rank, local_rank


def reset_hccl_group_address_log_for_aclgraph() -> None:
    """开启新一轮 capture generation，用于控制 HCCL 地址日志只打印一次。"""
    global _hccl_group_addresses_logged_for_capture
    global _hccl_group_address_log_generation
    # 允许下一次 ACL graph capture 重新打印一条 HCCL 地址日志。
    _hccl_group_addresses_logged_for_capture = False
    # generation 用于区分 sleep 前和 wakeup 后的两轮 capture。
    _hccl_group_address_log_generation += 1


def log_hccl_group_addresses_for_aclgraph(reason: str, *, log_once: bool = True) -> list[str]:
    """打印并返回 ACL graph 捕获时观测到的 HCCL 地址快照。"""
    global _hccl_group_addresses_logged_for_capture

    try:
        # 每次都采集快照，保证 ACLGraphEntry 可以保存自己的地址信息。
        debug_info = _collect_hccl_group_debug_info()
    except Exception as exc:
        # 地址日志只是诊断信息，采集失败不能影响图捕获主流程。
        logger.warning("Failed to collect HCCL group addresses for ACL graph capture (%s): %s", reason, exc)
        return []

    # 日志只打印一次，但返回值仍给每个 ACLGraphEntry 使用。
    if log_once and _hccl_group_addresses_logged_for_capture:
        return debug_info

    # 标记本轮 generation 已经打印过，避免 PIECEWISE 多子图刷屏。
    _hccl_group_addresses_logged_for_capture = True
    # rank/local_rank 放进日志，方便只比较同一 rank 的 sleep 前后地址。
    rank, local_rank = _get_hccl_group_log_rank_info()
    log_context = (
        f"generation={_hccl_group_address_log_generation}, rank={rank}, "
        f"local_rank={local_rank}, {reason}"
    )
    # 无 active HCCL group 时也打印一次，说明本轮 capture 没采集到地址。
    if not debug_info:
        logger.info("ACL graph capture recorded HCCL group addresses (%s): no active HCCL device groups.", log_context)
        return []
    # 打印本轮 capture 记录到的所有 active HCCL group 地址。
    logger.info("ACL graph capture recorded HCCL group addresses (%s): %s", log_context, "; ".join(debug_info))
    return debug_info


def _tensor_nbytes(tensor: Any, seen_storages: set[tuple[Any, Any]]) -> int:
    """统计底层 storage 大小；多个 tensor 共享 storage 时只计一次。"""
    if not isinstance(tensor, torch.Tensor):
        return 0
    try:
        # untyped_storage 能得到真实底层存储，避免按 view 的 numel 重复统计。
        storage = tensor.untyped_storage()
        # device + data_ptr 组成 storage 唯一键，用于去重。
        storage_key = (tensor.device, storage.data_ptr())
        if storage_key in seen_storages:
            return 0
        seen_storages.add(storage_key)
        return storage.nbytes()
    except Exception:
        # 某些 weak/ref tensor 可能没有完整 storage API，退化为形状估算。
        return tensor.numel() * tensor.element_size()


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None

    # 该图捕获时观测到的 HCCL device group 地址快照；
    # 用于对比 sleep 前和 wakeup 后的捕获是否使用了同一通信域对象。
    hccl_group_addresses: list[str] | None = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle
        # 维护进程内弱引用注册表，供 sleep 统一清理 ACL graph 缓存。
        _acl_graph_wrappers.add(self)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self._runnable_str}"
            )
        raise AttributeError(f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for more details.")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def reset_aclgraph_cache(self) -> int:
        # 返回清理前条目数，便于 sleep 统计清理了多少张图。
        num_entries = len(self.concrete_aclgraph_entries)
        # 删除 BatchDescriptor -> ACLGraphEntry 映射，释放旧图对象引用。
        self.concrete_aclgraph_entries.clear()
        # 下一次调用需要重新完成 first-run/capture 逻辑。
        self.first_run_finished = False
        # graph pool 也重新从平台获取，避免复用旧捕获池状态。
        self.graph_pool = current_platform.get_global_graph_pool()
        return num_entries

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()
            # 将捕获时的 HCCL 地址快照保存到图条目中，便于同一 rank
            # 对比 sleep 前和 wakeup 后是否换了通信域对象。
            entry.hccl_group_addresses = log_hccl_group_addresses_for_aclgraph(
                f"wrapper mode={self.runtime_mode.name}, batch={entry.batch_descriptor}, inputs={input_addresses}"
            )

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            global _draft_graph_params
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        # If we do not in main model and in full-graph mode when using merge-eagle-graph,
        # we do not need to synchronize.
        # When enable_enpu is on, model_runner orders update vs replay; skip here.
        # When EAGLE draft (merge path), replay does not need this barrier.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        if not self.enable_enpu and not is_draft_eagle:
            torch.npu.current_stream().synchronize()
        entry.aclgraph.replay()
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None,
):
    impl_cls = attn_backend.get_impl_cls()
    impl_cls.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        num_dcp_pcp_tokens,
        draft_attn_metadatas,
    )


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: GraphParams | None = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


def _reset_graph_params(params: GraphParams | None) -> None:
    """清理指向旧 capture 状态的 graph task 元数据。"""
    if params is None:
        return
    # events/handles/attn_params 都和旧 ACL graph capture 绑定，必须清空。
    for num_tokens in params.events:
        params.events[num_tokens] = []
    for num_tokens in params.handles:
        params.handles[num_tokens] = []
    for num_tokens in params.attn_params:
        params.attn_params[num_tokens] = []


def _reset_attention_workspaces(params: GraphParams | None, seen_storages: set[tuple[Any, Any]]) -> int:
    """清理 attention workspace，并返回已跟踪的 storage 字节数。"""
    if params is None:
        return 0
    # workspace_bytes 是按 tensor storage 统计的理论释放量。
    workspace_bytes = 0
    for num_tokens, workspace in params.workspaces.items():
        # 同一个 workspace 可能被 graph/draft graph 共享，使用 seen_storages 去重。
        workspace_bytes += _tensor_nbytes(workspace, seen_storages)
        # 置空后下一次 capture/update 会重新申请 workspace。
        params.workspaces[num_tokens] = None
    return workspace_bytes


def reset_attention_workspaces_for_sleep() -> int:
    """进入 sleep 前释放所有缓存的 attention workspace。"""
    # 跨 graph 和 draft graph 共用一个 seen 集合，避免重复统计共享 storage。
    seen_storages: set[tuple[Any, Any]] = set()
    return _reset_attention_workspaces(_graph_params, seen_storages) + _reset_attention_workspaces(
        _draft_graph_params, seen_storages
    )


def reset_graph_params_for_sleep() -> None:
    """失效 ACL graph 元数据；workspace 清理和统计由单独函数处理。"""
    _reset_graph_params(_graph_params)
    _reset_graph_params(_draft_graph_params)


def reset_aclgraph_caches_for_sleep() -> int:
    """清理已捕获 ACL graph，并允许下一轮 capture 再打印一次 HCCL 地址。"""
    reset_hccl_group_address_log_for_aclgraph()
    return sum(wrapper.reset_aclgraph_cache() for wrapper in list(_acl_graph_wrappers))


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params
