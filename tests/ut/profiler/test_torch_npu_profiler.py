import subprocess
import tempfile
from unittest.mock import MagicMock, patch

from vllm.config import ProfilerConfig

from tests.ut.base import TestBase


class TestTorchNPUProfilerWrapper(TestBase):
    def test_init_creates_underlying_profiler(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler) as mock_create:
            wrapper = TorchNPUProfilerWrapper(profiler_config, "dp0_pp0_tp0_dcp0_ep0_rank0")

        mock_create.assert_called_once_with(profiler_config, "dp0_pp0_tp0_dcp0_ep0_rank0")
        self.assertIs(wrapper.profiler, mock_profiler)

    def test_start_stop_delegate_to_underlying_profiler(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")

        wrapper._start()
        wrapper._stop()

        mock_profiler.start.assert_called_once()
        mock_profiler.stop.assert_called_once()

    @patch("vllm_ascend.profiler.torch_npu_profiler.envs_ascend")
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    @patch("torch_npu.profiler._ExperimentalConfig")
    @patch("torch_npu.profiler.profile")
    @patch("torch_npu.profiler.schedule")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    @patch("torch_npu.profiler.ExportType")
    @patch("torch_npu.profiler.ProfilerLevel")
    @patch("torch_npu.profiler.AiCMetrics")
    @patch("torch_npu.profiler.ProfilerActivity")
    def test_create_profiler_enabled(
        self,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_schedule,
        mock_profile,
        mock_experimental_config,
        mock_get_ascend_config,
        mock_envs_ascend,
    ):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 0
        mock_get_ascend_config.side_effect = RuntimeError("Ascend config is not initialized")

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            torch_profiler_with_stack=True,
            torch_profiler_with_memory=True,
        )

        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.AiCoreNone = "AiCoreNone"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"

        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_profiler_instance = MagicMock()
        mock_profile.return_value = mock_profiler_instance

        result = TorchNPUProfilerWrapper._create_profiler(
            profiler_config,
            "warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
        )

        mock_experimental_config.assert_called_once()
        config_kwargs = mock_experimental_config.call_args.kwargs
        expected_config = {
            "export_type": "Text",
            "profiler_level": "Level1",
            "msprof_tx": False,
            "aic_metrics": "AiCoreNone",
            "l2_cache": False,
            "op_attr": False,
            "data_simplification": True,
            "record_op_args": False,
            "gc_detect_threshold": None,
        }
        for key, expected_value in expected_config.items():
            self.assertEqual(config_kwargs[key], expected_value)

        mock_trace_handler.assert_called_once_with(
            "/path/to/traces",
            worker_name="warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
        )

        mock_profile.assert_called_once()
        profile_kwargs = mock_profile.call_args.kwargs
        self.assertEqual(profile_kwargs["activities"], ["CPU", "NPU"])
        self.assertTrue(profile_kwargs["profile_memory"])
        self.assertEqual(profile_kwargs["with_modules"], True)
        self.assertNotIn("schedule", profile_kwargs)
        mock_schedule.assert_not_called()
        profile_kwargs["on_trace_ready"](MagicMock())
        mock_trace_handler_instance.assert_called_once()
        self.assertEqual(result, mock_profiler_instance)

    @patch("vllm_ascend.profiler.torch_npu_profiler.envs_ascend")
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    @patch("torch_npu.profiler._ExperimentalConfig")
    @patch("torch_npu.profiler.profile")
    @patch("torch_npu.profiler.schedule")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    @patch("torch_npu.profiler.ExportType")
    @patch("torch_npu.profiler.ProfilerLevel")
    @patch("torch_npu.profiler.AiCMetrics")
    @patch("torch_npu.profiler.ProfilerActivity")
    def test_create_profiler_uses_schedule_when_max_iterations_set(
        self,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_schedule,
        mock_profile,
        mock_experimental_config,
        mock_get_ascend_config,
        mock_envs_ascend,
    ):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 0
        mock_get_ascend_config.side_effect = RuntimeError("Ascend config is not initialized")
        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.AiCoreNone = "AiCoreNone"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"
        mock_trace_handler.return_value = MagicMock()
        mock_schedule.return_value = "schedule"
        mock_profile.return_value = MagicMock()

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            max_iterations=2,
        )

        TorchNPUProfilerWrapper._create_profiler(profiler_config, "trace_name")

        mock_schedule.assert_called_once_with(
            wait=0,
            warmup=0,
            active=2,
            repeat=1,
            skip_first=0,
        )
        self.assertEqual(mock_profile.call_args.kwargs["schedule"], "schedule")

    @patch("vllm_ascend.profiler.torch_npu_profiler.multiprocessing.current_process")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    def test_trace_handler_starts_offline_analyse_in_daemon_process(self, mock_trace_handler, mock_current_process):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_current_process.return_value.daemon = True

        with patch.object(TorchNPUProfilerWrapper, "_start_offline_analyse") as mock_start_offline_analyse:
            on_trace_ready = TorchNPUProfilerWrapper._create_trace_handler(profiler_config, "trace_name")
            on_trace_ready(MagicMock())

        mock_trace_handler.assert_called_once_with(
            "/path/to/traces",
            worker_name="trace_name",
        )
        mock_trace_handler_instance.assert_not_called()
        mock_start_offline_analyse.assert_called_once_with(profiler_config)

    @patch("vllm_ascend.profiler.torch_npu_profiler.multiprocessing.current_process")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    def test_trace_handler_parses_in_non_daemon_process(self, mock_trace_handler, mock_current_process):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_current_process.return_value.daemon = False
        mock_profiler = MagicMock()

        on_trace_ready = TorchNPUProfilerWrapper._create_trace_handler(profiler_config, "trace_name")
        on_trace_ready(mock_profiler)

        mock_trace_handler_instance.assert_called_once_with(mock_profiler)

    @patch("vllm_ascend.profiler.torch_npu_profiler.subprocess.Popen")
    def test_start_offline_analyse_runs_in_new_process(self, mock_popen):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_popen.return_value.pid = 1234

        with tempfile.TemporaryDirectory() as trace_dir:
            profiler_config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=trace_dir,
            )

            TorchNPUProfilerWrapper._start_offline_analyse(profiler_config)

        command = mock_popen.call_args.args[0]
        kwargs = mock_popen.call_args.kwargs
        self.assertIn("from torch_npu.profiler.profiler import analyse", command[2])
        self.assertEqual(command[-1], trace_dir)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["start_new_session"])

    def test_create_profiler_disabled(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(profiler=None, torch_profiler_dir="")

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn("Unrecognized profiler: None", str(cm.exception))

    def test_create_profiler_empty_dir(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = MagicMock()
        profiler_config.profiler = "torch"
        profiler_config.torch_profiler_dir = ""

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn("torch_profiler_dir cannot be empty", str(cm.exception))

    @patch("vllm_ascend.profiler.torch_npu_profiler.envs_ascend")
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_raises_when_msmonitor_env_enabled(self, mock_get_ascend_config, mock_envs_ascend):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 1
        mock_get_ascend_config.side_effect = RuntimeError("Ascend config is not initialized")
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn(
            "MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.",
            str(cm.exception),
        )

    @patch("vllm_ascend.profiler.torch_npu_profiler.envs_ascend")
    @patch("torch_npu.profiler._ExperimentalConfig")
    @patch("torch_npu.profiler.profile")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    @patch("torch_npu.profiler.ExportType")
    @patch("torch_npu.profiler.ProfilerLevel")
    @patch("torch_npu.profiler.AiCMetrics")
    @patch("torch_npu.profiler.ProfilerActivity")
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_config_overrides_msmonitor_env(
        self,
        mock_get_ascend_config,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_profile,
        mock_experimental_config,
        mock_envs_ascend,
    ):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 1
        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_get_ascend_config.return_value = mock_ascend_config

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.AiCoreNone = "AiCoreNone"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"
        mock_profile.return_value = MagicMock()

        TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        mock_profile.assert_called_once()

    @patch("vllm_ascend.profiler.torch_npu_profiler.envs_ascend")
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_config_enables_msmonitor_over_env(self, mock_get_ascend_config, mock_envs_ascend):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 0
        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = True
        mock_get_ascend_config.return_value = mock_ascend_config
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn(
            "MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.",
            str(cm.exception),
        )

    def test_profiler_step_returns_true(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")

        self.assertTrue(wrapper._profiler_step())
        mock_profiler.step.assert_called_once()

    def test_step_calls_underlying_start_after_delay_iterations(self):
        """Work matches vLLM WorkerProfiler: first N worker steps defer _start."""
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            delay_iterations=3,
            max_iterations=0,
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")

        wrapper.start()
        mock_profiler.start.assert_not_called()

        wrapper.step()
        wrapper.step()
        mock_profiler.start.assert_not_called()

        wrapper.step()
        mock_profiler.start.assert_called_once()

    def test_step_stops_underlying_profiler_after_max_iterations(self):
        """Work matches vLLM WorkerProfiler: stop when recorded steps exceed max_iterations."""
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            delay_iterations=0,
            max_iterations=1,
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")

        wrapper.start()
        mock_profiler.start.assert_called_once()
        mock_profiler.stop.assert_not_called()

        wrapper.step()
        mock_profiler.stop.assert_not_called()

        wrapper.step()
        mock_profiler.stop.assert_called_once()
