# -*- coding: utf-8 -*-
"""本地潮滩推理可信执行闭环单元测试（inference_agent_loop + bridge 门闩）。

覆盖用户规格 §十一 的 20 项检查点：
  1  合法输入生成计划          2  缺少输入时阻断
  3  输入不可读取时阻断         4  波段不足时阻断
  5  缺少权重时阻断             6  非法权重路径阻断
  7  CUDA 不可用时按策略处理     8  未确认不执行
  9  重复确认不重复执行        10  页面 rerun 不重复执行
 11  推理异常返回失败          12  后处理异常返回失败
 13  Final TIF 不存在时验证失败 14  Final TIF 无效时验证失败
 15  成功后登记资产            16  失败不登记
 17  地图失败只产生 warning    18  Copilot 只展示真实结果
 19  完成后能力状态刷新        20  M5/E1/PDF 原流程不受影响

原则：不加载真实模型；用 fake pre/post engine 在 tmp 目录写出真实小 GeoTIFF /
Shapefile，使 execute → verify → register 走真实磁盘产物。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import inference_agent_loop as ial  # noqa: E402
from agent_command_bridge import (  # noqa: E402
    apply_system_command,
    build_pending_task,
    init_ui_session_defaults,
    propose_inference_plan,
)

# ---------------------------------------------------------------
#  磁盘产物 helpers：写出可被 rasterio / geopandas 读取的真实小文件
# ---------------------------------------------------------------
def _make_tif(
    path,
    width=64,
    height=64,
    crs="EPSG:4326",
    transform=None,
    bands=3,
    nodata=None,
    fill=None,
):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(120.0, 30.0, 0.01, 0.01)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for b in range(1, bands + 1):
            arr = np.zeros((height, width), dtype="uint8")
            if fill is not None:
                arr[:] = fill
            else:
                arr[10:20, 10:20] = 200  # 有值区域
            dst.write(arr, b)
    return path


def _make_shp(shp_path, tif_path):
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    shp_path = Path(shp_path)
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(tif_path)) as src:
        b = src.bounds
        crs = src.crs
    gdf = gpd.GeoDataFrame({"tidal": [1]}, geometry=[box(b.left, b.bottom, b.right, b.top)], crs=crs)
    gdf.to_file(str(shp_path), driver="ESRI Shapefile")
    return shp_path


def _make_fake_weight(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weight")
    return path


def _make_task_dir(root: Path, task: str, tifs=("a.tif", "b.tif")) -> Path:
    d = root / task
    d.mkdir(parents=True, exist_ok=True)
    for t in tifs:
        _make_tif(d / t)
    return d


def _happy_plan(tmp: Path, task: str = "task_a", prob=0.05, cnt=2, device="cpu"):
    """构造 ready=True 的计划（validate 用 check_weight_load=False 跳过权重结构校验）。"""
    root = tmp / "root"
    final = tmp / "final"
    mask = tmp / "mask"
    weight = _make_fake_weight(tmp / "weights" / "cdnet.pth")
    _make_task_dir(root, task)
    final.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)
    plan = ial.build_inference_plan(
        task_id=task,
        root_dir=str(root),
        final_root=str(final),
        mask_root=str(mask),
        model_path=str(weight),
        prob_threshold=prob,
        count_threshold=cnt,
        device_policy="auto",
    )
    ok, blockers, device = ial.validate_inference_plan(plan, check_weight_load=False)
    plan["device"] = device or "cpu"
    return plan, root, final, mask, weight


# ---------------------------------------------------------------
#  Fake 引擎：只在 tmp 上写真实文件，不碰 torch / YYnet
# ---------------------------------------------------------------
class FakePreEngine:
    """复刻 pre_engine 的 load_model / process_geotiff 签名。"""

    def __init__(self, fail_load=False, fail_process=False, stop=False):
        self.fail_load = fail_load
        self.fail_process = fail_process
        self.stop = stop
        self.calls = []

    def load_model(self, weight_path, device):
        self.calls.append(("load_model", str(weight_path), device))
        if self.fail_load:
            raise RuntimeError("fake load fail")
        return object()

    def process_geotiff(self, model, tif_path, save_path, device,
                        current_idx=0, total_batch=0, stop_callback=None):
        self.calls.append(("process_geotiff", os.path.basename(tif_path), device))
        if self.stop or (stop_callback and stop_callback()):
            return False
        if self.fail_process:
            raise RuntimeError("fake process fail")
        # 写出单波段 mask（与输入同 CRS/transform）
        import numpy as np
        import rasterio

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(tif_path)) as src:
            crs, transform, w, h = src.crs, src.transform, src.width, src.height
        with rasterio.open(
            str(save_path), "w", driver="GTiff", width=w, height=h,
            count=1, dtype="uint8", crs=crs, transform=transform,
        ) as dst:
            arr = np.zeros((h, w), dtype="uint8")
            arr[10:20, 10:20] = 255
            dst.write(arr, 1)
        return True


class FakePostEngine:
    """复刻 post_engine.generate_double_constraint_complete 签名。"""

    def __init__(self, fail=False, stop=False, skip_shp=False, skip_tif=False):
        self.fail = fail
        self.stop = stop
        self.skip_shp = skip_shp
        self.skip_tif = skip_tif
        self.calls = []

    def generate_double_constraint_complete(
        self, source_folder, mask_folder, output_path, shp_path,
        prob_threshold, min_absolute_count, logger=print,
        stop_callback=None, keep_final_tif=False,
    ):
        self.calls.append({
            "source_folder": source_folder, "output_path": output_path,
            "shp_path": shp_path, "prob_threshold": prob_threshold,
            "min_absolute_count": min_absolute_count, "keep_final_tif": keep_final_tif,
        })
        if self.stop or (stop_callback and stop_callback()):
            return False
        if self.fail:
            raise RuntimeError("fake post fail")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path  # output_path 就是 final tif 全路径（keep_final_tif=True 语义）
        if not self.skip_tif:
            _make_tif(stem, bands=1, fill=255)
        if not self.skip_shp:
            _make_shp(stem.with_suffix(".shp"), stem)
        return True


# ===============================================================
#  1 / 2 / 5 / 6 / 越界 / shp_path：计划构建
# ===============================================================
class TestPlanBuild(unittest.TestCase):
    def test_cross_platform_rel_path_does_not_echo_windows_path(self):
        label = ial.rel_path(r"C:\Users\chl\private\result.tif")
        self.assertIn("result.tif", label)
        self.assertNotIn(r"C:\Users\chl", label)

    def test_01_legal_input_builds_ready_plan(self):
        with tempfile.TemporaryDirectory() as td:
            plan, root, final, mask, weight = _happy_plan(Path(td))
            self.assertTrue(plan["ready"])
            self.assertFalse(plan["blockers"])
            self.assertEqual(plan["schema"], "local_tidal_flat_inference_plan_v1")
            self.assertEqual(plan["bands"], [1, 2, 3])
            self.assertEqual(plan["model_id"], "cdnet_resnet50")
            self.assertEqual(plan["status"], "waiting_confirmation")
            self.assertEqual(plan["input_path"], os.path.normpath(str(root / "task_a")))
            self.assertEqual(plan["output_dir"], os.path.normpath(str(final / "task_a")))
            self.assertEqual(plan["mask_dir"], os.path.normpath(str(mask / "task_a")))
            self.assertEqual(plan["weight_path"], os.path.normpath(str(weight)))
            self.assertEqual(plan["available_tasks"], ["task_a"])

    def test_02_missing_task_dir_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            root.mkdir(); final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="ghost", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("目标任务目录不存在" in b for b in plan["blockers"]))

    def test_02b_missing_root_dir_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            weight = _make_fake_weight(tmp / "w.pth")
            (tmp / "final").mkdir(); (tmp / "mask").mkdir()
            plan = ial.build_inference_plan(
                task_id="t", root_dir=str(tmp / "no_root"), final_root=str(tmp / "final"),
                mask_root=str(tmp / "mask"), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("原始影像根目录不存在" in b for b in plan["blockers"]))

    def test_05_missing_weight_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            _make_task_dir(root, "task_a"); final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(tmp / "nonexistent.pth"),
                prob_threshold=0.05, count_threshold=2,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("未找到可用模型权重" in b for b in plan["blockers"]))

    def test_06_url_weight_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            _make_task_dir(root, "task_a"); final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path="https://evil.example/x.pth",
                prob_threshold=0.05, count_threshold=2,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("网络地址" in b or "未找到可用模型权重" in b for b in plan["blockers"]))

    def test_param_range_blockers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            _make_task_dir(root, "task_a"); final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.99, count_threshold=0,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("概率阈值" in b for b in plan["blockers"]))
            self.assertTrue(any("最少出现次数" in b for b in plan["blockers"]))

    def test_shp_path_handling(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, root, final, mask, weight = _happy_plan(tmp)
            # 存在 → 写入计划；不存在 → 仅告警不阻断
            coast = tmp / "coast.shp"
            coast.write_bytes(b"")
            plan2, *_ = _happy_plan(tmp)
            plan2 = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
                shp_path=str(coast),
            )
            self.assertEqual(plan2["shp_path"], os.path.normpath(str(coast)))
            plan3 = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
                shp_path=str(tmp / "ghost.shp"),
            )
            self.assertIsNone(plan3["shp_path"])
            self.assertTrue(any("海岸线裁剪矢量不存在" in w for w in plan3["warnings"]))
            self.assertTrue(plan3["ready"])


# ===============================================================
#  3 / 4 / 7：执行前验证
# ===============================================================
class TestValidate(unittest.TestCase):
    def test_03_unreadable_input_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            d = root / "task_a"
            d.mkdir(parents=True, exist_ok=True)
            _make_tif(d / "a.tif")
            (d / "0bad.tif").write_bytes(b"not a tiff at all")
            final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
            )
            ok, blockers, _ = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertFalse(ok)
            self.assertTrue(any("影像不可读取" in b for b in blockers))

    def test_04_insufficient_bands_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            d = root / "task_a"
            d.mkdir(parents=True, exist_ok=True)
            _make_tif(d / "a.tif")
            _make_tif(d / "0single.tif", bands=1)
            final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_a", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
            )
            ok, blockers, _ = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertFalse(ok)
            self.assertTrue(any("波段不足" in b for b in blockers))

    def test_07_cuda_required_without_cuda_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td), device="cpu")
            plan["device_policy"] = "cuda_required"
            with mock.patch("torch.cuda.is_available", return_value=False):
                ok, blockers, device = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertFalse(ok)
            self.assertTrue(any("cuda_required" in b for b in blockers))
            self.assertEqual(device, "")

    def test_07b_auto_policy_falls_back_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td), device="cpu")
            plan["device_policy"] = "auto"
            with mock.patch("torch.cuda.is_available", return_value=False):
                ok, blockers, device = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertTrue(ok, blockers)
            self.assertEqual(device, "cpu")
            self.assertEqual(plan["device"], "cpu")

    def test_a1_single_scene_blocks_before_gpu(self):
        """A1：有效景数 < 频次阈值 → 提前阻断，不加载模型/不设 GPU 设备。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            d = root / "task_single"
            d.mkdir(parents=True, exist_ok=True)
            _make_tif(d / "only.tif")  # 只有 1 景
            final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_single", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=2,
            )
            ok, blockers, device = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertFalse(ok)
            self.assertTrue(any("频次阈值为 2" in b and "只有 1 景" in b for b in blockers))
            # 阻断时设备未启动（不虚报 GPU/CPU）
            self.assertEqual(device, "")
            self.assertEqual(plan["device"], "")

    def test_a1_single_scene_ok_when_cnt_1(self):
        """A1：cnt=1 时单景不阻断（阈值语义允许）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
            weight = _make_fake_weight(tmp / "w.pth")
            d = root / "task_single"
            d.mkdir(parents=True, exist_ok=True)
            _make_tif(d / "only.tif")
            final.mkdir(); mask.mkdir()
            plan = ial.build_inference_plan(
                task_id="task_single", root_dir=str(root), final_root=str(final),
                mask_root=str(mask), model_path=str(weight),
                prob_threshold=0.05, count_threshold=1,
            )
            ok, blockers, _ = ial.validate_inference_plan(plan, check_weight_load=False)
            self.assertTrue(ok, blockers)
            self.assertFalse(any("频次阈值" in b for b in blockers))


# ===============================================================
#  8 / 9 / 10：确认门闩
# ===============================================================
class TestConfirmGate(unittest.TestCase):
    def test_08_unconfirmed_does_not_execute(self):
        state: dict = {}
        init_ui_session_defaults(state)
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            state["_inference_pending_plan"] = plan
            # confirmed=False → build_pending_task 拒绝
            pt, at, errs = build_pending_task(
                state, {"type": "run_inference", "confirmed": False}
            )
            self.assertIsNone(pt)
            self.assertTrue(any("确认" in e for e in errs))
            # confirm_inference_plan 要求 plan_id 匹配
            ok, _ = ial.confirm_inference_plan(state, "wrong-id")
            self.assertFalse(ok)

    def test_09_double_confirm_rejected(self):
        state: dict = {}
        init_ui_session_defaults(state)
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            state["_inference_pending_plan"] = plan
            ok, err = ial.confirm_inference_plan(state, plan["plan_id"])
            self.assertTrue(ok)
            self.assertIsNone(err)
            ok2, err2 = ial.confirm_inference_plan(state, plan["plan_id"])
            self.assertFalse(ok2)
            self.assertIn("请勿重复确认", err2)

    def test_10_rerun_keeps_confirmed_once(self):
        state: dict = {}
        init_ui_session_defaults(state)
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            state["_inference_pending_plan"] = plan
            ial.confirm_inference_plan(state, plan["plan_id"])
            self.assertTrue(ial.is_plan_confirmed(state, plan["plan_id"]))
            # 模拟页面 rerun：状态保留，不重新确认
            pt, _, errs = build_pending_task(
                state, {"type": "run_inference", "confirmed": True,
                        "plan_id": plan["plan_id"]}
            )
            self.assertFalse(errs)
            self.assertIsNotNone(pt)
            self.assertEqual(pt["mode"], "dl")
            self.assertEqual(pt["inference_plan"]["plan_id"], plan["plan_id"])

    def test_confirm_without_pending_plan_fails(self):
        state: dict = {}
        init_ui_session_defaults(state)
        ok, err = ial.confirm_inference_plan(state, "abc")
        self.assertFalse(ok)
        self.assertIn("没有待确认的推理计划", err)

    def test_cancel_clears_pending(self):
        state: dict = {}
        init_ui_session_defaults(state)
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            state["_inference_pending_plan"] = plan
            ial.cancel_inference_plan(state)
            self.assertNotIn("_inference_pending_plan", state)


# ===============================================================
#  11 / 12：真实执行（fake engine 故障）
# ===============================================================
class TestExecuteFailure(unittest.TestCase):
    def test_11_inference_exception_returns_failure(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            result = ial.execute_local_inference(
                plan,
                pre_engine_mod=FakePreEngine(fail_process=True),
                post_engine_mod=FakePostEngine(),
            )
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("单景推理异常", result["error"])
            self.assertFalse(result["outputs"])

    def test_11b_model_load_failure_returns_failure(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            result = ial.execute_local_inference(
                plan,
                pre_engine_mod=FakePreEngine(fail_load=True),
                post_engine_mod=FakePostEngine(),
            )
            self.assertFalse(result["success"])
            self.assertIn("模型加载失败", result["error"])

    def test_12_postprocess_exception_returns_failure(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            result = ial.execute_local_inference(
                plan,
                pre_engine_mod=FakePreEngine(),
                post_engine_mod=FakePostEngine(fail=True),
            )
            self.assertFalse(result["success"])
            self.assertIn("后处理异常", result["error"])

    def test_12b_postprocess_true_without_artifacts_returns_failure(self):
        """后处理返回 True 但未写出成果时，执行层不得伪报成功。"""
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            result = ial.execute_local_inference(
                plan,
                pre_engine_mod=FakePreEngine(),
                post_engine_mod=FakePostEngine(skip_tif=True, skip_shp=True),
            )
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("成果文件", result["error"])
            self.assertFalse(result["outputs"])

    def test_stop_event_interrupts(self):
        import threading
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            stop = threading.Event()
            stop.set()
            result = ial.execute_local_inference(
                plan,
                stop_event=stop,
                pre_engine_mod=FakePreEngine(stop=True),
                post_engine_mod=FakePostEngine(),
            )
            self.assertFalse(result["success"])
            self.assertIn("中断", result["error"])

    def test_stop_event_set_after_postprocess_cancels_before_success(self):
        """后处理返回成功后才收到停止请求，也不得继续报告/提交成功。"""
        import threading

        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            stop = threading.Event()
            base_post = FakePostEngine()

            class LateStopPost:
                def generate_double_constraint_complete(self, *args, **kwargs):
                    result = base_post.generate_double_constraint_complete(*args, **kwargs)
                    stop.set()
                    return result

            result = ial.execute_local_inference(
                plan,
                stop_event=stop,
                pre_engine_mod=FakePreEngine(),
                post_engine_mod=LateStopPost(),
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "cancelled")
            self.assertIn("中断", result["error"])


# ===============================================================
#  13 / 14 / 15 / 16 / 17：验证与登记
# ===============================================================
class TestVerifyAndRegister(unittest.TestCase):
    def _run_happy(self, tmp: Path, post_kwargs=None, pre_kwargs=None):
        plan, root, final, mask, weight = _happy_plan(tmp)
        result = ial.execute_local_inference(
            plan,
            pre_engine_mod=FakePreEngine(**(pre_kwargs or {})),
            post_engine_mod=FakePostEngine(**(post_kwargs or {})),
        )
        return plan, result

    def test_13_final_tif_missing_fails_verify(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp, post_kwargs={"skip_tif": True})
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            self.assertFalse(ver["ok"])
            names = [c["name"] for c in ver["checks"] if not c["passed"]]
            self.assertIn("final_tif_path", names)

    def test_14_final_tif_invalid_fails_verify(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp)
            # 手动把 final tif 覆盖为“全 NoData”的无效成果
            final_tif = result["outputs"]["final_tif"]
            from rasterio.transform import from_origin
            import rasterio
            import numpy as np

            ft = Path(final_tif)
            ft.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                str(ft), "w", driver="GTiff", width=64, height=64, count=1,
                dtype="uint8", crs="EPSG:4326",
                transform=from_origin(120.0, 30.0, 0.01, 0.01), nodata=255,
            ) as dst:
                dst.write(np.full((64, 64), 255, dtype="uint8"), 1)
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            self.assertFalse(ver["ok"])
            names = [c["name"] for c in ver["checks"] if not c["passed"]]
            self.assertIn("final_tif_has_data", names)

    def test_15_success_verify_then_register(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp)
            self.assertTrue(result["success"], result.get("error"))
            self.assertTrue(os.path.isfile(result["outputs"]["final_tif"]))
            self.assertTrue(os.path.isfile(result["outputs"]["final_shp"]))
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            self.assertTrue(ver["ok"], [c for c in ver["checks"] if not c["passed"]])
            reg_path = tmp / "registry.json"
            asset_id = ial.register_inference_asset(
                plan, result, ver, registry_path=str(reg_path)
            )
            self.assertTrue(asset_id)
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            entry = next(v for v in reg.values()
                         if isinstance(v, dict) and v.get("plan_id") == plan["plan_id"])
            self.assertEqual(entry["status"], "verified")
            self.assertEqual(entry["asset_type"], "tidal_flat_prediction")
            self.assertEqual(entry["parameters"]["prob_threshold"], 0.05)
            # find_inference_asset 可回查
            found = ial.find_inference_asset(plan["plan_id"], registry_path=str(reg_path))
            self.assertIsNotNone(found)
            self.assertEqual(found["asset_id"], asset_id)

    def test_corrupt_shared_registry_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.json"
            path.write_text('{"broken": [', encoding="utf-8")
            with self.assertRaises(ValueError):
                ial._load_registry(str(path))
            self.assertTrue(list(Path(td).glob("registry.json.corrupt-*")))

    def test_16_failure_never_registers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp, post_kwargs={"skip_shp": True})
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            self.assertFalse(ver["ok"])  # final_shp 缺失 → 验证失败
            reg_path = tmp / "registry.json"
            asset_id = ial.register_inference_asset(
                plan, result, ver, registry_path=str(reg_path)
            )
            self.assertIsNone(asset_id)
            self.assertFalse(reg_path.exists())

    def test_registration_rechecks_verified_artifacts(self):
        """A forged/stale verification result cannot commit missing outputs."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan = {"task_id": "task_a", "plan_id": "p" * 32}
            result = {"success": True}
            verification = {
                "ok": True,
                "final_tif": str(tmp / "missing_Final.tif"),
                "final_shp": str(tmp / "missing_Final.shp"),
            }
            reg_path = tmp / "registry.json"

            asset_id = ial.register_inference_asset(
                plan, result, verification, registry_path=str(reg_path)
            )

            self.assertIsNone(asset_id)
            self.assertFalse(reg_path.exists())

    def test_17_missing_map_only_warns_no_register(self):
        # 校验失败但最终不阻塞主流程：不登记、不伪报完成
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp)
            # 模拟执行结束后地图资产被外部清理，验证层仍必须 fail-closed。
            Path(result["outputs"]["final_shp"]).unlink()
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            self.assertFalse(ver["ok"])
            summary = ial.summarize_inference_result_for_chat(result, ver)
            self.assertIn("校验未完全通过", summary)
            self.assertNotIn("（已验证）", summary)

    def test_duplicate_plan_never_double_registers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan, result = self._run_happy(tmp)
            ver = ial.verify_inference_outputs(plan, result, started_at=time.time() - 60)
            reg_path = tmp / "registry.json"
            a1 = ial.register_inference_asset(plan, result, ver, registry_path=str(reg_path))
            a2 = ial.register_inference_asset(plan, result, ver, registry_path=str(reg_path))
            self.assertEqual(a1, a2)
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            # 同 plan_id 只登记一次（兼容键 + 扩展键各一条，但 asset_id 唯一）
            entries = [v for v in reg.values()
                       if isinstance(v, dict) and v.get("plan_id") == plan["plan_id"]]
            self.assertEqual(len({e["asset_id"] for e in entries}), 1)


# ===============================================================
#  18 / 19：面向 Copilot 的真实展示
# ===============================================================
class TestSummaryAndContext(unittest.TestCase):
    def _fake_success_result(self):
        return {
            "success": True, "task_id": "task_a", "plan_id": "p" * 32,
            "tool": "local_tidal_flat_inference", "status": "completed",
            "inputs": {
                "input_asset_id": "ui_selected",
                "input_path": "root/task_a",
                "model_id": "cdnet_resnet50",
                "weight_id": "ui_selected",
                "device": "cpu",
            },
            "parameters": {"prob_threshold": 0.05, "count_threshold": 2},
            "outputs": {
                "prediction_tif": "mask/a_mask.tif",
                "final_tif": "out/task_a_Final_p0.05_c2.tif",
                "final_shp": "out/task_a_Final_p0.05_c2.shp",
            },
            "metrics": {"elapsed_seconds": 12.5, "processed_tiles": 1, "tif_count": 1},
            "warnings": [], "error": None,
        }

    def test_18_summary_uses_only_real_metrics(self):
        result = self._fake_success_result()
        ver = {"ok": True, "final_tif": "out/x.tif", "final_shp": "out/x.shp"}
        text = ial.summarize_inference_result_for_chat(result, ver)
        self.assertIn("12.5", text)
        self.assertIn("1/1", text)
        self.assertIn("cpu", text)
        self.assertIn("已验证", text)

    def test_18b_failure_summary_no_fake_completion(self):
        result = {
            "success": False, "task_id": "task_a", "tool": "local_tidal_flat_inference",
            "error": "模型加载失败: boom",
        }
        text = ial.summarize_inference_result_for_chat(result, None)
        self.assertIn("未完成", text)
        self.assertIn("模型加载失败", text)
        self.assertNotIn("已验证", text)

    def test_18d_failure_summary_redacts_local_path(self):
        text = ial.summarize_inference_result_for_chat({
            "success": False,
            "task_id": "task_a",
            "error": "模型失败 /Users/chl/private/secret.tif",
        }, None)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("secret.tif", text)

    def test_18c_format_plan_truthful(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            text = ial.format_inference_plan_for_user(plan)
            self.assertIn("潮滩智能提取 · 执行计划", text)
            self.assertIn("task_a", text)
            self.assertIn("cdnet_resnet50", text)
            blocked = ial.format_inference_plan_for_user({"ready": False, "blockers": ["x"]})
            self.assertIn("暂不可执行", blocked)

    def test_19_context_includes_pending_plan(self):
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            ctx = ial.build_inference_context_for_agent(
                root_dir=str(Path(td) / "root"),
                task_options=["task_a"],
                model_path="cdnet.pth",
                prob_threshold=0.05,
                count_threshold=2,
                device="cpu",
                pending_plan=plan,
            )
            self.assertIn("潮滩智能提取", ctx)
            self.assertIn("local_tidal_flat_inference", ctx)
            self.assertIn("task_a", ctx)
            self.assertIn(plan["plan_id"][:8], ctx)


# ===============================================================
#  20：M5 / E1 / PDF 原流程不受影响
# ===============================================================
class TestExistingFlowsUntouched(unittest.TestCase):
    def test_20_phases_keep_legacy_entries(self):
        import task_timeline as tt

        phases = tt.PHASES
        self.assertIn("INFERENCE", phases)
        self.assertIn("POST_PROCESS", phases)
        self.assertIn("EXECUTE", phases)  # M5/E1 兼容
        self.assertIn("REPORT", phases)

    def test_20b_inference_state_isolated_from_m5_e1(self):
        state: dict = {}
        init_ui_session_defaults(state)
        with tempfile.TemporaryDirectory() as td:
            plan, *_ = _happy_plan(Path(td))
            state["_inference_pending_plan"] = plan
            # 生成推理计划不触碰 M5/E1 键
            self.assertNotIn("_m5_pending_plan", state)
            self.assertNotIn("_e1_pending_plan", state)


# ===============================================================
#  Bridge：propose → confirm → run_inference 门闩
# ===============================================================
class TestBridgeInferenceFlow(unittest.TestCase):
    def _bridge_state(self, tmp: Path) -> dict:
        root = tmp / "root"; final = tmp / "final"; mask = tmp / "mask"
        weight = _make_fake_weight(tmp / "weights" / "cdnet.pth")
        _make_task_dir(root, "task_a")
        final.mkdir(parents=True, exist_ok=True)
        mask.mkdir(parents=True, exist_ok=True)
        s: dict = {}
        init_ui_session_defaults(s)
        s["ui_root_dir"] = str(root)
        s["ui_final_root"] = str(final)
        s["ui_mask_root"] = str(mask)
        s["ui_model_path"] = str(weight)
        s["ui_selected_task"] = "task_a"
        s["ui_prob_th"] = 0.05
        s["ui_min_cnt"] = 2
        s["ui_shp_path"] = ""
        return s

    def test_propose_then_confirm_requires_confirm(self):
        with tempfile.TemporaryDirectory() as td:
            state = self._bridge_state(Path(td))
            r = apply_system_command(
                state, {"pending_action": {"type": "propose_inference", "task": "task_a"}}
            )
            self.assertEqual(r.action_type, "propose_inference")
            self.assertIsNotNone(r.inference_plan)
            self.assertIn("潮滩智能提取", r.inference_plan_text)
            self.assertTrue(state["_inference_pending_plan"]["ready"])
            self.assertNotIn("pending_task", state)

            pt, _, errs = build_pending_task(
                state, {"type": "run_inference", "confirmed": False}
            )
            self.assertIsNone(pt)
            self.assertTrue(any("确认" in e for e in errs))

            pt2, _, errs2 = build_pending_task(
                state, {"type": "run_inference", "confirmed": True}
            )
            self.assertFalse(errs2)
            self.assertEqual(pt2["mode"], "dl")
            self.assertTrue(pt2["inference_plan"])
            self.assertEqual(pt2["inference_plan"]["task_id"], "task_a")

    def test_confirm_action_converts_to_run(self):
        with tempfile.TemporaryDirectory() as td:
            state = self._bridge_state(Path(td))
            propose_inference_plan(state, {"task": "task_a"})
            pid = state["_inference_pending_plan"]["plan_id"]
            r = apply_system_command(
                state, {"pending_action": {"type": "confirm_inference", "plan_id": pid}}
            )
            self.assertEqual(r.action_type, "run_inference")
            self.assertTrue(ial.is_plan_confirmed(state, pid))
            self.assertIn("pending_task", state)
            self.assertEqual(state["pending_task"]["plan_id"], pid)

    def test_blocked_propose_reports_errors(self):
        with tempfile.TemporaryDirectory() as td:
            state = self._bridge_state(Path(td))
            state["ui_root_dir"] = str(Path(td) / "no_root")
            r = apply_system_command(
                state, {"pending_action": {"type": "propose_inference", "task": "task_a"}}
            )
            self.assertFalse(state["_inference_pending_plan"]["ready"])
            self.assertTrue(r.errors)


class TestInferenceRecoveryClassification(unittest.TestCase):
    def test_complete_final_is_reused_only_after_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            task_dir = _make_task_dir(root, "task_a")
            final = Path(td) / "final"; mask = Path(td) / "mask"
            final.mkdir(); mask.mkdir()
            tif = final / "task_a_Final_p0.05_c2.tif"; shp = final / "task_a_Final_p0.05_c2.shp"
            tif.write_bytes(b"tif"); shp.write_bytes(b"shp")
            plan = {"task_id": "task_a", "input_path": str(task_dir), "output_dir": str(final), "mask_dir": str(mask), "prob_threshold": 0.05, "count_threshold": 2}
            result = ial.classify_inference_recovery(plan, verify_fn=lambda *args, **kwargs: {"ok": True})
            self.assertEqual(result["action"], "COMPLETE")

    def test_complete_masks_resume_post_process(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            task_dir = _make_task_dir(root, "task_a")
            final = Path(td) / "final"; mask = Path(td) / "mask"; final.mkdir(); mask.mkdir()
            for name in ("scene_001_mask.tif", "scene_002_mask.tif"):
                (mask / name).write_bytes(b"mask")
            plan = {"task_id": "task_a", "input_path": str(task_dir), "output_dir": str(final), "mask_dir": str(mask), "prob_threshold": 0.05, "count_threshold": 2}
            self.assertEqual(ial.classify_inference_recovery(plan)["action"], "RESUME_POST_PROCESS")

    def test_partial_masks_wait_for_confirmed_isolation_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            task_dir = _make_task_dir(root, "task_a")
            final = Path(td) / "final"; mask = Path(td) / "mask"; final.mkdir(); mask.mkdir()
            (mask / "scene_001_mask.tif").write_bytes(b"mask")
            (mask / "scene_002_mask.tif").touch()
            plan = {"task_id": "task_a", "input_path": str(task_dir), "output_dir": str(final), "mask_dir": str(mask), "prob_threshold": 0.05, "count_threshold": 2}
            result = ial.classify_inference_recovery(plan)
            self.assertEqual(result["action"], "ISOLATE_AND_RETRY")
            self.assertEqual(result["invalid_count"], 1)

    def test_isolation_requires_confirmation_and_preserves_audit_copy(self):
        with tempfile.TemporaryDirectory() as td:
            mask = Path(td) / "mask"; final = Path(td) / "final"
            mask.mkdir(); final.mkdir()
            source = mask / "scene_001_mask.tif"
            source.write_bytes(b"partial")
            plan = {"plan_id": "plan/unsafe", "mask_dir": str(mask), "output_dir": str(final)}
            waiting = ial.isolate_inference_checkpoints(plan)
            self.assertEqual(waiting["status"], "WAITING_CONFIRMATION")
            self.assertTrue(source.exists())
            isolated = ial.isolate_inference_checkpoints(plan, confirmed=True)
            self.assertTrue(isolated["ok"])
            self.assertFalse(source.exists())
            self.assertTrue((Path(isolated["quarantine_dir"]) / source.name).exists())


if __name__ == "__main__":
    unittest.main()
