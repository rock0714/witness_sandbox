"""
witness_sandbox.py の自動テストスイート。
オフスクリーン環境（QT_QPA_PLATFORM="offscreen"）で各機能・コンポーネントを
項目ごとに独立して検証する。
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

# PyQt6 のインポート前にオフスクリーンモードを設定
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

import witness_sandbox
from app import canvas, capture, vision, window
from app.canvas import (
    DrawMode,
    ImageCanvasBase,
    PerspectiveCanvas,
    SandboxCanvas,
    cv_to_qpixmap,
)
from app.vision import (
    create_sample_witness_image,
    detect_grid_dimensions,
    load_image_bgr,
    normalize_save_image_path,
    order_points,
    point_to_segment_distance,
    validate_quad_points,
    warp_perspective,
)
from app.window import DARK_STYLE_QSS, MainWindow

# プロセス内で単一の QApplication インスタンスを保持
_QAPP: QApplication | None = None

# サンプル盤面における基準4隅座標
SAMPLE_RAW_POINTS = np.array(
    [
        [890, 580],  # 右下
        [220, 110],  # 左上
        [150, 610],  # 左下
        [780, 70],   # 右上
    ],
    dtype="float32",
)


def get_qapp() -> QApplication:
    """オフスクリーン検証用の QApplication インスタンスを取得または生成する。"""
    global _QAPP
    instance = QApplication.instance()
    if instance is not None:
        return instance  # type: ignore[return-value]
    _QAPP = QApplication(sys.argv)
    return _QAPP


# ==============================================================================
# 1. 幾何計算・画像処理・サンプル生成の単体テスト
# ==============================================================================


class TestGeometryAndVision(unittest.TestCase):
    """幾何計算・グリッド自動検知・透視投影変換のテスト。"""

    sample_img: np.ndarray
    ordered_pts: np.ndarray

    @classmethod
    def setUpClass(cls):
        get_qapp()
        cls.sample_img = create_sample_witness_image()
        cls.ordered_pts = order_points(SAMPLE_RAW_POINTS)

    def test_create_sample_witness_image(self):
        """サンプルパズル画像が期待サイズ (700x1000x3) で生成され QPixmap 変換できること。"""
        self.assertIsNotNone(self.sample_img, "サンプル画像が生成できませんでした")
        self.assertEqual(
            self.sample_img.shape,
            (700, 1000, 3),
            f"サンプル画像の形状が不正です: {self.sample_img.shape}",
        )
        pixmap = cv_to_qpixmap(self.sample_img)
        self.assertFalse(pixmap.isNull(), "QPixmap への変換に失敗しました")

    def test_order_points(self):
        """任意の順序の4頂点が [左上, 右上, 右下, 左下] に整列されること。"""
        ordered = order_points(SAMPLE_RAW_POINTS)
        np.testing.assert_allclose(ordered[0], [220, 110], err_msg="TLが一致しません")
        np.testing.assert_allclose(ordered[1], [780, 70], err_msg="TRが一致しません")
        np.testing.assert_allclose(ordered[2], [890, 580], err_msg="BRが一致しません")
        np.testing.assert_allclose(ordered[3], [150, 610], err_msg="BLが一致しません")

    def test_detect_grid_dimensions(self):
        """サンプル画像から列数4・行数4および妥当なマージン割合が自動検知されること。"""
        det_cols, det_rows, det_margin = detect_grid_dimensions(
            self.sample_img, self.ordered_pts
        )
        self.assertEqual(
            (det_cols, det_rows),
            (4, 4),
            f"検知されたマス数が不正です: {det_cols}x{det_rows} (期待値: 4x4)",
        )
        self.assertTrue(
            0.05 <= det_margin <= 0.15,
            f"検知されたマージンが不正です: {det_margin}",
        )

    def test_warp_perspective_square(self):
        """正方形 (1:1) モードの透視投影変換で縦横同サイズの画像が得られること。"""
        warped_sq = warp_perspective(
            self.sample_img, self.ordered_pts, aspect_mode="square"
        )
        self.assertIsNotNone(warped_sq)
        self.assertGreater(warped_sq.size, 0)
        h_sq, w_sq, _ = warped_sq.shape
        self.assertEqual(h_sq, w_sq, f"正方形比率になっていません: {w_sq}x{h_sq}")

    def test_warp_perspective_grid_aspect(self):
        """グリッド比率 (6:3=2:1) モードの透視投影変換で指定アスペクト比になること。"""
        warped_grid = warp_perspective(
            self.sample_img, self.ordered_pts, aspect_mode="grid", cols=6, rows=3
        )
        h_gr, w_gr, _ = warped_grid.shape
        self.assertLess(
            abs(w_gr / h_gr - 2.0),
            0.05,
            f"グリッド比率(6:3=2:1)になっていません: {w_gr}x{h_gr}",
        )

    def test_point_to_segment_distance(self):
        """点と線分の最短ユークリッド距離（線分上投影および端点外）が正確であること。"""
        d1 = point_to_segment_distance(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(d1, 5.0, places=5, msg=f"線分との距離が不正: {d1}")

        d2 = point_to_segment_distance(12.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(d2, 2.0, places=5, msg=f"端点外の距離が不正: {d2}")

    def test_order_points_rotated_diamond_and_degenerate_validation(self):
        """大きく回転したひし形でも4点が重複なく整列され、重複・一直線・非凸4点は拒否されること。"""
        # 旧アルゴリズム (x+y / x-y 極値) では重複選択が起きていたひし形座標
        diamond_shuffled = np.array(
            [
                [300, 500],  # 下
                [120, 320],  # 左
                [300, 100],  # 上
                [500, 280],  # 右
            ],
            dtype="float32",
        )
        ordered = validate_quad_points(diamond_shuffled)
        # 4頂点すべてが一意に保持され、時計回り [上, 右, 下, 左] になること
        np.testing.assert_allclose(ordered[0], [300, 100])
        np.testing.assert_allclose(ordered[1], [500, 280])
        np.testing.assert_allclose(ordered[2], [300, 500])
        np.testing.assert_allclose(ordered[3], [120, 320])

        warped = warp_perspective(self.sample_img, diamond_shuffled, aspect_mode="square")
        self.assertEqual(warped.shape[0], warped.shape[1])

        # 1. 重複点を含む4点 -> ValueError
        duplicate_pts = np.array(
            [[100, 100], [100, 100], [400, 100], [400, 400]], dtype="float32"
        )
        with self.assertRaises(ValueError):
            validate_quad_points(duplicate_pts)
        with self.assertRaises(ValueError):
            warp_perspective(self.sample_img, duplicate_pts)

        # 2. 一直線 (共線) 上の4点 -> ValueError
        collinear_pts = np.array(
            [[100, 100], [200, 200], [300, 300], [400, 400]], dtype="float32"
        )
        with self.assertRaises(ValueError):
            validate_quad_points(collinear_pts)

        # 3. 凹四角形 (1点が他3点の三角形内部にあるブーメラン型) -> ValueError
        concave_pts = np.array(
            [[100, 100], [500, 100], [250, 220], [100, 500]], dtype="float32"
        )
        with self.assertRaises(ValueError):
            validate_quad_points(concave_pts)

        # 4. 極小面積 (各辺 >= 5px だが面積 36 < 100) および NaN 入力 -> ValueError
        tiny_pts = np.array(
            [[100, 100], [106, 100], [106, 106], [100, 106]], dtype="float32"
        )
        with self.assertRaises(ValueError):
            validate_quad_points(tiny_pts)

        nan_pts = np.array(
            [[100, 100], [float("nan"), 100], [200, 200], [100, 200]], dtype="float32"
        )
        with self.assertRaises(ValueError):
            order_points(nan_pts)


# ==============================================================================
# 2. PerspectiveCanvas コンポーネントのテスト
# ==============================================================================


class TestPerspectiveCanvas(unittest.TestCase):
    """四隅指定キャンバス (PerspectiveCanvas) のテスト。"""

    sample_img: np.ndarray

    @classmethod
    def setUpClass(cls):
        get_qapp()
        cls.sample_img = create_sample_witness_image()

    def setUp(self):
        self.canvas = PerspectiveCanvas()
        self.canvas.resize(800, 600)

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()

    def test_set_image_and_four_points(self):
        """画像のセットで QPixmap が生成され、4隅の座標を保持できること。"""
        self.canvas.set_image(self.sample_img)
        self.assertFalse(self.canvas.pixmap.isNull(), "Pixmap が空です")

        self.canvas.points = [
            QPointF(220, 110),
            QPointF(780, 70),
            QPointF(890, 580),
            QPointF(150, 610),
        ]
        self.assertEqual(len(self.canvas.points), 4, "4点が保持されていません")

    def test_common_image_canvas_base_inheritance_and_coords(self):
        """両キャンバスが ImageCanvasBase を継承し、座標往復変換と空プレースホルダ描画が機能すること。"""
        self.assertTrue(issubclass(PerspectiveCanvas, ImageCanvasBase))
        self.assertTrue(issubclass(SandboxCanvas, ImageCanvasBase))

        # 重複メソッドがサブクラス側で再定義されていないこと
        for method_name in (
            "get_display_scale_and_offset",
            "widget_to_image_coords",
            "image_to_widget_coords",
        ):
            self.assertNotIn(method_name, PerspectiveCanvas.__dict__)
            self.assertNotIn(method_name, SandboxCanvas.__dict__)

        # 空画像状態でのスケール計算およびプレースホルダ描画
        self.assertEqual(self.canvas.get_display_scale_and_offset(), (1.0, 0.0, 0.0))
        empty_grab = self.canvas.grab()
        self.assertFalse(empty_grab.isNull())

        empty_sb = SandboxCanvas()
        empty_sb.resize(400, 300)
        self.assertFalse(empty_sb.grab().isNull())
        empty_sb.close()
        empty_sb.deleteLater()

        # 画像設定後の横長・縦長リサイズにおけるレターボックス計算と往復変換の一致
        self.canvas.set_image(self.sample_img)  # 1000x700
        for w, h in ((1000, 500), (500, 900)):
            self.canvas.resize(w, h)
            scale, ox, oy = self.canvas.get_display_scale_and_offset()
            self.assertGreater(scale, 0.0)
            self.assertTrue(ox > 0.0 or oy > 0.0)
            orig_pt = QPointF(345.5, 210.25)
            roundtrip = self.canvas.widget_to_image_coords(
                self.canvas.image_to_widget_coords(orig_pt)
            )
            self.assertAlmostEqual(roundtrip.x(), orig_pt.x(), places=5)
            self.assertAlmostEqual(roundtrip.y(), orig_pt.y(), places=5)


# ==============================================================================
# 3. SandboxCanvas コンポーネントのテスト
# ==============================================================================


class TestSandboxCanvas(unittest.TestCase):
    """サンドボックス描画キャンバス (SandboxCanvas) の独立機能テスト。"""

    warped_sq: np.ndarray
    det_margin: float

    @classmethod
    def setUpClass(cls):
        get_qapp()
        sample_img = create_sample_witness_image()
        ordered = order_points(SAMPLE_RAW_POINTS)
        _, _, cls.det_margin = detect_grid_dimensions(sample_img, ordered)
        cls.warped_sq = warp_perspective(sample_img, ordered, aspect_mode="square")

    def setUp(self):
        self.canvas = SandboxCanvas()
        self.canvas.resize(600, 600)
        self.canvas.show()
        self.canvas.set_warped_image(self.warped_sq)
        self.canvas.set_grid_dimensions(4, 4)
        self.canvas.set_margin_percent(self.det_margin)
        self.canvas.set_draw_mode(DrawMode.SNAP_LINE)

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()

    def test_edge_click_toggle_and_undo(self):
        """辺のクリックで塗り・再クリックで消去され、Undo で段階的に戻ること。"""
        a = self.canvas.vertex_to_widget_coords(0, 0)
        b = self.canvas.vertex_to_widget_coords(1, 0)
        mid = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
        key = ((0, 0), (1, 0))

        self.assertEqual(
            self.canvas.find_nearest_edge(mid), key, "辺の当たり判定に失敗しました"
        )

        # 左クリックで塗り -> 同じ辺を再クリックで消去
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid.toPoint(),
        )
        self.assertIn(key, self.canvas.edges, "クリックで辺が塗られていません")

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid.toPoint(),
        )
        self.assertNotIn(
            key, self.canvas.edges, "塗り済みの辺を再クリックしても消えません"
        )

        # Undo: 消去の取り消し -> 塗りの取り消し
        self.canvas.undo()
        self.assertIn(key, self.canvas.edges, "Undoで消去が復元されていません")
        self.canvas.undo()
        self.assertNotIn(key, self.canvas.edges, "Undoで塗りが取り消されていません")

    def test_edge_drag_paint_and_erase(self):
        """未塗りの辺からのドラッグで連続塗り、塗り済み辺からのドラッグで連続消去されること。"""
        start = self.canvas.vertex_to_widget_coords(0, 0)
        end = self.canvas.vertex_to_widget_coords(4, 0)

        # 上辺 (0,0)->(4,0) を左から右へドラッグ塗り
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(start.x() + 5, start.y()).toPoint(),
        )
        for i in range(1, 41):
            x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
            QTest.mouseMove(self.canvas, QPointF(x, start.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(end.x() - 5, end.y()).toPoint(),
        )

        top_edges = [((c, 0), (c + 1, 0)) for c in range(4)]
        self.assertTrue(
            all(e in self.canvas.edges for e in top_edges),
            f"ドラッグで上辺4本が塗られていません: {list(self.canvas.edges)}",
        )
        self.assertEqual(
            len(self.canvas.edges),
            4,
            f"余計な辺が塗られています: {list(self.canvas.edges)}",
        )

        # 塗り済みの辺から開始したドラッグで消去
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(start.x() + 5, start.y()).toPoint(),
        )
        for i in range(1, 41):
            x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
            QTest.mouseMove(self.canvas, QPointF(x, start.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(end.x() - 5, end.y()).toPoint(),
        )
        self.assertEqual(
            len(self.canvas.edges),
            0,
            f"ドラッグ消去できていません: {list(self.canvas.edges)}",
        )

    def test_eraser_removes_edge_stroke_and_cell_fill(self):
        """消しゴム (erase_at_pos) で塗り辺・手書きストローク・セル塗りがそれぞれ消去できること。"""
        a = self.canvas.vertex_to_widget_coords(0, 0)
        b = self.canvas.vertex_to_widget_coords(1, 0)
        mid = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
        key = ((0, 0), (1, 0))

        self.canvas.apply_edge(key, "paint")
        cell = (1, 1)
        self.canvas.cell_fills[cell] = QColor(0, 200, 255)
        self.canvas.freehand_strokes.append(
            {
                "points": [
                    self.canvas.widget_to_image_coords(QPointF(100, 100)),
                    self.canvas.widget_to_image_coords(QPointF(120, 120)),
                ],
                "color": QColor(255, 255, 255),
                "width": 4,
            }
        )

        self.assertTrue(self.canvas.erase_at_pos(mid))
        self.assertNotIn(
            key, self.canvas.edges, "消しゴムで塗り辺が消去されていません"
        )

        self.assertTrue(self.canvas.erase_at_pos(QPointF(110, 110)))
        self.assertEqual(
            len(self.canvas.freehand_strokes),
            0,
            "消しゴムで手書きストロークが消去されていません",
        )

        cell_wpt = self.canvas.vertex_to_widget_coords(1, 1)
        cell_center = QPointF(cell_wpt.x() + 20, cell_wpt.y() + 20)
        self.assertTrue(self.canvas.erase_at_pos(cell_center))
        self.assertNotIn(
            cell, self.canvas.cell_fills, "消しゴムでセル塗りが消去されていません"
        )

    def test_freehand_stroke_stays_aligned_on_resize(self):
        """ウィンドウを縦長・横長にリサイズしても自由手書きが盤面位置からずれず消去も可能なこと。"""
        self.canvas.set_draw_mode(DrawMode.FREEHAND)

        # 600x600 表示時にグリッド頂点 (1, 1) -> (3, 2) を結ぶストロークをマウス描画
        p_start_init = QPointF(self.canvas.vertex_to_widget_coords(1, 1).toPoint())
        p_end_init = QPointF(self.canvas.vertex_to_widget_coords(3, 2).toPoint())
        expected_img_start = self.canvas.widget_to_image_coords(p_start_init)
        expected_img_end = self.canvas.widget_to_image_coords(p_end_init)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p_start_init.toPoint(),
        )
        for i in range(1, 11):
            t = i / 10.0
            pt = QPointF(
                p_start_init.x() + (p_end_init.x() - p_start_init.x()) * t,
                p_start_init.y() + (p_end_init.y() - p_start_init.y()) * t,
            )
            QTest.mouseMove(self.canvas, pt.toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p_end_init.toPoint(),
        )

        self.assertEqual(len(self.canvas.freehand_strokes), 1, "自由手書きが記録されていません")
        stroke = self.canvas.freehand_strokes[0]
        self.assertEqual(stroke["width"], 4, "手書きの線幅は4固定であるべきです")
        self.assertAlmostEqual(stroke["points"][0].x(), expected_img_start.x(), places=4)
        self.assertAlmostEqual(stroke["points"][0].y(), expected_img_start.y(), places=4)
        self.assertAlmostEqual(stroke["points"][-1].x(), expected_img_end.x(), places=4)
        self.assertAlmostEqual(stroke["points"][-1].y(), expected_img_end.y(), places=4)

        # 横長 (左右余白発生) および縦長 (上下余白発生) へリサイズしても盤面頂点と一致し続ける
        for new_w, new_h in [(1000, 500), (480, 900)]:
            self.canvas.resize(new_w, new_h)
            # 実際のレンダリング処理 (paintEvent) が例外なく完了することも確認
            rendered = self.canvas.grab()
            self.assertFalse(rendered.isNull())

            v_start_now = self.canvas.vertex_to_widget_coords(1, 1)
            v_end_now = self.canvas.vertex_to_widget_coords(3, 2)
            s_start_now = self.canvas.image_to_widget_coords(stroke["points"][0])
            s_end_now = self.canvas.image_to_widget_coords(stroke["points"][-1])

            # 初期クリック時の整数ピクセル丸め (最大 1px * 倍率比) の範囲内で盤面頂点に追従していること
            self.assertLessEqual(
                np.hypot(s_start_now.x() - v_start_now.x(), s_start_now.y() - v_start_now.y()),
                2.0,
                f"リサイズ ({new_w}x{new_h}) 後に手書き始点が盤面からずれています",
            )
            self.assertLessEqual(
                np.hypot(s_end_now.x() - v_end_now.x(), s_end_now.y() - v_end_now.y()),
                2.0,
                f"リサイズ ({new_w}x{new_h}) 後に手書き終点が盤面からずれています",
            )

        # リサイズ後 (480x900) の表示位置の中点で消しゴム当たり判定・消去が機能すること
        s_start_now = self.canvas.image_to_widget_coords(stroke["points"][0])
        s_end_now = self.canvas.image_to_widget_coords(stroke["points"][-1])
        mid_resized = QPointF(
            (s_start_now.x() + s_end_now.x()) / 2.0,
            (s_start_now.y() + s_end_now.y()) / 2.0,
        )
        self.assertTrue(
            self.canvas.erase_at_pos(mid_resized),
            "リサイズ後の表示座標で自由手書きを消しゴム消去できません",
        )
        self.assertEqual(len(self.canvas.freehand_strokes), 0)

    def test_grid_adjust_corner_edge_and_move(self):
        """枠調整モードで角リサイズ・辺リサイズ・全体移動が機能し、線が塗られないこと。"""
        self.canvas.set_margin_percent(0.30)
        self.canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        r0 = list(self.canvas.grid_rect)

        # 左上の角を外側へドラッグ -> left/top のみ減少
        tl = self.canvas.vertex_to_widget_coords(0, 0)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            tl.toPoint(),
        )
        QTest.mouseMove(self.canvas, QPointF(tl.x() - 60, tl.y() - 60).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(tl.x() - 60, tl.y() - 60).toPoint(),
        )
        r1 = list(self.canvas.grid_rect)
        self.assertTrue(
            r1[0] < r0[0] and r1[1] < r0[1],
            f"角ドラッグで外枠が広がっていません: {r0} -> {r1}",
        )
        self.assertAlmostEqual(r1[2], r0[2], places=6, msg="右辺が動いてしまいました")
        self.assertAlmostEqual(r1[3], r0[3], places=6, msg="下辺が動いてしまいました")

        # 右辺の中点を外側へドラッグ -> right のみ増加
        rp = self.canvas.grid_handle_positions()["r"]
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            rp.toPoint(),
        )
        QTest.mouseMove(self.canvas, QPointF(rp.x() + 40, rp.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(rp.x() + 40, rp.y()).toPoint(),
        )
        r2 = list(self.canvas.grid_rect)
        self.assertGreater(r2[2], r1[2], f"右辺ドラッグが反映されていません: {r1} -> {r2}")
        self.assertAlmostEqual(r2[0], r1[0], places=6)
        self.assertAlmostEqual(r2[3], r1[3], places=6)

        # 枠の内側をドラッグ -> サイズを保ったまま全体移動
        c = self.canvas.vertex_to_widget_coords(2, 2)
        w0, h0 = r2[2] - r2[0], r2[3] - r2[1]
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            c.toPoint(),
        )
        QTest.mouseMove(self.canvas, QPointF(c.x() + 20, c.y() + 10).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(c.x() + 20, c.y() + 10).toPoint(),
        )
        r3 = list(self.canvas.grid_rect)
        self.assertAlmostEqual(r3[2] - r3[0], w0, places=6, msg="全体移動で幅が変わりました")
        self.assertAlmostEqual(r3[3] - r3[1], h0, places=6, msg="全体移動で高さが変わりました")
        self.assertNotEqual(r3[0], r2[0], "全体移動できていません")
        self.assertEqual(len(self.canvas.edges), 0, "枠調整中に辺が塗られています")

    def test_single_direction_per_drag_operation(self):
        """1回のドラッグ操作中は、交点から2方向へ同時に分岐して線が伸びないこと。"""
        p11 = self.canvas.vertex_to_widget_coords(1, 1)
        p21 = self.canvas.vertex_to_widget_coords(2, 1)
        p22 = self.canvas.vertex_to_widget_coords(2, 2)
        p31 = self.canvas.vertex_to_widget_coords(3, 1)
        mid_11_21 = QPointF((p11.x() + p21.x()) / 2.0, (p11.y() + p21.y()) / 2.0)
        mid_21_31 = QPointF((p21.x() + p31.x()) / 2.0, (p21.y() + p31.y()) / 2.0)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_11_21.toPoint(),
        )
        self.assertIn(((1, 1), (2, 1)), self.canvas.edges)

        for i in range(1, 11):
            pt = QPointF(p21.x(), p21.y() + (p22.y() - p21.y()) * i / 10.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        self.assertIn(((2, 1), (2, 2)), self.canvas.edges)
        self.assertEqual(len(self.canvas.edges), 2)

        # マウスを離さずに別分岐 ((2,1)-(3,1)) に触れても塗られない
        QTest.mouseMove(self.canvas, mid_21_31.toPoint())
        self.assertNotIn(
            ((2, 1), (3, 1)),
            self.canvas.edges,
            "1操作中に交点から2方向へ同時に伸びてしまっています",
        )
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p22.toPoint(),
        )

    def test_third_branch_edge_in_separate_operation(self):
        """別の操作（クリック）を行えば、既存の交点に3本目の線（T字路）を描画できること。"""
        p11 = self.canvas.vertex_to_widget_coords(1, 1)
        p21 = self.canvas.vertex_to_widget_coords(2, 1)
        p22 = self.canvas.vertex_to_widget_coords(2, 2)
        p31 = self.canvas.vertex_to_widget_coords(3, 1)
        mid_11_21 = QPointF((p11.x() + p21.x()) / 2.0, (p11.y() + p21.y()) / 2.0)
        mid_21_22 = QPointF((p21.x() + p22.x()) / 2.0, (p21.y() + p22.y()) / 2.0)
        mid_21_31 = QPointF((p21.x() + p31.x()) / 2.0, (p21.y() + p31.y()) / 2.0)

        # 交点 (2,1) に2本の辺を接続させる
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_11_21.toPoint(),
        )
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_21_22.toPoint(),
        )
        self.assertEqual(len(self.canvas.edges), 2)

        # 別操作で3本目 ((2,1), (3,1)) をクリック描画
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_21_31.toPoint(),
        )
        self.assertIn(
            ((2, 1), (3, 1)),
            self.canvas.edges,
            "同一交点への3本目の線が描画できませんでした",
        )
        self.assertEqual(len(self.canvas.edges), 3)

    def test_clear_all_undo_restores_drawings_and_history(self):
        """全消去 (clear_all) 後に Undo 1回で全描画が復元され、それ以前の履歴も保たれること。"""
        # 空盤面での clear_all は無駄な Undo 履歴を積まない
        self.canvas.clear_all()
        self.assertEqual(len(self.canvas.undo_stack), 0)

        # 1. 辺塗り
        p00 = self.canvas.vertex_to_widget_coords(0, 0)
        p10 = self.canvas.vertex_to_widget_coords(1, 0)
        mid_edge = QPointF((p00.x() + p10.x()) / 2.0, (p00.y() + p10.y()) / 2.0)
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_edge.toPoint(),
        )
        edge_key = ((0, 0), (1, 0))
        self.assertIn(edge_key, self.canvas.edges)

        # 2. マス目塗り
        self.canvas.set_draw_mode(DrawMode.CELL_FILL)
        p11 = self.canvas.vertex_to_widget_coords(1, 1)
        p22 = self.canvas.vertex_to_widget_coords(2, 2)
        cell_center = QPointF((p11.x() + p22.x()) / 2.0, (p11.y() + p22.y()) / 2.0)
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            cell_center.toPoint(),
        )
        self.assertIn((1, 1), self.canvas.cell_fills)

        # 3. 自由手書き
        self.canvas.set_draw_mode(DrawMode.FREEHAND)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p11.toPoint(),
        )
        QTest.mouseMove(self.canvas, p22.toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p22.toPoint(),
        )
        self.assertEqual(len(self.canvas.freehand_strokes), 1)

        # 全消去を実行 -> すべて消える
        self.canvas.clear_all()
        self.assertEqual(len(self.canvas.edges), 0)
        self.assertEqual(len(self.canvas.cell_fills), 0)
        self.assertEqual(len(self.canvas.freehand_strokes), 0)

        # Undo 1回で 辺・マス目塗り・自由手書きの3種すべてが一括復元される
        self.canvas.undo()
        self.assertIn(edge_key, self.canvas.edges, "全消去後のUndoで辺が復元されていません")
        self.assertIn((1, 1), self.canvas.cell_fills, "全消去後のUndoでセル塗りが復元されていません")
        self.assertEqual(
            len(self.canvas.freehand_strokes), 1, "全消去後のUndoで自由手書きが復元されていません"
        )

        # さらに Undo を重ねると、全消去より前の個別操作（手書き -> セル塗り -> 辺）が順に取り消される
        self.canvas.undo()
        self.assertEqual(len(self.canvas.freehand_strokes), 0)
        self.assertIn((1, 1), self.canvas.cell_fills)
        self.canvas.undo()
        self.assertNotIn((1, 1), self.canvas.cell_fills)
        self.assertIn(edge_key, self.canvas.edges)
        self.canvas.undo()
        self.assertNotIn(edge_key, self.canvas.edges)

    def test_drag_operations_undo_in_single_step(self):
        """線塗り・線消去・セル塗り・消しゴムの連続ドラッグ操作がそれぞれ Undo 1回で戻ること。"""
        start = self.canvas.vertex_to_widget_coords(0, 0)
        end = self.canvas.vertex_to_widget_coords(4, 0)

        # 1. 線塗り連続ドラッグ (4辺) -> Undo 1回で全消去
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(start.x() + 5, start.y()).toPoint(),
        )
        for i in range(1, 41):
            x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
            QTest.mouseMove(self.canvas, QPointF(x, start.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(end.x() - 5, end.y()).toPoint(),
        )
        self.assertEqual(len(self.canvas.edges), 4)
        self.canvas.undo()
        self.assertEqual(
            len(self.canvas.edges), 0, "連続ドラッグ塗りが Undo 1回で戻っていません"
        )

        # 2. 線消去連続ドラッグ (4辺) -> Undo 1回で4辺復元
        for c in range(4):
            self.canvas.apply_edge(((c, 0), (c + 1, 0)), "paint")
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(start.x() + 5, start.y()).toPoint(),
        )
        for i in range(1, 41):
            x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
            QTest.mouseMove(self.canvas, QPointF(x, start.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(end.x() - 5, end.y()).toPoint(),
        )
        self.assertEqual(len(self.canvas.edges), 0)
        self.canvas.undo()
        self.assertEqual(
            len(self.canvas.edges), 4, "連続ドラッグ消去が Undo 1回で復元されていません"
        )

        # 3. マス目塗り連続ドラッグ (余白から進入して3セル連続塗り) -> Undo 1回で3セル消去
        self.canvas.set_draw_mode(DrawMode.CELL_FILL)
        c0_tl = self.canvas.vertex_to_widget_coords(0, 1)
        c0_br = self.canvas.vertex_to_widget_coords(1, 2)
        c2_tl = self.canvas.vertex_to_widget_coords(2, 1)
        c2_br = self.canvas.vertex_to_widget_coords(3, 2)
        pt_margin_outside = QPointF(2.0, (c0_tl.y() + c0_br.y()) / 2.0)
        pt_c2 = QPointF((c2_tl.x() + c2_br.x()) / 2.0, (c2_tl.y() + c2_br.y()) / 2.0)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            pt_margin_outside.toPoint(),
        )
        for i in range(1, 31):
            x = pt_margin_outside.x() + (pt_c2.x() - pt_margin_outside.x()) * i / 30.0
            QTest.mouseMove(
                self.canvas,
                QPointF(x, pt_c2.y()).toPoint(),
            )
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            pt_c2.toPoint(),
        )
        self.assertEqual(
            set(self.canvas.cell_fills.keys()),
            {(0, 1), (1, 1), (2, 1)},
            f"余白からのドラッグ進入で3セルが塗られていません: {list(self.canvas.cell_fills)}",
        )
        self.canvas.undo()
        self.assertEqual(
            len(self.canvas.cell_fills), 0, "セル連続塗りが Undo 1回で戻っていません"
        )

        # 4. 消しゴム連続ドラッグ (mouseMoveEvent 経由で4辺を消去) -> Undo 1回で一括復元
        self.canvas.set_draw_mode(DrawMode.ERASER)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(start.x() + 5, start.y()).toPoint(),
        )
        for i in range(1, 41):
            x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
            QTest.mouseMove(self.canvas, QPointF(x, start.y()).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(end.x() - 5, end.y()).toPoint(),
        )
        self.assertEqual(len(self.canvas.edges), 0)
        self.canvas.undo()
        self.assertEqual(
            len(self.canvas.edges), 4, "消しゴム連続ドラッグ消去が Undo 1回で復元されていません"
        )

    def test_drag_backtrack_does_not_resurrect_rewound_edge_on_undo(self):
        """ドラッグ中に巻き戻して消した線が Undo 時に蘇らず、既存別色の上書き巻き戻しも復元されること。"""
        p00 = self.canvas.vertex_to_widget_coords(0, 0)
        p10 = self.canvas.vertex_to_widget_coords(1, 0)
        p20 = self.canvas.vertex_to_widget_coords(2, 0)
        p21 = self.canvas.vertex_to_widget_coords(2, 1)
        mid_00_10 = QPointF((p00.x() + p10.x()) / 2.0, (p00.y() + p10.y()) / 2.0)

        # あらかじめ ((2,0),(2,1)) にシアン色を塗っておく
        cyan = QColor(0, 208, 255)
        self.canvas.set_color(cyan)
        self.canvas.apply_edge(((2, 0), (2, 1)), "paint")
        self.canvas.set_color(QColor(255, 170, 0))  # オレンジに戻す

        # (0,0)-(1,0) から (1,0)-(2,0)、さらに (2,0)-(2,1) までドラッグしてオレンジで上書き
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_00_10.toPoint(),
        )
        for i in range(1, 16):
            pt = QPointF(p10.x() + (p20.x() - p10.x()) * i / 15.0, p10.y())
            QTest.mouseMove(self.canvas, pt.toPoint())
        for i in range(1, 16):
            pt = QPointF(p20.x(), p20.y() + (p21.y() - p20.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())

        self.assertEqual(
            self.canvas.edges.get(((2, 0), (2, 1))),
            QColor(255, 170, 0),
            "3辺目までオレンジで伸びていません",
        )

        # マウスを頂点 (2,0) へ戻して ((2,0),(2,1)) を巻き戻し、リリースする
        for i in range(1, 16):
            pt = QPointF(p21.x(), p21.y() + (p20.y() - p21.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p20.toPoint(),
        )

        # 巻き戻した ((2,0),(2,1)) は元のシアン色に戻り、新規の2辺のみオレンジで残る
        self.assertEqual(self.canvas.edges.get(((2, 0), (2, 1))), cyan)
        self.assertIn(((0, 0), (1, 0)), self.canvas.edges)
        self.assertIn(((1, 0), (2, 0)), self.canvas.edges)
        self.assertEqual(len(self.canvas.edges), 3)

        # 空盤面からの通常巻き戻しも検証するため Undo でドラッグ前の状態 (シアン1辺のみ) へ1回で戻ることを確認
        self.canvas.undo()
        self.assertEqual(
            set(self.canvas.edges.keys()),
            {((2, 0), (2, 1))},
            "Undo 1回でドラッグ開始前の辺集合に戻っていません",
        )
        self.assertEqual(self.canvas.edges[((2, 0), (2, 1))], cyan)

        # シアン辺も Undo して完全に空にしてから、未塗り辺の巻き戻しで蘇らないことを確認
        self.canvas.undo()
        self.assertEqual(len(self.canvas.edges), 0)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_00_10.toPoint(),
        )
        for i in range(1, 16):
            pt = QPointF(p10.x() + (p20.x() - p10.x()) * i / 15.0, p10.y())
            QTest.mouseMove(self.canvas, pt.toPoint())
        for i in range(1, 16):
            pt = QPointF(p20.x(), p20.y() + (p21.y() - p20.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        self.assertIn(((2, 0), (2, 1)), self.canvas.edges)

        # 頂点 (2,0) へ巻き戻して消去しリリース
        for i in range(1, 16):
            pt = QPointF(p21.x(), p21.y() + (p20.y() - p21.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p20.toPoint(),
        )
        self.assertNotIn(((2, 0), (2, 1)), self.canvas.edges)
        self.assertEqual(len(self.canvas.edges), 2)

        # Undo 1回で残り2辺が一括消去され、巻き戻した ((2,0),(2,1)) が蘇らないこと
        self.canvas.undo()
        self.assertEqual(
            len(self.canvas.edges),
            0,
            f"Undo 後に巻き戻した辺や残骸が残っています: {list(self.canvas.edges)}",
        )
        self.canvas.undo()
        self.assertEqual(len(self.canvas.edges), 0)

    def test_grid_adjust_undo_restores_grid_rect(self):
        """枠調整 (GRID_ADJUST) のドラッグ変更が1操作ごとに Undo で元の枠へ戻ること。"""
        self.canvas.set_margin_percent(0.25)
        self.canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        r0 = list(self.canvas.grid_rect)

        # 1操作目: 左上角ドラッグ
        tl = self.canvas.vertex_to_widget_coords(0, 0)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            tl.toPoint(),
        )
        QTest.mouseMove(self.canvas, QPointF(tl.x() - 50, tl.y() - 40).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(tl.x() - 50, tl.y() - 40).toPoint(),
        )
        r1 = list(self.canvas.grid_rect)
        self.assertNotEqual(r1, r0)

        # 2操作目: 内側ドラッグ全体移動
        c = self.canvas.vertex_to_widget_coords(2, 2)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            c.toPoint(),
        )
        QTest.mouseMove(self.canvas, QPointF(c.x() + 25, c.y() + 15).toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(c.x() + 25, c.y() + 15).toPoint(),
        )
        r2 = list(self.canvas.grid_rect)
        self.assertNotEqual(r2, r1)

        # Undo 1回目 -> r1 に戻る
        self.canvas.undo()
        np.testing.assert_allclose(self.canvas.grid_rect, r1, atol=1e-6)

        # Undo 2回目 -> r0 に戻る
        self.canvas.undo()
        np.testing.assert_allclose(self.canvas.grid_rect, r0, atol=1e-6)

    def test_mode_move_handlers_reverse_end_and_adjust_cursors(self):
        """モード別移動ハンドラで1辺目の逆側端点からの延長・巻き戻し、および枠調整カーソル更新が機能すること。"""
        # 1. SNAP_LINE: 最初の1辺 ((1,1), (2,1)) をクリック後、左端 (1,1) 側から下方向 (1,2) へ伸ばす
        p11 = self.canvas.vertex_to_widget_coords(1, 1)
        p21 = self.canvas.vertex_to_widget_coords(2, 1)
        p12 = self.canvas.vertex_to_widget_coords(1, 2)
        mid_11_21 = QPointF((p11.x() + p21.x()) / 2.0, (p11.y() + p21.y()) / 2.0)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid_11_21.toPoint(),
        )
        for i in range(1, 16):
            pt = QPointF(p11.x(), p11.y() + (p12.y() - p11.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        self.assertIn(
            ((1, 1), (1, 2)),
            self.canvas.edges,
            "最初の1辺の始点側頂点 (1,1) から隣接辺へ延長できていません",
        )

        # (1,1) へマウスを戻すと ((1,1), (1,2)) が巻き戻されること
        for i in range(1, 16):
            pt = QPointF(p12.x(), p12.y() + (p11.y() - p12.y()) * i / 15.0)
            QTest.mouseMove(self.canvas, pt.toPoint())
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            p11.toPoint(),
        )
        self.assertNotIn(((1, 1), (1, 2)), self.canvas.edges)
        self.assertEqual(set(self.canvas.edges.keys()), {((1, 1), (2, 1))})

        # 2. GRID_ADJUST: 非ドラッグ時のマウス移動で対象部位に応じたカーソル形状へ変化すること
        self.canvas.set_margin_percent(0.20)
        self.canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        handles = self.canvas.grid_handle_positions()
        center_pt = self.canvas.vertex_to_widget_coords(2, 2)

        expected_cursors = [
            (handles["tl"], Qt.CursorShape.SizeFDiagCursor),
            (handles["tr"], Qt.CursorShape.SizeBDiagCursor),
            (handles["t"], Qt.CursorShape.SizeVerCursor),
            (handles["l"], Qt.CursorShape.SizeHorCursor),
            (center_pt, Qt.CursorShape.SizeAllCursor),
            (QPointF(2.0, 2.0), Qt.CursorShape.ArrowCursor),
        ]
        for pt, expected_shape in expected_cursors:
            QTest.mouseMove(self.canvas, pt.toPoint())
            self.assertEqual(
                self.canvas.cursor().shape(),
                expected_shape,
                f"座標 {pt} での枠調整カーソル形状が期待値と異なります",
            )


# ==============================================================================
# 4. MainWindow 統合・スライダー連動・画面キャプチャのテスト
# ==============================================================================


class TestMainWindowAndCapture(unittest.TestCase):
    """MainWindow の画面遷移・UI連動・F4キャプチャ制御のテスト。"""

    @classmethod
    def setUpClass(cls):
        get_qapp()

    def setUp(self):
        self._hotkey_patcher = mock.patch.object(MainWindow, "_register_global_hotkey")
        self._hotkey_patcher.start()
        self.addCleanup(self._hotkey_patcher.stop)
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_main_window_detect_grid_and_execute_warp(self):
        """四隅指定からマス目自動検知・正方形パース補正・サンドボックス遷移まで動作すること。"""
        self.assertIsNotNone(self.window.captured_bgr)
        self.assertEqual(self.window.stack.currentIndex(), 0)

        self.window.perspective_canvas.points = [
            QPointF(220, 110),
            QPointF(780, 70),
            QPointF(890, 580),
            QPointF(150, 610),
        ]
        self.window._on_perspective_points_updated(
            self.window.perspective_canvas.points
        )
        self.assertTrue(self.window.btn_warp.isEnabled())
        self.assertTrue(self.window.btn_detect_grid.isEnabled())

        # マス目検知の実行
        self.window.action_detect_grid()
        self.assertEqual(
            (self.window.slider_cols.value(), self.window.slider_rows.value()),
            (4, 4),
            "マス目検知値が反映されていません",
        )

        # 透視投影変換の実行（デフォルト: 正方形 1:1）
        self.window.action_execute_warp()
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.assertIsNotNone(self.window.warped_bgr)
        h, w, _ = self.window.warped_bgr.shape
        self.assertEqual(h, w, f"補正画像が正方形(1:1)になっていません: {w}x{h}")

    def test_grid_sliders_and_unlock_range_sync(self):
        """行・列スライダー変更で表示ラベル・キャンバス行列数が連動し、制限解除で範囲が切り替わること。"""
        self.assertEqual(
            (self.window.slider_cols.minimum(), self.window.slider_cols.maximum()),
            (3, 8),
        )
        self.window.slider_cols.setValue(5)
        self.assertEqual(
            self.window.lbl_cols_val.text(),
            "5",
            f"列ラベルが更新されていません: {self.window.lbl_cols_val.text()}",
        )
        self.assertEqual(
            self.window.sandbox_canvas.cols,
            5,
            f"キャンバスの列数が更新されていません: {self.window.sandbox_canvas.cols}",
        )

        self.window.slider_rows.setValue(3)
        self.assertEqual(
            self.window.lbl_rows_val.text(),
            "3",
            f"行ラベルが更新されていません: {self.window.lbl_rows_val.text()}",
        )
        self.assertEqual(
            self.window.sandbox_canvas.rows,
            3,
            f"キャンバスの行数が更新されていません: {self.window.sandbox_canvas.rows}",
        )

        # 制限解除チェック ON で 1〜20、OFF で 3〜8 に切り替わること
        self.window.chk_unlock_size.setChecked(True)
        self.assertEqual(
            (self.window.slider_cols.minimum(), self.window.slider_cols.maximum()),
            (1, 20),
        )
        self.window.slider_cols.setValue(12)
        self.assertEqual(self.window.sandbox_canvas.cols, 12)
        self.window.chk_unlock_size.setChecked(False)
        self.assertEqual(
            (self.window.slider_cols.minimum(), self.window.slider_cols.maximum()),
            (3, 8),
        )

    def test_apply_detected_grid_settings_unlocks_and_syncs(self):
        """通常範囲外の検知結果に対して action_detect_grid と action_execute_warp の双方が制限解除・同期すること。"""
        # 1. action_detect_grid で 2x2 (3未満) が返された場合
        self.assertFalse(self.window.chk_unlock_size.isChecked())
        with mock.patch.object(
            vision, "detect_grid_dimensions", return_value=(2, 2, 0.08)
        ):
            self.window.action_detect_grid()
        self.assertTrue(self.window.chk_unlock_size.isChecked())
        self.assertEqual(
            (self.window.slider_cols.value(), self.window.slider_rows.value()), (2, 2)
        )
        self.assertEqual(
            (self.window.sandbox_canvas.cols, self.window.sandbox_canvas.rows), (2, 2)
        )
        self.assertEqual(self.window.slider_margin.value(), 8)

        # 一旦制限解除を戻し、2. action_execute_warp で 10x6 (8超) が返された場合
        self.window.chk_unlock_size.setChecked(False)
        with mock.patch.object(
            vision, "detect_grid_dimensions", return_value=(10, 6, 0.04)
        ):
            self.window.action_execute_warp()
        self.assertTrue(self.window.chk_unlock_size.isChecked())
        self.assertEqual(
            (self.window.slider_cols.value(), self.window.slider_rows.value()), (10, 6)
        )
        self.assertEqual(
            (self.window.lbl_cols_val.text(), self.window.lbl_rows_val.text()),
            ("10", "6"),
        )
        self.assertEqual(
            (self.window.sandbox_canvas.cols, self.window.sandbox_canvas.rows), (10, 6)
        )
        self.assertAlmostEqual(self.window.sandbox_canvas.margin_percent, 0.04, places=5)

        # プリセット色・ダイアログ色の共通反映 (_apply_pen_color) とサンプル再読込時の画面初期化も確認
        self.window.action_set_preset_color("#00FF66")
        self.assertEqual(self.window.sandbox_canvas.current_color, QColor("#00FF66"))
        self.assertIn("#00ff66", self.window.btn_color_indicator.styleSheet().lower())

        with mock.patch.object(
            window.QColorDialog, "getColor", return_value=QColor("#FF3366")
        ):
            self.window.action_choose_color()
        self.assertEqual(self.window.sandbox_canvas.current_color, QColor("#FF3366"))
        self.assertIn("#ff3366", self.window.btn_color_indicator.styleSheet().lower())

        self.window.action_load_sample()
        self.assertEqual(self.window.stack.currentIndex(), 0)
        self.assertFalse(self.window.btn_goto_sandbox.isEnabled())
        self.assertEqual(len(self.window.perspective_canvas.points), 4)

    def test_capture_foreground_window_and_fallback(self):
        """前面ウィンドウ矩形でのキャプチャと、自ウィンドウ・矩形取得失敗時の全画面フォールバック。"""
        grab_calls: list[tuple[int, int, int, int] | None] = []

        def fake_grab(bbox=None, all_screens=False, **kw):
            grab_calls.append(bbox)
            w, h = (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else (1920, 1080)
            return Image.new("RGB", (w, h), (10, 20, 30))

        # 1. 前面ウィンドウの矩形が取得できる場合: bbox 指定で撮影
        with (
            mock.patch.object(capture.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(
                capture, "get_window_rect", return_value=(100, 50, 740, 530)
            ),
        ):
            img = capture.capture_foreground_window_bgr(
                target_hwnd=1234, exclude_hwnd=9999
            )
        self.assertEqual(
            grab_calls[-1],
            (100, 50, 740, 530),
            f"前面ウィンドウ矩形で撮影されていません: {grab_calls[-1]}",
        )
        self.assertEqual(
            img.shape, (480, 640, 3), f"撮影サイズが不正です: {img.shape}"
        )

        # 2. 対象が自アプリ自身の場合は全画面へフォールバック
        with (
            mock.patch.object(capture.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(
                capture, "get_window_rect", return_value=(100, 50, 740, 530)
            ),
        ):
            capture.capture_foreground_window_bgr(
                target_hwnd=9999, exclude_hwnd=9999
            )
        self.assertIsNone(
            grab_calls[-1], "自アプリが対象のときは全画面フォールバックになるべきです"
        )

        # 3. ウィンドウ矩形が取得できない場合は全画面へフォールバック
        with (
            mock.patch.object(capture.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(capture, "get_window_rect", return_value=None),
        ):
            capture.capture_foreground_window_bgr(
                target_hwnd=1234, exclude_hwnd=9999
            )
        self.assertIsNone(
            grab_calls[-1], "矩形取得失敗時は全画面フォールバックになるべきです"
        )

    def test_action_capture_screen_hides_and_restores_window(self):
        """撮影実行中に自ウィンドウが一時非表示になり、撮影後に再表示・画像反映されること。"""
        self.window.show()
        state: dict[str, object] = {}

        def fake_capture(target_hwnd=None, exclude_hwnd=None):
            state["visible_during_capture"] = self.window.isVisible()
            state["target"] = target_hwnd
            return np.zeros((300, 400, 3), dtype=np.uint8)

        with mock.patch.object(
            capture, "capture_foreground_window_bgr", side_effect=fake_capture
        ):
            self.window.action_capture_screen()

        self.assertIs(
            state.get("visible_during_capture"),
            False,
            "撮影中に本アプリが表示されたままです (写り込みの原因)",
        )
        self.assertTrue(
            self.window.isVisible(), "撮影後に本アプリが再表示されていません"
        )
        self.assertEqual(
            self.window.captured_bgr.shape,
            (300, 400, 3),
            "キャプチャ画像が反映されていません",
        )

    def test_clear_button_and_margin_slider_undo_after_grid_adjust(self):
        """MainWindow の全消去ボタンおよび枠調整後のマージンスライダー変更が Undo で復元できること。"""
        self.window.show()
        self.window.resize(900, 700)
        self.window.action_execute_warp()
        canvas = self.window.sandbox_canvas
        self.assertEqual(len(canvas.undo_stack), 0, "ワープ直後の履歴は空であるべきです")

        # 1. 枠調整で非対称な外枠を設定
        canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        tl = canvas.vertex_to_widget_coords(0, 0)
        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            tl.toPoint(),
        )
        QTest.mouseMove(canvas, QPointF(tl.x() + 30, tl.y() + 15).toPoint())
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointF(tl.x() + 30, tl.y() + 15).toPoint(),
        )
        custom_rect = list(canvas.grid_rect)
        cl, ct, cr, cb = custom_rect
        expected_avg = int(round(((cl + ct + (1.0 - cr) + (1.0 - cb)) / 4.0) * 100))
        self.assertEqual(
            self.window.slider_margin.value(),
            expected_avg,
            "枠調整後の4辺平均マージンがスライダー表示に同期されていません",
        )

        # 2a. マージンスライダーをドラッグ操作した後に離す -> 均等マージンへ変化するが1操作として Undo 可能
        self.window.slider_margin.sliderPressed.emit()
        self.window.slider_margin.setValue(18)
        self.window.slider_margin.setValue(22)
        self.window.slider_margin.sliderReleased.emit()
        np.testing.assert_allclose(canvas.grid_rect, [0.22, 0.22, 0.78, 0.78], atol=1e-6)

        # 2b. ドラッグを伴わない単発スライダー変更 (クリックジャンプ・キー操作相当)
        self.window.slider_margin.setValue(26)
        np.testing.assert_allclose(canvas.grid_rect, [0.26, 0.26, 0.74, 0.74], atol=1e-6)

        # Undo 1回目 -> 単発変更前の 22% に戻りスライダー表示も 22 に同期
        canvas.undo()
        np.testing.assert_allclose(canvas.grid_rect, [0.22, 0.22, 0.78, 0.78], atol=1e-6)
        self.assertEqual(self.window.slider_margin.value(), 22)

        # Undo 2回目 -> スライダー操作前の個別枠調整状態 custom_rect と平均スライダー表示に完全復元
        canvas.undo()
        np.testing.assert_allclose(
            canvas.grid_rect,
            custom_rect,
            atol=1e-6,
            err_msg="マージンスライダー操作後の Undo で個別枠調整が復元されていません",
        )
        self.assertEqual(self.window.slider_margin.value(), expected_avg)

        # 3. 辺を描画して MainWindow の「全消去」ボタンをクリック -> Undo で復元される
        canvas.set_draw_mode(DrawMode.SNAP_LINE)
        p00 = canvas.vertex_to_widget_coords(0, 0)
        p10 = canvas.vertex_to_widget_coords(1, 0)
        mid = QPointF((p00.x() + p10.x()) / 2.0, (p00.y() + p10.y()) / 2.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mid.toPoint(),
        )
        self.assertIn(((0, 0), (1, 0)), canvas.edges)

        self.window.btn_clear.click()
        self.assertEqual(len(canvas.edges), 0)
        canvas.undo()
        self.assertIn(
            ((0, 0), (1, 0)),
            canvas.edges,
            "全消去ボタンクリック後に Undo で辺が復元されていません",
        )

    def test_action_capture_screen_non_windows_fallback(self):
        """ctypes.windll が存在しない非 Windows 環境でもエラーダイアログを出さず全画面フォールバックすること。"""
        self.window.show()
        fake_full_shot = Image.new("RGB", (640, 360), (40, 80, 120))

        with (
            mock.patch.object(capture, "ctypes", new=mock.MagicMock(spec=[])),
            mock.patch.object(capture.ImageGrab, "grab", return_value=fake_full_shot) as mock_grab,
            mock.patch.object(window.QMessageBox, "critical") as mock_crit,
        ):
            self.window.action_capture_screen()

        mock_crit.assert_not_called()
        mock_grab.assert_called_once()
        self.assertEqual(self.window.captured_bgr.shape, (360, 640, 3))

    def test_load_image_bgr_validation_and_action_open_file_error_handling(self):
        """日本語パス読込、空/破損/ファイルサイズ超過/解像度超過 (Image Bomb) の検証と例外保護。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 日本語ファイル名の正常画像読込
            jp_path = os.path.join(tmpdir, "サンプル盤面_テスト.png")
            Image.new("RGB", (120, 90), (10, 100, 200)).save(jp_path, format="PNG")
            loaded_bgr = load_image_bgr(jp_path)
            self.assertEqual(loaded_bgr.shape, (90, 120, 3))

            # 2. 存在しないファイル -> OSError
            with self.assertRaises(OSError):
                load_image_bgr(os.path.join(tmpdir, "not_found.png"))

            # 3. 空ファイル (0 byte) -> ValueError
            empty_path = os.path.join(tmpdir, "empty.png")
            with open(empty_path, "wb"):
                pass
            with self.assertRaises(ValueError):
                load_image_bgr(empty_path)

            # 4. 壊れた画像バイト列 -> ValueError (with クローズにより即座に削除可能であることも確認)
            corrupt_path = os.path.join(tmpdir, "corrupt.png")
            with open(corrupt_path, "wb") as f:
                f.write(b"not-an-image-payload")
            with self.assertRaises(ValueError):
                load_image_bgr(corrupt_path)
            os.remove(corrupt_path)

            # 5. ファイルサイズ上限超過 -> ValueError
            with self.assertRaises(ValueError):
                load_image_bgr(jp_path, max_file_bytes=10)

            # 6. 展開前解像度上限超過 (Image Bomb 対策: cv2.imdecode が呼ばれる前に拒否)
            with mock.patch.object(vision.cv2, "imdecode") as mock_imdecode:
                with self.assertRaises(ValueError):
                    load_image_bgr(jp_path, max_side_px=50)
                with self.assertRaises(ValueError):
                    load_image_bgr(jp_path, max_pixels=1000)
                mock_imdecode.assert_not_called()

            # 7. MainWindow.action_open_file での例外捕捉と警告ダイアログ表示
            with (
                mock.patch.object(
                    window.QFileDialog,
                    "getOpenFileName",
                    return_value=(empty_path, "画像ファイル (*.png)"),
                ),
                mock.patch.object(window.QMessageBox, "warning") as mock_warn,
            ):
                self.window.action_open_file()
            mock_warn.assert_called_once()

    def test_export_image_native_resolution_without_overlays_and_save_extension(self):
        """保存画像が元解像度・黒帯なし・一時UI（枠調整ハンドル/ホバー）なしで出力され拡張子も補完されること。"""
        self.window.action_execute_warp()
        canvas = self.window.sandbox_canvas
        native_w = canvas.pixmap.width()
        native_h = canvas.pixmap.height()

        # あえて小さい横長サイズにして左右に大きな黒帯余白 (25, 25, 28) を発生させる
        canvas.resize(500, 260)
        canvas.show()

        # 辺塗りとセル塗りを1つずつ入れ、さらに枠調整モード (オレンジ枠+ハンドル表示) & ホバー状態にする
        canvas.apply_edge(((0, 0), (1, 0)), "paint")
        canvas.cell_fills[(1, 1)] = QColor(0, 208, 255)
        canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        canvas.hovered_edge = ((2, 0), (3, 0))

        exported = canvas.render_export_image()
        self.assertFalse(exported.isNull())
        self.assertEqual(
            (exported.width(), exported.height()),
            (native_w, native_h),
            "ウィジェットサイズではなく補正後画像の元解像度で出力されるべきです",
        )

        # 左上端ピクセルがウィジェット余白の黒帯色 (25, 25, 28) ではなく盤面色であること
        corner_col = exported.pixelColor(2, 2)
        self.assertNotEqual(
            (corner_col.red(), corner_col.green(), corner_col.blue()),
            (25, 25, 28),
            "ウィジェット余白の黒帯が混入しています",
        )

        # 右下ハンドル位置に枠調整用のオレンジ塗り (255, 170, 0) が焼き込まれていないこと
        br_img_pt = canvas.vertex_to_image_coords(canvas.cols, canvas.rows)
        br_col = exported.pixelColor(int(br_img_pt.x()), int(br_img_pt.y()))
        self.assertNotEqual(
            (br_col.red(), br_col.green(), br_col.blue()),
            (255, 170, 0),
            "枠調整ハンドルが保存画像に写り込んでいます",
        )

        # SNAP_LINE モードでホバー中の辺があっても、ホバーなし時と保存画像が完全一致すること
        canvas.set_draw_mode(DrawMode.SNAP_LINE)
        canvas.hovered_edge = None
        exported_no_hover = canvas.render_export_image()
        canvas.hovered_edge = ((1, 0), (2, 0))
        exported_with_hover = canvas.render_export_image()
        p10 = canvas.vertex_to_image_coords(1, 0)
        p20 = canvas.vertex_to_image_coords(2, 0)
        mid_x, mid_y = int((p10.x() + p20.x()) / 2.0), int(p10.y())
        self.assertEqual(
            exported_with_hover.pixelColor(mid_x, mid_y),
            exported_no_hover.pixelColor(mid_x, mid_y),
            "ホバーハイライトが保存画像に写り込んでいます",
        )

        # 拡張子補完と実ファイル保存の検証 (日本語ディレクトリ・拡張子省略時)
        self.assertEqual(
            normalize_save_image_path("result_img", "PNG画像 (*.png)"),
            "result_img.png",
        )
        self.assertEqual(
            normalize_save_image_path("result_img", "JPEG画像 (*.jpg *.jpeg)"),
            "result_img.jpg",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_save_base = os.path.join(tmpdir, "考察結果_出力")
            with (
                mock.patch.object(
                    window.QFileDialog,
                    "getSaveFileName",
                    return_value=(raw_save_base, "PNG画像 (*.png)"),
                ),
                mock.patch.object(window.QMessageBox, "information") as mock_info,
            ):
                self.window.action_save_sandbox_image()

            expected_saved = raw_save_base + ".png"
            self.assertTrue(os.path.isfile(expected_saved), "拡張子補完されたPNGが保存されていません")
            mock_info.assert_called_once()
            saved_img = load_image_bgr(expected_saved)
            self.assertEqual(saved_img.shape, (native_h, native_w, 3))

    def test_win32_register_hotkey_lifecycle_and_native_event(self):
        """keyboard 非依存で Win32 RegisterHotKey 登録・TopMost再登録・WM_HOTKEY受信・解除が機能すること。"""
        import pathlib
        import re
        from PyQt6 import sip
        from PyQt6.QtCore import QByteArray

        # 1. requirements.txt と pyproject.toml から keyboard が削除され完全一致していること (Python 3.10 互換フォールバック付き)
        root = pathlib.Path(__file__).resolve().parent
        reqs = [
            line.strip()
            for line in (root / "requirements.txt").read_text("utf-8").splitlines()
            if line.strip()
        ]
        pyproject_text = (root / "pyproject.toml").read_text("utf-8")
        try:
            import tomllib  # Python 3.11+
            pyproject_deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
        except ModuleNotFoundError:
            dep_match = re.search(
                r"dependencies\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL
            )
            self.assertIsNotNone(dep_match, "pyproject.toml に dependencies が見つかりません")
            pyproject_deps = re.findall(r'"([^"]+)"', dep_match.group(1))  # type: ignore[union-attr]
        self.assertEqual(pyproject_deps, reqs)
        self.assertTrue(all("keyboard" not in dep.lower() for dep in reqs))

        # setUp で当てている _register_global_hotkey のモックを一時解除して実メソッドを検証
        self._hotkey_patcher.stop()
        try:
            # 非 Windows 環境 (wintypes=None または windll 不在) では例外なく False を返すこと
            with mock.patch.object(capture, "wintypes", None):
                self.assertFalse(self.window._register_global_hotkey())
            with mock.patch.object(capture, "ctypes", new=mock.MagicMock(spec=[])):
                self.assertFalse(self.window._register_global_hotkey())

            fake_user32 = mock.MagicMock()
            # 1回目は MOD_NOREPEAT 成功、次は失敗->MOD_NONE フォールバック成功を検証
            fake_user32.RegisterHotKey.side_effect = [1, 0, 1, 0, 0]
            fake_user32.UnregisterHotKey.return_value = 1
            fake_windll = mock.MagicMock()
            fake_windll.user32 = fake_user32
            effective_wintypes = (
                capture.wintypes
                if capture.wintypes is not None
                else mock.MagicMock(HWND=lambda h: h)
            )

            # keyboard パッケージが存在しない環境でも動作することを強制確認
            with (
                mock.patch.dict(sys.modules, {"keyboard": None}),
                mock.patch.object(capture.ctypes, "windll", fake_windll, create=True),
                mock.patch.object(capture, "wintypes", effective_wintypes),
            ):
                # 登録成功 (MOD_NOREPEAT)
                self.assertTrue(self.window._register_global_hotkey())
                self.assertIsNotNone(self.window._registered_hotkey_hwnd)
                first_call_args = fake_user32.RegisterHotKey.call_args[0]
                self.assertEqual(first_call_args[1], capture.HOTKEY_ID_F4)
                self.assertEqual(
                    first_call_args[2], capture.MOD_NONE | capture.MOD_NOREPEAT
                )
                self.assertEqual(first_call_args[3], capture.VK_F4)

                # TopMost 切替時: いったん UnregisterHotKey され、MOD_NONE フォールバックで再登録される
                self.window.action_toggle_topmost(True)
                self.assertGreaterEqual(fake_user32.UnregisterHotKey.call_count, 1)
                self.assertIsNotNone(self.window._registered_hotkey_hwnd)

                # 他アプリ競合時 (RegisterHotKey が両方 0 を返す場合): 例外なく False を返す
                self.assertFalse(self.window._register_global_hotkey())
                self.assertIsNone(self.window._registered_hotkey_hwnd)

                # 再登録状態にして close() すると確実に UnregisterHotKey が呼ばれ、遅延タイマー再登録も抑止される
                fake_user32.RegisterHotKey.side_effect = [1]
                self.assertTrue(self.window._register_global_hotkey())
                unreg_before_close = fake_user32.UnregisterHotKey.call_count
                self.window.close()
                self.assertEqual(
                    fake_user32.UnregisterHotKey.call_count, unreg_before_close + 1
                )
                self.assertFalse(self.window._hotkey_init_timer.isActive())
                self.assertFalse(self.window._register_global_hotkey())

                # activate_window_native: hwnd=0 は no-op、有効 hwnd は ShowWindow/SetForegroundWindow を呼び例外も吸収
                capture.activate_window_native(0)
                fake_user32.ShowWindow.assert_not_called()
                capture.activate_window_native(4321)
                fake_user32.ShowWindow.assert_called_once_with(4321, 1)
                fake_user32.SetForegroundWindow.assert_called_once_with(4321)
                fake_user32.SetForegroundWindow.side_effect = RuntimeError("denied")
                capture.activate_window_native(4321)  # 例外が外部へ漏れないこと

            # 非 Windows 環境での activate_window_native も安全に no-op となること
            with mock.patch.object(capture, "ctypes", new=mock.MagicMock(spec=[])):
                capture.activate_window_native(4321)

            # 2. nativeEvent で WM_HOTKEY (HOTKEY_ID_F4) を受信すると capture_requested が発火すること
            if capture.wintypes is not None:
                triggered: list[bool] = []
                self.window.hotkey_signaler.capture_requested.disconnect()
                self.window.hotkey_signaler.capture_requested.connect(
                    lambda: triggered.append(True)
                )

                # 対象外のホットキー ID は無視されて基底へ委譲される
                other_msg = capture.wintypes.MSG()
                other_msg.message = capture.WM_HOTKEY
                other_msg.wParam = 9999
                handled, _ = self.window.nativeEvent(
                    QByteArray(b"windows_generic_MSG"),
                    sip.voidptr(capture.ctypes.addressof(other_msg)),
                )
                QApplication.processEvents()
                self.assertFalse(handled)
                self.assertEqual(len(triggered), 0)

                # HOTKEY_ID_F4 の WM_HOTKEY で (True, 0) を返し capture_requested が発火する
                f4_msg = capture.wintypes.MSG()
                f4_msg.message = capture.WM_HOTKEY
                f4_msg.wParam = capture.HOTKEY_ID_F4
                handled, result = self.window.nativeEvent(
                    QByteArray(b"windows_generic_MSG"),
                    sip.voidptr(capture.ctypes.addressof(f4_msg)),
                )
                QApplication.processEvents()
                self.assertTrue(handled)
                self.assertEqual(result, 0)
                self.assertEqual(len(triggered), 1, "WM_HOTKEY でキャプチャ要求が発火していません")
        finally:
            self._hotkey_patcher.start()


# ==============================================================================
# 5. モジュール分割アーキテクチャ・コード衛生・ドキュメント同期の回帰テスト
# ==============================================================================


class TestCodeHygieneAndDocsSync(unittest.TestCase):
    """モジュール責務分離・未使用インポート不在・旧名称削除・ドキュメント同期を検証するテスト。"""

    def test_module_decoupling_and_entrypoint_reexports(self):
        """vision / capture が PyQt6 非依存に分離され、witness_sandbox からの再エクスポートが一致すること。"""
        import ast
        import pathlib
        import subprocess

        root = pathlib.Path(__file__).resolve().parent
        expected_modules = ("__init__.py", "vision.py", "capture.py", "canvas.py", "window.py")
        for mod_file in expected_modules:
            self.assertTrue(
                (root / "app" / mod_file).is_file(),
                f"app/{mod_file} が存在しません",
            )

        # 1. 別プロセスで app.vision と app.capture のみを import した際、PyQt6 が一切ロードされないこと
        check_code = (
            "import sys; import app.vision, app.capture; "
            "qt_mods = [m for m in sys.modules if m.startswith('PyQt6')]; "
            "assert not qt_mods, f'PyQt6 leaked into vision/capture: {qt_mods}'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", check_code],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"vision/capture の PyQt6 疎結合検証に失敗しました: {proc.stderr}",
        )

        # 2. window.py が ctypes / cv2 に直接依存せず capture / vision / canvas に委譲していること
        window_tree = ast.parse((root / "app" / "window.py").read_text("utf-8"))
        window_imports = {
            alias.name
            for node in ast.walk(window_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(window_tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("ctypes", window_imports)
        self.assertNotIn("cv2", window_imports)

        # 3. witness_sandbox.py がエントリポイントに徹し、主要クラス・関数を同一実体で再エクスポートしていること
        entry_tree = ast.parse((root / "witness_sandbox.py").read_text("utf-8"))
        top_defs = [
            node.name
            for node in entry_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        self.assertEqual(
            top_defs,
            [],
            f"witness_sandbox.py に重複定義が残っています: {top_defs}",
        )

        self.assertIs(witness_sandbox.MainWindow, window.MainWindow)
        self.assertIs(witness_sandbox.main, window.main)
        self.assertIs(witness_sandbox.ImageCanvasBase, canvas.ImageCanvasBase)
        self.assertIs(witness_sandbox.PerspectiveCanvas, canvas.PerspectiveCanvas)
        self.assertIs(witness_sandbox.SandboxCanvas, canvas.SandboxCanvas)
        self.assertIs(witness_sandbox.order_points, vision.order_points)
        self.assertIs(
            witness_sandbox.capture_foreground_window_bgr,
            capture.capture_foreground_window_bgr,
        )
        for exported_name in witness_sandbox.__all__:
            self.assertTrue(
                hasattr(witness_sandbox, exported_name),
                f"__all__ の '{exported_name}' が witness_sandbox に存在しません",
            )

    def test_no_unused_imports_in_python_sources(self):
        """リポジトリ内の全実装・テスト Python ファイルに未使用の import が存在しないこと。"""
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent
        py_files = [
            root / "witness_sandbox.py",
            root / "test_witness_sandbox.py",
            *sorted((root / "app").glob("*.py")),
        ]

        for py_path in py_files:
            source = py_path.read_text("utf-8")
            tree = ast.parse(source, filename=str(py_path))

            # モジュールトップレベル (直下およびトップレベル try ブロック内) の import シンボルを検査
            exported_all: set[str] = set()
            top_stmts: list[ast.stmt] = []
            for node in tree.body:
                top_stmts.append(node)
                if isinstance(node, ast.Try):
                    top_stmts.extend(node.body)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        exported_all.add(elt.value)

            imported_names: dict[str, int] = {}
            for node in top_stmts:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        bound = alias.asname or alias.name.split(".")[0]
                        imported_names[bound] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "__future__":
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        bound = alias.asname or alias.name
                        imported_names[bound] = node.lineno

            used_names: set[str] = set(exported_all)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)

            unused = {
                name: line
                for name, line in imported_names.items()
                if name not in used_names
            }
            self.assertEqual(
                unused,
                {},
                f"{py_path.name} に未使用のインポートが残っています: {unused}",
            )

    def test_dead_code_and_legacy_identifiers_removed(self):
        """削除対象のデッドコード・旧 SpinBox 名残・旧モジュール別名が残存していないこと。"""
        import pathlib

        self.assertFalse(hasattr(SandboxCanvas, "find_nearest_vertex"))
        self.assertFalse(hasattr(MainWindow, "_on_grid_size_changed"))
        self.assertFalse(hasattr(MainWindow, "spin_cols"))
        self.assertFalse(hasattr(MainWindow, "spin_rows"))
        self.assertNotIn("QSpinBox", DARK_STYLE_QSS)

        test_src = pathlib.Path(__file__).resolve().read_text("utf-8")
        legacy_alias = "main" + "_mod"
        self.assertNotIn(legacy_alias, test_src, "テストコードに旧別名が残っています")

    def test_readme_and_gitignore_synced_with_implementation(self):
        """README.md と .gitignore の記述が現行 UI・プロジェクト構成と同期していること。"""
        import pathlib

        root = pathlib.Path(__file__).resolve().parent
        readme = (root / "README.md").read_text("utf-8")
        gitignore = (root / ".gitignore").read_text("utf-8")

        # 全5種のツール名およびスライダー範囲仕様が README に記載されていること
        for required_term in (
            "線塗り",
            "マス目塗り",
            "自由手書き",
            "消しゴム",
            "枠調整",
            "3〜8",
            "制限解除",
            "1〜20",
        ):
            self.assertIn(required_term, readme, f"README.md に '{required_term}' の記載がありません")

        # 旧仕様・旧メディア表記が残っていないこと
        self.assertNotIn("動画での利用例", readme)
        self.assertNotIn("格子点の近くをクリック", readme)

        # .gitignore から未存在ファイル・未追跡フォルダ指定が除去されていること
        self.assertNotIn("test_warped.png", gitignore)
        self.assertNotIn("!docs/*.png", gitignore)
        self.assertIn("witness_sandbox_solution.png", gitignore)
        self.assertIn("!assets/*.png", gitignore)


if __name__ == "__main__":
    unittest.main(verbosity=2)
