"""
witness_sandbox.py の自動テストスイート。
オフスクリーン環境（QT_QPA_PLATFORM="offscreen"）で各機能・コンポーネントを
項目ごとに独立して検証する。
"""

import os
import sys
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

import witness_sandbox as main_mod
from witness_sandbox import (
    DrawMode,
    MainWindow,
    PerspectiveCanvas,
    SandboxCanvas,
    create_sample_witness_image,
    cv_to_qpixmap,
    detect_grid_dimensions,
    order_points,
    point_to_segment_distance,
    warp_perspective,
)

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
            (self.window.spin_cols.value(), self.window.spin_rows.value()),
            (4, 4),
            "マス目検知値が反映されていません",
        )

        # 透視投影変換の実行（デフォルト: 正方形 1:1）
        self.window.action_execute_warp()
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.assertIsNotNone(self.window.warped_bgr)
        h, w, _ = self.window.warped_bgr.shape
        self.assertEqual(h, w, f"補正画像が正方形(1:1)になっていません: {w}x{h}")

    def test_slider_and_compat_properties_sync(self):
        """行・列スライダー変更で表示ラベル・キャンバス行列数・互換プロパティが連動すること。"""
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
        self.assertEqual(
            self.window.spin_cols.value(), 5, "互換プロパティ spin_cols が一致しません"
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
        self.assertEqual(
            self.window.spin_rows.value(), 3, "互換プロパティ spin_rows が一致しません"
        )

    def test_capture_foreground_window_and_fallback(self):
        """前面ウィンドウ矩形でのキャプチャと、自ウィンドウ・矩形取得失敗時の全画面フォールバック。"""
        grab_calls: list[tuple[int, int, int, int] | None] = []

        def fake_grab(bbox=None, all_screens=False, **kw):
            grab_calls.append(bbox)
            w, h = (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else (1920, 1080)
            return Image.new("RGB", (w, h), (10, 20, 30))

        # 1. 前面ウィンドウの矩形が取得できる場合: bbox 指定で撮影
        with (
            mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(
                main_mod, "get_window_rect", return_value=(100, 50, 740, 530)
            ),
        ):
            img = main_mod.capture_foreground_window_bgr(
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
            mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(
                main_mod, "get_window_rect", return_value=(100, 50, 740, 530)
            ),
        ):
            main_mod.capture_foreground_window_bgr(
                target_hwnd=9999, exclude_hwnd=9999
            )
        self.assertIsNone(
            grab_calls[-1], "自アプリが対象のときは全画面フォールバックになるべきです"
        )

        # 3. ウィンドウ矩形が取得できない場合は全画面へフォールバック
        with (
            mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab),
            mock.patch.object(main_mod, "get_window_rect", return_value=None),
        ):
            main_mod.capture_foreground_window_bgr(
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
            main_mod, "capture_foreground_window_bgr", side_effect=fake_capture
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
