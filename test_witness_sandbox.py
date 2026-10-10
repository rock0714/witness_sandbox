"""
main.py のセルフチェック用自動テストスクリプト
ヘッドレス環境（QApplication [offscreen]）で主要コンポーネントのロジックを検証する。
"""

import sys
import os
import numpy as np
import cv2

# PyQt6のオフスクリーンモード設定
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor

# witness_sandbox.py のインポート
from witness_sandbox import (
    order_points,
    warp_perspective,
    detect_grid_dimensions,
    point_to_segment_distance,
    create_sample_witness_image,
    cv_to_qpixmap,
    PerspectiveCanvas,
    SandboxCanvas,
    DrawMode,
    MainWindow
)

def run_tests():
    print("=== [セルフチェック開始] ===")
    app = QApplication(sys.argv)

    # 1. サンプル画像生成テスト
    print("1. サンプルパズル画像の生成テスト...")
    sample_img = create_sample_witness_image()
    assert sample_img is not None, "サンプル画像が生成できませんでした"
    assert sample_img.shape == (700, 1000, 3), f"サンプル画像の形状が不正です: {sample_img.shape}"
    print("   -> OK: サンプル画像生成成功 (700x1000)")

    # 2. 四隅点整列ロジックテスト (order_points)
    print("2. 4点ソート (order_points) テスト...")
    raw_pts = np.array([
        [890, 580],  # 右下
        [220, 110],  # 左上
        [150, 610],  # 左下
        [780, 70],   # 右上
    ], dtype="float32")

    ordered = order_points(raw_pts)
    np.testing.assert_allclose(ordered[0], [220, 110], err_msg="TLが一致しません")
    np.testing.assert_allclose(ordered[1], [780, 70], err_msg="TRが一致しません")
    np.testing.assert_allclose(ordered[2], [890, 580], err_msg="BRが一致しません")
    np.testing.assert_allclose(ordered[3], [150, 610], err_msg="BLが一致しません")
    print("   -> OK: 4点整列成功 (TL, TR, BR, BL)")

    # 3. マス目自動検知 (detect_grid_dimensions) テスト
    print("3. マス目自動検知 (detect_grid_dimensions) テスト...")
    det_cols, det_rows, det_margin = detect_grid_dimensions(sample_img, ordered)
    assert det_cols == 4 and det_rows == 4, f"検知されたマス数が不正です: {det_cols}x{det_rows} (期待値: 4x4)"
    assert 0.05 <= det_margin <= 0.15, f"検知されたマージンが不正です: {det_margin}"
    print(f"   -> OK: マス目自動検知成功 ({det_cols}列 x {det_rows}行, マージン: {det_margin*100:.0f}%)")

    # 4. 正方形(1:1)比率透視投影変換 (warp_perspective) テスト
    print("4. 正方形(1:1)比率透視投影変換 (warp_perspective) テスト...")
    warped_sq = warp_perspective(sample_img, ordered, aspect_mode="square")
    assert warped_sq is not None and warped_sq.size > 0
    h_sq, w_sq, _ = warped_sq.shape
    assert h_sq == w_sq, f"正方形比率になっていません: {w_sq}x{h_sq}"
    print(f"   -> OK: 正方形(1:1)透視変換成功 (サイズ: {w_sq}x{h_sq}, 比率 1:1)")

    # グリッド比率テスト
    warped_grid = warp_perspective(sample_img, ordered, aspect_mode="grid", cols=6, rows=3)
    h_gr, w_gr, _ = warped_grid.shape
    assert abs(w_gr / h_gr - 2.0) < 0.05, f"グリッド比率(6:3=2:1)になっていません: {w_gr}x{h_gr}"
    print(f"   -> OK: グリッド比率(6:3)透視変換成功 ({w_gr}x{h_gr})")

    # 5. 点と線分の最短距離ヘルパーテスト
    print("5. 幾何計算 (point_to_segment_distance) テスト...")
    d1 = point_to_segment_distance(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(d1 - 5.0) < 1e-5, f"線分との距離が不正: {d1}"
    d2 = point_to_segment_distance(12.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(d2 - 2.0) < 1e-5, f"端点外の距離が不正: {d2}"
    print("   -> OK: 最短距離計算正常")

    # 6. PerspectiveCanvas のテスト
    print("6. PerspectiveCanvas コンポーネントテスト...")
    p_canvas = PerspectiveCanvas()
    p_canvas.resize(800, 600)
    p_canvas.set_image(sample_img)
    assert not p_canvas.pixmap.isNull()

    p_canvas.points = [
        QPointF(220, 110),
        QPointF(780, 70),
        QPointF(890, 580),
        QPointF(150, 610)
    ]
    assert len(p_canvas.points) == 4
    print("   -> OK: PerspectiveCanvas 動作正常")

    # 7. SandboxCanvas (線塗りトグル、ドラッグ塗り、マス目塗り、消しゴム、Undo) テスト
    print("7. SandboxCanvas テスト...")
    from PyQt6.QtTest import QTest
    s_canvas = SandboxCanvas()
    s_canvas.resize(600, 600)
    s_canvas.show()
    s_canvas.set_warped_image(warped_sq)
    s_canvas.set_grid_dimensions(4, 4)
    s_canvas.set_margin_percent(det_margin)
    s_canvas.set_draw_mode(DrawMode.SNAP_LINE)

    # 辺 (0,0)-(1,0) の中点 (ウィジェット座標)
    a = s_canvas.vertex_to_widget_coords(0, 0)
    b = s_canvas.vertex_to_widget_coords(1, 0)
    mid = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
    key = ((0, 0), (1, 0))
    assert s_canvas.find_nearest_edge(mid) == key, "辺の当たり判定に失敗しました"

    # 実マウスクリックで塗り -> 同じ辺をもう一度クリックで消去
    QTest.mouseClick(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mid.toPoint())
    assert key in s_canvas.edges, "クリックで辺が塗られていません"
    QTest.mouseClick(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mid.toPoint())
    assert key not in s_canvas.edges, "塗り済みの辺を再クリックしても消えません"
    # Undo (消去を取り消し -> 塗られた状態へ、さらに戻して空へ)
    s_canvas.undo()
    assert key in s_canvas.edges, "Undoで消去が復元されていません"
    s_canvas.undo()
    assert key not in s_canvas.edges, "Undoで塗りが取り消されていません"

    # ドラッグ塗り: 上辺 (0,0)->(4,0) を左から右へなぞる
    start = s_canvas.vertex_to_widget_coords(0, 0)
    end = s_canvas.vertex_to_widget_coords(4, 0)
    QTest.mousePress(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(start.x() + 5, start.y()).toPoint())
    for i in range(1, 41):
        x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
        QTest.mouseMove(s_canvas, QPointF(x, start.y()).toPoint())
    QTest.mouseRelease(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(end.x() - 5, end.y()).toPoint())
    top_edges = [((c, 0), (c + 1, 0)) for c in range(4)]
    assert all(e in s_canvas.edges for e in top_edges), f"ドラッグで上辺4本が塗られていません: {list(s_canvas.edges)}"
    assert len(s_canvas.edges) == 4, f"余計な辺が塗られています: {list(s_canvas.edges)}"
    # 塗り済みの辺から開始したドラッグは消去モード
    QTest.mousePress(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(start.x() + 5, start.y()).toPoint())
    for i in range(1, 41):
        x = start.x() + 5 + (end.x() - start.x() - 10) * i / 40.0
        QTest.mouseMove(s_canvas, QPointF(x, start.y()).toPoint())
    QTest.mouseRelease(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(end.x() - 5, end.y()).toPoint())
    assert len(s_canvas.edges) == 0, f"ドラッグ消去できていません: {list(s_canvas.edges)}"
    print("   -> OK: 線塗りトグル・ドラッグ塗り/消去・Undo 正常")

    # 消しゴム機能テスト (erase_at_pos)
    s_canvas.apply_edge(key, "paint")
    cell = (1, 1)
    s_canvas.cell_fills[cell] = QColor(0, 200, 255)
    s_canvas.freehand_strokes.append({
        "points": [QPointF(100, 100), QPointF(120, 120)],
        "color": QColor(255, 255, 255),
        "width": 4
    })
    assert s_canvas.erase_at_pos(mid) is True
    assert key not in s_canvas.edges, "消しゴムで塗り辺が消去されていません"
    assert s_canvas.erase_at_pos(QPointF(110, 110)) is True
    assert len(s_canvas.freehand_strokes) == 0, "消しゴムで手書きストロークが消去されていません"
    cell_wpt = s_canvas.vertex_to_widget_coords(1, 1)
    cell_center = QPointF(cell_wpt.x() + 20, cell_wpt.y() + 20)
    assert s_canvas.erase_at_pos(cell_center) is True
    assert cell not in s_canvas.cell_fills, "消しゴムでセル塗りが消去されていません"
    print("   -> OK: 消しゴム機能 (塗り辺・手書き・セル塗り消去) 正常")

    # 7b. 枠調整 (グリッド外枠の位置・サイズをドラッグで変更) テスト
    print("7b. 枠調整テスト...")
    s_canvas.clear_all(clear_history=True)
    s_canvas.set_margin_percent(0.30)
    s_canvas.set_draw_mode(DrawMode.GRID_ADJUST)
    r0 = list(s_canvas.grid_rect)
    # 左上の角を外側(左上)へドラッグ -> left/top が小さくなり、right/bottom は不変
    tl = s_canvas.vertex_to_widget_coords(0, 0)
    QTest.mousePress(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, tl.toPoint())
    QTest.mouseMove(s_canvas, QPointF(tl.x() - 60, tl.y() - 60).toPoint())
    QTest.mouseRelease(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(tl.x() - 60, tl.y() - 60).toPoint())
    r1 = s_canvas.grid_rect
    assert r1[0] < r0[0] and r1[1] < r0[1], f"角ドラッグで外枠が広がっていません: {r0} -> {r1}"
    assert abs(r1[2] - r0[2]) < 1e-6 and abs(r1[3] - r0[3]) < 1e-6, "反対側の辺が動いてしまいました"
    # 右辺の中点を外側へドラッグ -> rightのみ増加
    handles = s_canvas.grid_handle_positions()
    rp = handles["r"]
    QTest.mousePress(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, rp.toPoint())
    QTest.mouseMove(s_canvas, QPointF(rp.x() + 40, rp.y()).toPoint())
    QTest.mouseRelease(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(rp.x() + 40, rp.y()).toPoint())
    r2 = s_canvas.grid_rect
    assert r2[2] > r1[2] and abs(r2[0] - r1[0]) < 1e-6 and abs(r2[3] - r1[3]) < 1e-6, f"辺ドラッグが不正: {r1} -> {r2}"
    # 内側ドラッグ -> サイズを保ったまま移動
    c = s_canvas.vertex_to_widget_coords(2, 2)
    w0, h0 = r2[2] - r2[0], r2[3] - r2[1]
    QTest.mousePress(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c.toPoint())
    QTest.mouseMove(s_canvas, QPointF(c.x() + 20, c.y() + 10).toPoint())
    QTest.mouseRelease(s_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(c.x() + 20, c.y() + 10).toPoint())
    r3 = s_canvas.grid_rect
    assert abs((r3[2] - r3[0]) - w0) < 1e-6 and abs((r3[3] - r3[1]) - h0) < 1e-6, "全体移動でサイズが変わりました"
    assert r3[0] != r2[0], "全体移動できていません"
    # 枠調整中は線が塗られない
    assert len(s_canvas.edges) == 0
    s_canvas.set_draw_mode(DrawMode.SNAP_LINE)
    print("   -> OK: 枠調整 (角/辺/全体移動) 正常")

    # 8. MainWindow 統合テスト (マス目検知・正方形補正)
    print("8. MainWindow 統合テスト...")
    window = MainWindow()
    assert window.captured_bgr is not None
    assert window.stack.currentIndex() == 0

    # 4隅を設定
    window.perspective_canvas.points = [
        QPointF(220, 110),
        QPointF(780, 70),
        QPointF(890, 580),
        QPointF(150, 610)
    ]
    window._on_perspective_points_updated(window.perspective_canvas.points)
    assert window.btn_warp.isEnabled()
    assert window.btn_detect_grid.isEnabled()

    # 「マス目検知」ボタンの実行
    window.action_detect_grid()
    assert window.spin_cols.value() == 4 and window.spin_rows.value() == 4, "マス目検知値が反映されていません"

    # 補正実行 (デフォルト: 正方形 1:1)
    window.action_execute_warp()
    assert window.stack.currentIndex() == 1
    assert window.warped_bgr is not None
    h, w, _ = window.warped_bgr.shape
    assert h == w, f"補正画像が正方形(1:1)になっていません: {w}x{h}"
    print(f"   -> OK: MainWindow 統合テスト成功 (マス目検知 4x4, 正方形補正 {w}x{h})")

    # 9. F4キャプチャ (前面ウィンドウのみ撮影・自アプリ非表示) のロジックテスト (モック)
    print("9. F4キャプチャ (前面ウィンドウのみ) テスト...")
    from unittest import mock
    from PIL import Image
    import witness_sandbox as main_mod

    grab_calls = []

    def fake_grab(bbox=None, all_screens=False, **kw):
        grab_calls.append(bbox)
        w, h = (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else (1920, 1080)
        return Image.new("RGB", (w, h), (10, 20, 30))

    # 前面ウィンドウの矩形が取れる場合: bbox 指定で撮影される (全画面ではない)
    with mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab), \
         mock.patch.object(main_mod, "get_window_rect", return_value=(100, 50, 740, 530)):
        img = main_mod.capture_foreground_window_bgr(target_hwnd=1234, exclude_hwnd=9999)
    assert grab_calls[-1] == (100, 50, 740, 530), f"前面ウィンドウ矩形で撮影されていません: {grab_calls[-1]}"
    assert img.shape == (480, 640, 3), f"撮影サイズが不正です: {img.shape}"

    # 対象が自アプリ、または矩形が取れない場合は全画面にフォールバック
    with mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab), \
         mock.patch.object(main_mod, "get_window_rect", return_value=(100, 50, 740, 530)):
        main_mod.capture_foreground_window_bgr(target_hwnd=9999, exclude_hwnd=9999)
    assert grab_calls[-1] is None, "自アプリが対象のときは全画面フォールバックになるべきです"
    with mock.patch.object(main_mod.ImageGrab, "grab", side_effect=fake_grab), \
         mock.patch.object(main_mod, "get_window_rect", return_value=None):
        main_mod.capture_foreground_window_bgr(target_hwnd=1234, exclude_hwnd=9999)
    assert grab_calls[-1] is None, "矩形取得失敗時は全画面フォールバックになるべきです"

    # action_capture_screen: 撮影時に自アプリが非表示で、撮影後に再表示される
    cap_window = MainWindow()
    cap_window.show()
    state = {}

    def fake_capture(target_hwnd=None, exclude_hwnd=None):
        state["visible_during_capture"] = cap_window.isVisible()
        state["target"] = target_hwnd
        return np.zeros((300, 400, 3), dtype=np.uint8)

    with mock.patch.object(main_mod, "capture_foreground_window_bgr", side_effect=fake_capture):
        cap_window.action_capture_screen()
    assert state.get("visible_during_capture") is False, "撮影中に本アプリが表示されたままです (写り込みの原因)"
    assert cap_window.isVisible(), "撮影後に本アプリが再表示されていません"
    assert cap_window.captured_bgr.shape == (300, 400, 3), "キャプチャ画像が反映されていません"
    # 10. 追加仕様テスト (add_fuction.txt)
    print("10. 追加仕様テスト (1操作1方向制限、3本目描画、行・列スライダー)...")

    # 10a. 1操作で1方向にしか線が伸びないことのテスト
    test_sb = SandboxCanvas()
    test_sb.resize(600, 600)
    test_sb.show()
    test_sb.set_warped_image(warped_sq)
    test_sb.set_grid_dimensions(4, 4)
    test_sb.set_draw_mode(DrawMode.SNAP_LINE)

    p11 = test_sb.vertex_to_widget_coords(1, 1)
    p21 = test_sb.vertex_to_widget_coords(2, 1)
    p22 = test_sb.vertex_to_widget_coords(2, 2)
    p31 = test_sb.vertex_to_widget_coords(3, 1)
    mid_11_21 = QPointF((p11.x() + p21.x()) / 2.0, (p11.y() + p21.y()) / 2.0)
    mid_21_31 = QPointF((p21.x() + p31.x()) / 2.0, (p21.y() + p31.y()) / 2.0)

    # 辺 ((1,1), (2,1)) の中点を押下して (2,1) から (2,2) へ下方向へドラッグ
    QTest.mousePress(test_sb, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mid_11_21.toPoint())
    assert ((1, 1), (2, 1)) in test_sb.edges
    assert len(test_sb.edges) == 1

    for i in range(1, 11):
        pt = QPointF(p21.x(), p21.y() + (p22.y() - p21.y()) * i / 10.0)
        QTest.mouseMove(test_sb, pt.toPoint())
    assert ((2, 1), (2, 2)) in test_sb.edges
    assert len(test_sb.edges) == 2

    # ここで (2,1) から (3,1) の方向（分岐）にマウスが触れても、先端は (2,2) なので (2,1)-(3,1) は塗られない
    QTest.mouseMove(test_sb, mid_21_31.toPoint())
    assert ((2, 1), (3, 1)) not in test_sb.edges, "1操作中に交点から2方向へ同時に伸びてしまっています"

    # マウスを離す
    QTest.mouseRelease(test_sb, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p22.toPoint())

    # 10b. 別の操作で同一交点に3本目の線（T字路）が引けることのテスト
    # 交点 (2,1) には現在 ((1,1),(2,1)) と ((2,1),(2,2)) の2本が接続している
    # ここに3本目として ((2,1), (3,1)) をクリックして描画
    QTest.mouseClick(test_sb, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mid_21_31.toPoint())
    assert ((2, 1), (3, 1)) in test_sb.edges, "同一交点への3本目の線が描画できませんでした"
    assert len(test_sb.edges) == 3
    print("   -> OK: 1操作1方向制限 & 3本目描画 正常")

    # 10c. 行・列スライダーとラベル連携テスト
    window.slider_cols.setValue(5)
    assert window.lbl_cols_val.text() == "5", f"列ラベルが更新されていません: {window.lbl_cols_val.text()}"
    assert window.sandbox_canvas.cols == 5, f"キャンバスの列数が更新されていません: {window.sandbox_canvas.cols}"
    assert window.spin_cols.value() == 5, "互換プロパティ spin_cols が一致しません"

    window.slider_rows.setValue(3)
    assert window.lbl_rows_val.text() == "3", f"行ラベルが更新されていません: {window.lbl_rows_val.text()}"
    assert window.sandbox_canvas.rows == 3, f"キャンバスの行数が更新されていません: {window.sandbox_canvas.rows}"
    assert window.spin_rows.value() == 3, "互換プロパティ spin_rows が一致しません"
    print("   -> OK: 行・列スライダー & ラベル更新 & キャンバス連動 正常")

    print("\n=== [すべてのセルフチェックに合格しました] ===")

if __name__ == "__main__":
    run_tests()

