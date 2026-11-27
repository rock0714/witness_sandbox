"""
キャンバス共通基底クラスおよび描画キャンバスウィジェット群モジュール。

- `cv_to_qpixmap`: OpenCV BGR 画像から QPixmap への変換
- `ImageCanvasBase`: アスペクト比維持中央表示・画像⇔表示座標変換の共通基底
- `PerspectiveCanvas`: パズル盤面の四隅指定キャンバス
- `DrawMode` / `SandboxCanvas`: 仮想グリッド重畳・線塗り・マス目塗り・自由手書き・消しゴム・枠調整キャンバス
"""

import math

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget

from app.vision import point_to_segment_distance


def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """OpenCVのBGR画像をPyQtのQPixmapに変換する。"""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ==============================================================================
# キャンバス共通基底クラス (ImageCanvasBase)
# ==============================================================================


class ImageCanvasBase(QWidget):
    """
    画像をアスペクト比維持でウィジェット中央に表示し、
    元画像座標系とウィジェット表示座標系の相互変換を提供する共通基底クラス。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_bgr: np.ndarray | None = None
        self.pixmap: QPixmap = QPixmap()

    def get_display_scale_and_offset(self) -> tuple[float, float, float]:
        """ウィジェットサイズに応じた画像のスケーリング倍率と描画オフセットを計算する。"""
        if self.pixmap.isNull() or self.pixmap.width() <= 0 or self.pixmap.height() <= 0:
            return 1.0, 0.0, 0.0
        w_ratio = self.width() / self.pixmap.width()
        h_ratio = self.height() / self.pixmap.height()
        scale = min(w_ratio, h_ratio)
        if scale <= 0:
            return 1.0, 0.0, 0.0
        offset_x = (self.width() - self.pixmap.width() * scale) / 2.0
        offset_y = (self.height() - self.pixmap.height() * scale) / 2.0
        return scale, offset_x, offset_y

    def widget_to_image_coords(self, pt: QPointF) -> QPointF:
        """ウィジェット上の表示座標を元画像内の座標に変換する。"""
        scale, ox, oy = self.get_display_scale_and_offset()
        if scale <= 0:
            return QPointF(pt)
        return QPointF((pt.x() - ox) / scale, (pt.y() - oy) / scale)

    def image_to_widget_coords(self, pt: QPointF) -> QPointF:
        """元画像内の座標をウィジェット上の表示座標に変換する。"""
        scale, ox, oy = self.get_display_scale_and_offset()
        return QPointF(pt.x() * scale + ox, pt.y() * scale + oy)

    def _draw_background_or_placeholder(
        self, painter: QPainter, empty_message: str
    ) -> bool:
        """背景を塗りつぶし、画像未設定時は案内メッセージを描画して False を返す。"""
        painter.fillRect(self.rect(), QColor(25, 25, 28))
        if self.pixmap.isNull():
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, empty_message)
            return False
        return True


# ==============================================================================
# モード1: 四隅指定キャンバス (PerspectiveCanvas)
# ==============================================================================


class PerspectiveCanvas(ImageCanvasBase):
    """
    キャプチャ画像上でパズル盤面の四隅をマウスクリックで指定するキャンバス。
    """

    points_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[QPointF] = []  # 元画像座標系での4点
        self.hovered_point_idx: int = -1
        self.dragged_point_idx: int = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, img_bgr: np.ndarray):
        """表示する画像をセットし、四隅指定をリセットする。"""
        self.image_bgr = img_bgr
        self.pixmap = cv_to_qpixmap(img_bgr)
        self.points.clear()
        self.hovered_point_idx = -1
        self.dragged_point_idx = -1
        self.points_updated.emit(self.points)
        self.update()

    def reset_points(self):
        """選択点を全クリアする。"""
        self.points.clear()
        self.hovered_point_idx = -1
        self.dragged_point_idx = -1
        self.points_updated.emit(self.points)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._draw_background_or_placeholder(
            painter,
            "画像がありません。\n[画面キャプチャ(F4)] または [サンプル画像読み込み] をクリックしてください。",
        ):
            return

        scale, ox, oy = self.get_display_scale_and_offset()
        dest_rect = QRectF(
            ox, oy, self.pixmap.width() * scale, self.pixmap.height() * scale
        )
        painter.drawPixmap(dest_rect.toRect(), self.pixmap)

        # 選択済みの点を接続する多角形の描画
        if len(self.points) > 1:
            poly = QPolygonF([self.image_to_widget_coords(p) for p in self.points])
            if len(self.points) == 4:
                painter.setBrush(QBrush(QColor(0, 200, 255, 45)))
                painter.setPen(QPen(QColor(0, 230, 255), 2.5, Qt.PenStyle.DashLine))
                painter.drawPolygon(poly)
            else:
                painter.setPen(QPen(QColor(0, 230, 255), 2, Qt.PenStyle.DashLine))
                painter.drawPolyline(poly)

        # 各点のマーカー描画
        for idx, pt in enumerate(self.points):
            w_pt = self.image_to_widget_coords(pt)
            is_hover = (idx == self.hovered_point_idx) or (idx == self.dragged_point_idx)

            radius = 12 if is_hover else 9
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(
                QBrush(QColor(255, 80, 80) if is_hover else QColor(0, 200, 255))
            )
            painter.drawEllipse(w_pt, radius, radius)

            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(
                QRectF(w_pt.x() - 15, w_pt.y() - 30, 30, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"P{idx + 1}",
            )

    def mousePressEvent(self, event):
        if self.pixmap.isNull():
            return

        pos = event.position()
        hit_radius = 15
        for idx, pt in enumerate(self.points):
            w_pt = self.image_to_widget_coords(pt)
            if math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y()) <= hit_radius:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragged_point_idx = idx
                    self.update()
                    return
                elif event.button() == Qt.MouseButton.RightButton:
                    self.points.pop(idx)
                    self.points_updated.emit(self.points)
                    self.update()
                    return

        if event.button() == Qt.MouseButton.LeftButton and len(self.points) < 4:
            img_pt = self.widget_to_image_coords(pos)
            img_pt.setX(max(0, min(img_pt.x(), self.pixmap.width() - 1)))
            img_pt.setY(max(0, min(img_pt.y(), self.pixmap.height() - 1)))
            self.points.append(img_pt)
            self.points_updated.emit(self.points)
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.dragged_point_idx >= 0:
            img_pt = self.widget_to_image_coords(pos)
            img_pt.setX(max(0, min(img_pt.x(), self.pixmap.width() - 1)))
            img_pt.setY(max(0, min(img_pt.y(), self.pixmap.height() - 1)))
            self.points[self.dragged_point_idx] = img_pt
            self.points_updated.emit(self.points)
            self.update()
            return

        hit_radius = 15
        new_hover = -1
        for idx, pt in enumerate(self.points):
            w_pt = self.image_to_widget_coords(pt)
            if math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y()) <= hit_radius:
                new_hover = idx
                break
        if new_hover != self.hovered_point_idx:
            self.hovered_point_idx = new_hover
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragged_point_idx >= 0:
            self.dragged_point_idx = -1
            self.update()


# ==============================================================================
# モード2: サンドボックス描画キャンバス (SandboxCanvas)
# ==============================================================================


class DrawMode:
    SNAP_LINE = 0    # グリッドスナップ一筆書き
    CELL_FILL = 1    # マス目塗りつぶし
    FREEHAND = 2     # 自由手書き
    ERASER = 3       # 消しゴム
    GRID_ADJUST = 4  # グリッド外枠の位置・サイズ調整


class SandboxCanvas(ImageCanvasBase):
    """
    パース補正後の画像上に仮想グリッドを重ね、スナップ描画や塗りつぶしを行うキャンバス。
    """

    # グリッド外枠が (ドラッグ等で) 変更されたとき
    grid_rect_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # グリッド設定
        self.rows: int = 4
        self.cols: int = 4
        self.margin_percent: float = 0.05  # 外枠マージン 0.0 ~ 0.3 (均等マージン設定用)
        # グリッド外枠の位置 (画像に対する割合): [left, top, right, bottom]
        self.grid_rect: list[float] = [0.05, 0.05, 0.95, 0.95]
        # 枠調整ドラッグ状態
        self.adjust_flags: tuple[int, int, int, int] | None = None
        self.adjust_start_pos: QPointF | None = None
        self.adjust_start_rect: list[float] | None = None

        # 描画設定
        self.draw_mode: int = DrawMode.SNAP_LINE
        self.current_color: QColor = QColor(255, 170, 0)  # The Witness風オレンジ
        self.line_width: int = 14

        # 描画データ
        # 1. 塗られた辺: {((c1, r1), (c2, r2)): QColor} (キーは頂点の昇順タプル)
        self.edges: dict[tuple[tuple[int, int], tuple[int, int]], QColor] = {}
        self.edge_drag_mode: str | None = None
        self.last_drag_pos: QPointF | None = None
        # 1操作での連続線伸ばし・巻き戻し追跡
        self.drag_tip: tuple[int, int] | None = None
        self.drag_base: tuple[int, int] | None = None
        self.op_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.op_vertex_edges: dict[tuple[int, int], list] = {}
        self.hovered_edge: tuple[tuple[int, int], tuple[int, int]] | None = None

        # 2. セル塗りつぶしデータ: {(col, row): QColor}
        self.cell_fills: dict[tuple[int, int], QColor] = {}

        # 3. フリーハンドストローク: list[dict] ({"points": [QPointF] (元画像座標系), "color": QColor, "width": int})
        self.freehand_strokes: list[dict] = []
        self.active_stroke: list[QPointF] = []

        # 4. 操作履歴 (Undo用) と 1操作トランザクション開始状態
        self.undo_stack: list[dict] = []
        self._tx_start_state: dict | None = None

        self.setMouseTracking(True)

    def set_warped_image(self, warped_bgr: np.ndarray):
        """補正後画像をセットする。"""
        self.image_bgr = warped_bgr
        self.pixmap = cv_to_qpixmap(warped_bgr)
        self.clear_all(clear_history=True)
        self.update()

    def set_grid_dimensions(self, rows: int, cols: int):
        """行数と列数を更新する。"""
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.update()

    def set_margin_percent(self, percent: float, record_undo: bool = False):
        """マージン割合を更新する (0.0 ~ 0.35)。record_undo=True 時は変更前状態を履歴に記録する。"""
        prev_state = (
            self._capture_state()
            if (record_undo and self._tx_start_state is None)
            else None
        )
        self.margin_percent = max(0.0, min(0.35, percent))
        m = self.margin_percent
        self.grid_rect = [m, m, 1.0 - m, 1.0 - m]
        if prev_state is not None and not self._states_equal(prev_state, self._capture_state()):
            self.undo_stack.append({"type": "snapshot", "state": prev_state})
        self.update()

    def set_draw_mode(self, mode: int):
        """描画モードを切り替える。"""
        self.active_stroke.clear()
        self.commit_undo_transaction()
        self.draw_mode = mode
        self.edge_drag_mode = None
        self.adjust_flags = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_color(self, color: QColor):
        """描画色を設定する。"""
        self.current_color = color

    # --------------------------------------------------------------------------
    # Undo スナップショット・トランザクション管理
    # --------------------------------------------------------------------------

    def _capture_state(self) -> dict:
        """現在の描画内容およびグリッド外枠のディープコピーを取得する。"""
        return {
            "edges": {k: QColor(v) for k, v in self.edges.items()},
            "cell_fills": {k: QColor(v) for k, v in self.cell_fills.items()},
            "freehand_strokes": [
                {
                    "points": [QPointF(p) for p in s.get("points", [])],
                    "color": QColor(s["color"]),
                    "width": s["width"],
                }
                for s in self.freehand_strokes
            ],
            "grid_rect": list(self.grid_rect),
        }

    @staticmethod
    def _states_equal(s1: dict, s2: dict) -> bool:
        """2つのスナップショット間に実質的な変化がないかを判定する。"""
        if s1["edges"] != s2["edges"]:
            return False
        if s1["cell_fills"] != s2["cell_fills"]:
            return False
        strokes1 = s1["freehand_strokes"]
        strokes2 = s2["freehand_strokes"]
        if len(strokes1) != len(strokes2):
            return False
        for st1, st2 in zip(strokes1, strokes2):
            if st1["color"] != st2["color"] or st1["width"] != st2["width"]:
                return False
            pts1, pts2 = st1["points"], st2["points"]
            if len(pts1) != len(pts2):
                return False
            for p1, p2 in zip(pts1, pts2):
                if abs(p1.x() - p2.x()) > 1e-6 or abs(p1.y() - p2.y()) > 1e-6:
                    return False
        r1, r2 = s1["grid_rect"], s2["grid_rect"]
        if any(abs(a - b) > 1e-9 for a, b in zip(r1, r2)):
            return False
        return True

    def _restore_state(self, state: dict):
        """スナップショットから描画内容およびグリッド外枠を復元する。"""
        prev_rect = list(self.grid_rect)
        self.edges = {k: QColor(v) for k, v in state["edges"].items()}
        self.cell_fills = {k: QColor(v) for k, v in state["cell_fills"].items()}
        self.freehand_strokes = [
            {
                "points": [QPointF(p) for p in s.get("points", [])],
                "color": QColor(s["color"]),
                "width": s["width"],
            }
            for s in state["freehand_strokes"]
        ]
        new_rect = list(state["grid_rect"])
        self.grid_rect = new_rect
        if any(abs(a - b) > 1e-9 for a, b in zip(prev_rect, new_rect)):
            self.grid_rect_changed.emit()
        self.update()

    def begin_undo_transaction(self):
        """1操作単位の Undo トランザクションを開始する。"""
        if self._tx_start_state is not None:
            self.commit_undo_transaction()
        self._tx_start_state = self._capture_state()

    def commit_undo_transaction(self) -> bool:
        """進行中の Undo トランザクションを確定し、変化があれば1件だけ履歴に記録する。"""
        if self._tx_start_state is None:
            return False
        start_state = self._tx_start_state
        self._tx_start_state = None
        curr_state = self._capture_state()
        if not self._states_equal(start_state, curr_state):
            self.undo_stack.append({"type": "snapshot", "state": start_state})
            return True
        return False

    def clear_all(self, clear_history: bool = False):
        """描画内容を全消去する。clear_history=False 時は消去前状態を Undo 可能に記録する。"""
        self.active_stroke.clear()
        if clear_history:
            self._tx_start_state = None
            self.edges.clear()
            self.cell_fills.clear()
            self.freehand_strokes.clear()
            self.undo_stack.clear()
        else:
            self.commit_undo_transaction()
            has_drawings = bool(self.edges or self.cell_fills or self.freehand_strokes)
            if has_drawings:
                prev_state = self._capture_state()
                self.edges.clear()
                self.cell_fills.clear()
                self.freehand_strokes.clear()
                self.undo_stack.append({"type": "snapshot", "state": prev_state})
        self.update()

    def undo(self):
        """直前の操作を1つ取り消す。"""
        if self._tx_start_state is not None:
            self.active_stroke.clear()
            self.edge_drag_mode = None
            self.adjust_flags = None
            self.commit_undo_transaction()

        if not self.undo_stack:
            return

        action = self.undo_stack.pop()
        action_type = action["type"]

        if action_type == "snapshot":
            self._restore_state(action["state"])
            return
        elif action_type == "edge":
            key = action["edge"]
            prev_color = action["prev_color"]
            if prev_color is None:
                self.edges.pop(key, None)
            else:
                self.edges[key] = prev_color
        elif action_type == "fill":
            key = action["cell"]
            prev_color = action["prev_color"]
            if prev_color is None:
                self.cell_fills.pop(key, None)
            else:
                self.cell_fills[key] = prev_color
        elif action_type == "add_stroke":
            if self.freehand_strokes:
                self.freehand_strokes.pop()
        elif action_type == "remove_stroke":
            idx = action["index"]
            data = action["data"]
            self.freehand_strokes.insert(min(idx, len(self.freehand_strokes)), data)

        self.update()

    # --------------------------------------------------------------------------
    # 辺 (グリッド線分) の塗り・消去
    # --------------------------------------------------------------------------

    @staticmethod
    def make_edge_key(v1: tuple[int, int], v2: tuple[int, int]):
        """2頂点から順序に依存しない辺キーを作る。"""
        return (v1, v2) if v1 <= v2 else (v2, v1)

    def all_edge_keys(self):
        """グリッド上の全ての辺 (水平・垂直) を列挙する。"""
        for r in range(self.rows + 1):
            for c in range(self.cols):
                yield ((c, r), (c + 1, r))
        for c in range(self.cols + 1):
            for r in range(self.rows):
                yield ((c, r), (c, r + 1))

    def edge_hit_threshold(self) -> float:
        """辺の当たり判定距離(px)。マス目サイズに応じて調整する。"""
        p0 = self.vertex_to_widget_coords(0, 0)
        p1 = self.vertex_to_widget_coords(1, 1)
        cell_px = min(abs(p1.x() - p0.x()), abs(p1.y() - p0.y()))
        return max(14.0, min(cell_px * 0.45, 40.0))

    def find_nearest_edge(self, widget_pos: QPointF, threshold_px: float | None = None):
        """指定座標に最も近いグリッド辺のキーを返す (範囲外は None)。"""
        if self.pixmap.isNull():
            return None
        if threshold_px is None:
            threshold_px = self.edge_hit_threshold()
        px, py = widget_pos.x(), widget_pos.y()
        best_key = None
        best_dist = float("inf")
        for key in self.all_edge_keys():
            a = self.vertex_to_widget_coords(*key[0])
            b = self.vertex_to_widget_coords(*key[1])
            d = point_to_segment_distance(px, py, a.x(), a.y(), b.x(), b.y())
            if d < best_dist and d <= threshold_px:
                best_dist = d
                best_key = key
        return best_key

    def apply_edge(self, key, mode: str) -> bool:
        """辺に塗り/消去を適用し、トランザクション外であれば Undo に記録する。"""
        prev = self.edges.get(key)
        if mode == "paint":
            if prev is not None and prev == self.current_color:
                return False
            self.edges[key] = QColor(self.current_color)
        else:
            if prev is None:
                return False
            self.edges.pop(key)
        if self._tx_start_state is None:
            self.undo_stack.append(
                {
                    "type": "edge",
                    "edge": key,
                    "prev_color": QColor(prev) if prev is not None else None,
                }
            )
        self.update()
        return True

    def toggle_edge(self, key) -> str:
        """辺をトグルする。既に同色で塗られていれば消去、それ以外は塗る。戻り値は適用したモード。"""
        mode = "erase" if self.edges.get(key) == self.current_color else "paint"
        self.apply_edge(key, mode)
        return mode

    def find_stroke_at_pos(self, widget_pos: QPointF, threshold_px: float = 12.0) -> int | None:
        """指定されたウィジェット座標にある手書きストロークのインデックスを返す。"""
        px = widget_pos.x()
        py = widget_pos.y()
        for idx in range(len(self.freehand_strokes) - 1, -1, -1):
            stroke = self.freehand_strokes[idx]
            pts = stroke.get("points", [])
            for i in range(len(pts) - 1):
                p1 = self.image_to_widget_coords(pts[i])
                p2 = self.image_to_widget_coords(pts[i + 1])
                dist = point_to_segment_distance(px, py, p1.x(), p1.y(), p2.x(), p2.y())
                if dist <= threshold_px:
                    return idx
        return None

    def delete_stroke_at_index(self, idx: int) -> bool:
        """指定インデックスの手書きストロークを削除し、トランザクション外であればUndoスタックに記録する。"""
        if 0 <= idx < len(self.freehand_strokes):
            data = self.freehand_strokes.pop(idx)
            if self._tx_start_state is None:
                self.undo_stack.append(
                    {
                        "type": "remove_stroke",
                        "index": idx,
                        "data": data,
                    }
                )
            self.update()
            return True
        return False

    def erase_at_pos(self, pos: QPointF) -> bool:
        """消しゴム機能: 指定座標にある塗り辺、セル塗り、または手書きストロークを消去する。"""
        standalone_call = self._tx_start_state is None
        if standalone_call:
            self.begin_undo_transaction()

        erased = False

        edge = self.find_nearest_edge(pos)
        if edge is not None and self.apply_edge(edge, "erase"):
            erased = True

        cell = self.find_cell_at_pos(pos)
        if cell in self.cell_fills:
            self.cell_fills.pop(cell)
            erased = True

        hit_stroke_idx = self.find_stroke_at_pos(pos)
        if hit_stroke_idx is not None:
            self.delete_stroke_at_index(hit_stroke_idx)
            erased = True

        if standalone_call:
            self.commit_undo_transaction()

        if erased:
            self.update()
        return erased

    # --------------------------------------------------------------------------
    # 座標変換・グリッド計算
    # --------------------------------------------------------------------------

    def get_grid_bounds(self) -> tuple[float, float, float, float]:
        """画像座標系におけるグリッドの (min_x, min_y, max_x, max_y) を計算する。"""
        if self.pixmap.isNull():
            return 0, 0, 100, 100
        w = self.pixmap.width()
        h = self.pixmap.height()
        l, t, r, b = self.grid_rect
        return w * l, h * t, w * r, h * b

    def vertex_to_image_coords(self, col: int, row: int) -> QPointF:
        """格子頂点インデックス (col, row) を画像座標に変換する。"""
        min_x, min_y, max_x, max_y = self.get_grid_bounds()
        x = min_x + col * (max_x - min_x) / self.cols
        y = min_y + row * (max_y - min_y) / self.rows
        return QPointF(x, y)

    def vertex_to_widget_coords(self, col: int, row: int) -> QPointF:
        """格子頂点インデックス (col, row) をウィジェット上の表示座標に変換する。"""
        return self.image_to_widget_coords(self.vertex_to_image_coords(col, row))

    # --------------------------------------------------------------------------
    # グリッド外枠の調整 (枠調整ツール)
    # --------------------------------------------------------------------------

    ADJUST_FLAGS = {
        "tl": (1, 1, 0, 0),
        "tr": (0, 1, 1, 0),
        "br": (0, 0, 1, 1),
        "bl": (1, 0, 0, 1),
        "t": (0, 1, 0, 0),
        "b": (0, 0, 0, 1),
        "l": (1, 0, 0, 0),
        "r": (0, 0, 1, 0),
    }

    def grid_handle_positions(self) -> dict[str, QPointF]:
        """枠調整ハンドル (四隅 + 各辺中点) のウィジェット座標を返す。"""
        tl = self.vertex_to_widget_coords(0, 0)
        br = self.vertex_to_widget_coords(self.cols, self.rows)
        cx = (tl.x() + br.x()) / 2.0
        cy = (tl.y() + br.y()) / 2.0
        return {
            "tl": QPointF(tl.x(), tl.y()),
            "tr": QPointF(br.x(), tl.y()),
            "br": QPointF(br.x(), br.y()),
            "bl": QPointF(tl.x(), br.y()),
            "t": QPointF(cx, tl.y()),
            "b": QPointF(cx, br.y()),
            "l": QPointF(tl.x(), cy),
            "r": QPointF(br.x(), cy),
        }

    def find_adjust_target(self, pos: QPointF) -> tuple[int, int, int, int] | None:
        """
        カーソル位置から、動かす対象 (辺/角/全体) のフラグを返す。
        優先順: 角・辺中点のハンドル > 枠線の近く(その辺) > 枠の内側(全体移動)。
        """
        if self.pixmap.isNull():
            return None
        handles = self.grid_handle_positions()
        for name, pt in handles.items():
            if math.hypot(pos.x() - pt.x(), pos.y() - pt.y()) <= 16.0:
                return self.ADJUST_FLAGS[name]

        L, T = handles["tl"].x(), handles["tl"].y()
        R, B = handles["br"].x(), handles["br"].y()
        near = 10.0
        in_x = L - near <= pos.x() <= R + near
        in_y = T - near <= pos.y() <= B + near
        if in_x and abs(pos.y() - T) <= near:
            return self.ADJUST_FLAGS["t"]
        if in_x and abs(pos.y() - B) <= near:
            return self.ADJUST_FLAGS["b"]
        if in_y and abs(pos.x() - L) <= near:
            return self.ADJUST_FLAGS["l"]
        if in_y and abs(pos.x() - R) <= near:
            return self.ADJUST_FLAGS["r"]
        if L < pos.x() < R and T < pos.y() < B:
            return (1, 1, 1, 1)
        return None

    def adjust_cursor_for(self, flags) -> Qt.CursorShape:
        """フラグに応じたカーソル形状。"""
        if flags is None:
            return Qt.CursorShape.ArrowCursor
        l, t, r, b = flags
        if (l, t, r, b) == (1, 1, 1, 1):
            return Qt.CursorShape.SizeAllCursor
        if (l and t) or (r and b):
            return Qt.CursorShape.SizeFDiagCursor
        if (r and t) or (l and b):
            return Qt.CursorShape.SizeBDiagCursor
        if l or r:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def apply_grid_adjust(self, pos: QPointF):
        """ドラッグ量に応じて grid_rect を更新する (最小サイズ 5%、画像内にクランプ)。"""
        if (
            self.adjust_flags is None
            or self.adjust_start_pos is None
            or self.adjust_start_rect is None
        ):
            return
        scale, _, _ = self.get_display_scale_and_offset()
        iw, ih = self.pixmap.width(), self.pixmap.height()
        dx = (pos.x() - self.adjust_start_pos.x()) / scale / iw
        dy = (pos.y() - self.adjust_start_pos.y()) / scale / ih
        l0, t0, r0, b0 = self.adjust_start_rect
        fl, ft, fr, fb = self.adjust_flags
        min_size = 0.05

        if (fl, ft, fr, fb) == (1, 1, 1, 1):
            dx = max(-l0, min(dx, 1.0 - r0))
            dy = max(-t0, min(dy, 1.0 - b0))
            self.grid_rect = [l0 + dx, t0 + dy, r0 + dx, b0 + dy]
        else:
            l, t, r, b = l0, t0, r0, b0
            if fl:
                l = max(0.0, min(l0 + dx, r0 - min_size))
            if fr:
                r = min(1.0, max(r0 + dx, l0 + min_size))
            if ft:
                t = max(0.0, min(t0 + dy, b0 - min_size))
            if fb:
                b = min(1.0, max(b0 + dy, t0 + min_size))
            self.grid_rect = [l, t, r, b]
        self.grid_rect_changed.emit()
        self.update()

    def find_cell_at_pos(self, widget_pos: QPointF) -> tuple[int, int] | None:
        """指定されたウィジェット座標が位置するマス目 (col, row) を検出する。"""
        img_pt = self.widget_to_image_coords(widget_pos)
        img_x, img_y = img_pt.x(), img_pt.y()

        min_x, min_y, max_x, max_y = self.get_grid_bounds()
        if not (min_x <= img_x <= max_x and min_y <= img_y <= max_y):
            return None

        cell_w = (max_x - min_x) / self.cols
        cell_h = (max_y - min_y) / self.rows

        col = int((img_x - min_x) // cell_w)
        row = int((img_y - min_y) // cell_h)

        if 0 <= col < self.cols and 0 <= row < self.rows:
            return (col, row)
        return None

    # --------------------------------------------------------------------------
    # レンダリング
    # --------------------------------------------------------------------------

    def _target_point(self, img_pt: QPointF, scale: float, ox: float, oy: float) -> QPointF:
        """元画像座標を指定スケール・オフセットの描画先座標へ変換する。"""
        return QPointF(img_pt.x() * scale + ox, img_pt.y() * scale + oy)

    def _vertex_to_target(
        self, col: int, row: int, scale: float, ox: float, oy: float
    ) -> QPointF:
        """格子頂点 (col, row) を指定スケール・オフセットの描画先座標へ変換する。"""
        return self._target_point(self.vertex_to_image_coords(col, row), scale, ox, oy)

    def _draw_cell_fills(
        self, painter: QPainter, scale: float, ox: float, oy: float
    ) -> None:
        """半透明のマス目塗りつぶしレイヤーを描画する。"""
        min_x, min_y, max_x, max_y = self.get_grid_bounds()
        cell_w = (max_x - min_x) / self.cols
        cell_h = (max_y - min_y) / self.rows

        for (c, r), col in self.cell_fills.items():
            cx = (min_x + c * cell_w) * scale + ox
            cy = (min_y + r * cell_h) * scale + oy
            fill_rect = QRectF(cx, cy, cell_w * scale, cell_h * scale)

            alpha_color = QColor(col.red(), col.green(), col.blue(), 110)
            painter.fillRect(fill_rect, alpha_color)

            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 180), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(fill_rect)

    def _draw_virtual_grid(
        self, painter: QPainter, scale: float, ox: float, oy: float
    ) -> None:
        """仮想グリッド線・外枠・格子頂点ドットを描画する。"""
        grid_pen = QPen(QColor(255, 255, 255, 80), 1.5, Qt.PenStyle.DashLine)
        outer_pen = QPen(QColor(255, 255, 255, 140), 2.5)

        painter.setPen(grid_pen)
        for r in range(self.rows + 1):
            painter.drawLine(
                self._vertex_to_target(0, r, scale, ox, oy),
                self._vertex_to_target(self.cols, r, scale, ox, oy),
            )
        for c in range(self.cols + 1):
            painter.drawLine(
                self._vertex_to_target(c, 0, scale, ox, oy),
                self._vertex_to_target(c, self.rows, scale, ox, oy),
            )

        painter.setPen(outer_pen)
        tl = self._vertex_to_target(0, 0, scale, ox, oy)
        br = self._vertex_to_target(self.cols, self.rows, scale, ox, oy)
        painter.drawRect(QRectF(tl, br))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
        dot_r = 3.5
        for c in range(self.cols + 1):
            for r in range(self.rows + 1):
                painter.drawEllipse(
                    self._vertex_to_target(c, r, scale, ox, oy), dot_r, dot_r
                )

    def _draw_freehand_strokes(
        self,
        painter: QPainter,
        scale: float,
        ox: float,
        oy: float,
        include_active_stroke: bool,
    ) -> None:
        """自由手書きストローク（元画像座標系・固定線幅）を描画する。"""
        all_strokes = list(self.freehand_strokes)
        if include_active_stroke and self.active_stroke:
            all_strokes.append(
                {
                    "points": self.active_stroke,
                    "color": self.current_color,
                    "width": 4,
                }
            )
        for stroke in all_strokes:
            pts = stroke["points"]
            if len(pts) < 2:
                continue
            painter.setPen(
                QPen(
                    stroke["color"],
                    stroke["width"],
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            t_pts = [self._target_point(p, scale, ox, oy) for p in pts]
            for i in range(len(t_pts) - 1):
                painter.drawLine(t_pts[i], t_pts[i + 1])

    def _draw_painted_edges(
        self, painter: QPainter, scale: float, ox: float, oy: float
    ) -> None:
        """塗られたグリッド辺（外側の発光表現＋本線の2パス）を描画する。"""
        line_w = self.line_width
        for key, color in self.edges.items():
            a = self._vertex_to_target(*key[0], scale, ox, oy)
            b = self._vertex_to_target(*key[1], scale, ox, oy)
            painter.setPen(
                QPen(
                    QColor(color.red(), color.green(), color.blue(), 70),
                    line_w + 8,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(a, b)
        for key, color in self.edges.items():
            a = self._vertex_to_target(*key[0], scale, ox, oy)
            b = self._vertex_to_target(*key[1], scale, ox, oy)
            painter.setPen(
                QPen(
                    color,
                    line_w,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(a, b)

    def _render_scene(
        self,
        painter: QPainter,
        scale: float,
        ox: float,
        oy: float,
        include_active_stroke: bool = True,
    ):
        """
        指定されたスケール・オフセットで盤面画像・セル塗り・仮想グリッド・
        自由手書き・塗り辺を共通描画する（一時UIオーバーレイは含まない）。
        """
        dest_rect = QRectF(
            ox, oy, self.pixmap.width() * scale, self.pixmap.height() * scale
        )
        painter.drawPixmap(dest_rect.toRect(), self.pixmap)
        self._draw_cell_fills(painter, scale, ox, oy)
        self._draw_virtual_grid(painter, scale, ox, oy)
        self._draw_freehand_strokes(painter, scale, ox, oy, include_active_stroke)
        self._draw_painted_edges(painter, scale, ox, oy)

    def render_export_image(self) -> QImage:
        """
        補正後画像の元解像度（周囲の黒帯なし）で、セル塗り・仮想グリッド・
        自由手書き・塗った線を合成した保存用 QImage を生成する。
        （ホバーハイライトや枠調整ハンドル等の一時UI表示は除外する）
        """
        if self.pixmap.isNull():
            return QImage()
        out_img = QImage(
            self.pixmap.width(),
            self.pixmap.height(),
            QImage.Format.Format_RGB32,
        )
        painter = QPainter(out_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._render_scene(
            painter, scale=1.0, ox=0.0, oy=0.0, include_active_stroke=False
        )
        painter.end()
        return out_img

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._draw_background_or_placeholder(
            painter,
            "補正後画像がありません。\nまずは四隅を指定して「補正してサンドボックスへ」を実行してください。",
        ):
            return

        scale, ox, oy = self.get_display_scale_and_offset()
        self._render_scene(painter, scale, ox, oy, include_active_stroke=True)

        # 枠調整モード: 外枠とハンドルを強調表示
        if self.draw_mode == DrawMode.GRID_ADJUST:
            tl = self.vertex_to_widget_coords(0, 0)
            br = self.vertex_to_widget_coords(self.cols, self.rows)
            painter.setPen(QPen(QColor(255, 170, 0), 3))
            painter.setBrush(QBrush(QColor(255, 170, 0, 30)))
            painter.drawRect(QRectF(tl, br))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(255, 170, 0)))
            for pt in self.grid_handle_positions().values():
                painter.drawRect(QRectF(pt.x() - 6, pt.y() - 6, 12, 12))

        # ホバー中の辺のハイライト
        if self.draw_mode in (DrawMode.SNAP_LINE, DrawMode.ERASER) and self.hovered_edge:
            a = self.vertex_to_widget_coords(*self.hovered_edge[0])
            b = self.vertex_to_widget_coords(*self.hovered_edge[1])
            painter.setPen(
                QPen(
                    QColor(255, 255, 255, 120),
                    self.line_width,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(a, b)

    # --------------------------------------------------------------------------
    # マウスイベント処理
    # --------------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self.pixmap.isNull():
            return

        pos = event.position()

        if self.draw_mode == DrawMode.GRID_ADJUST:
            if event.button() == Qt.MouseButton.LeftButton:
                flags = self.find_adjust_target(pos)
                if flags is not None:
                    self.begin_undo_transaction()
                    self.adjust_flags = flags
                    self.adjust_start_pos = pos
                    self.adjust_start_rect = list(self.grid_rect)
            return

        if self.draw_mode == DrawMode.SNAP_LINE:
            edge = self.find_nearest_edge(pos)
            if edge is None:
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.begin_undo_transaction()
                self.apply_edge(edge, "erase")
            elif event.button() == Qt.MouseButton.LeftButton:
                self.begin_undo_transaction()
                self.edge_drag_mode = self.toggle_edge(edge)
                self.last_drag_pos = pos

                if self.edge_drag_mode == "paint":
                    self.op_edges = [edge]
                    self.op_vertex_edges = {edge[0]: [edge], edge[1]: [edge]}
                    self.drag_tip = None
                    self.drag_base = None
                else:
                    self.op_edges = []
                    self.op_vertex_edges = {}
                    self.drag_tip = None
                    self.drag_base = None

        elif self.draw_mode == DrawMode.CELL_FILL:
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self.begin_undo_transaction()
                cell = self.find_cell_at_pos(pos)
                if cell is not None:
                    prev_color = self.cell_fills.get(cell)
                    if (
                        event.button() == Qt.MouseButton.RightButton
                        or prev_color == self.current_color
                    ):
                        self.cell_fills.pop(cell, None)
                    else:
                        self.cell_fills[cell] = QColor(self.current_color)
                    self.update()

        elif self.draw_mode == DrawMode.FREEHAND:
            if event.button() == Qt.MouseButton.LeftButton:
                self.begin_undo_transaction()
                self.active_stroke = [self.widget_to_image_coords(pos)]
                self.update()

        elif self.draw_mode == DrawMode.ERASER:
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self.begin_undo_transaction()
                self.erase_at_pos(pos)

    def _grid_neighbors(self, vertex: tuple[int, int]) -> list[tuple[int, int]]:
        """指定した格子頂点の上下左右の有効な隣接頂点リストを返す。"""
        tc, tr = vertex
        neighbors: list[tuple[int, int]] = []
        if tc + 1 <= self.cols:
            neighbors.append((tc + 1, tr))
        if tc - 1 >= 0:
            neighbors.append((tc - 1, tr))
        if tr + 1 <= self.rows:
            neighbors.append((tc, tr + 1))
        if tr - 1 >= 0:
            neighbors.append((tc, tr - 1))
        return neighbors

    def _try_rewind_snap_edge(self, sub_pos: QPointF, threshold: float) -> bool:
        """線塗りドラッグ中に1つ前の頂点 (drag_base) へ戻った場合、直前の辺を元の状態へ巻き戻す。"""
        if len(self.op_edges) <= 1 or self.drag_base is None:
            return False
        base_pt = self.vertex_to_widget_coords(*self.drag_base)
        if math.hypot(sub_pos.x() - base_pt.x(), sub_pos.y() - base_pt.y()) > threshold:
            return False

        last_edge = self.op_edges.pop()
        orig_color = (
            self._tx_start_state["edges"].get(last_edge)
            if self._tx_start_state is not None
            else None
        )
        if orig_color is None:
            self.edges.pop(last_edge, None)
        else:
            self.edges[last_edge] = QColor(orig_color)
        self.update()

        for v in last_edge:
            if v in self.op_vertex_edges and last_edge in self.op_vertex_edges[v]:
                self.op_vertex_edges[v].remove(last_edge)

        self.drag_tip = self.drag_base
        if len(self.op_edges) > 1:
            prev_e = self.op_edges[-1]
            self.drag_base = prev_e[0] if prev_e[1] == self.drag_tip else prev_e[1]
        else:
            self.drag_base = None
            self.drag_tip = None
        return True

    def _find_best_extension_neighbor(
        self,
        tip: tuple[int, int],
        exclude_vertex: tuple[int, int] | None,
        sub_pos: QPointF,
        threshold: float,
    ) -> tuple[float, tuple[int, int] | None]:
        """先端頂点 tip からマウス位置 sub_pos に向かって伸ばせる最短距離の隣接頂点を探索する。"""
        p_tip = self.vertex_to_widget_coords(*tip)
        dist_to_tip = math.hypot(sub_pos.x() - p_tip.x(), sub_pos.y() - p_tip.y())
        best_next: tuple[int, int] | None = None
        best_dist = float("inf")

        for nxt in self._grid_neighbors(tip):
            if nxt == exclude_vertex:
                continue
            p_nxt = self.vertex_to_widget_coords(*nxt)
            d = point_to_segment_distance(
                sub_pos.x(), sub_pos.y(), p_tip.x(), p_tip.y(), p_nxt.x(), p_nxt.y()
            )
            dist_to_nxt = math.hypot(sub_pos.x() - p_nxt.x(), sub_pos.y() - p_nxt.y())
            if d <= threshold and d < best_dist and dist_to_nxt < dist_to_tip + 20:
                best_dist = d
                best_next = nxt

        return best_dist, best_next

    def _append_snap_extension(
        self, base_vertex: tuple[int, int], next_vertex: tuple[int, int]
    ) -> None:
        """新しい辺を塗って現在のドラッグ操作の経路 (op_edges) と先端追跡状態を更新する。"""
        new_edge = self.make_edge_key(base_vertex, next_vertex)
        if new_edge not in self.op_edges:
            self.apply_edge(new_edge, "paint")
            self.op_edges.append(new_edge)
            self.op_vertex_edges.setdefault(base_vertex, []).append(new_edge)
            self.op_vertex_edges.setdefault(next_vertex, []).append(new_edge)
            self.drag_base = base_vertex
            self.drag_tip = next_vertex

    def _step_snap_paint(self, sub_pos: QPointF, threshold: float) -> None:
        """線塗りドラッグの補間1ステップ分の巻き戻しまたは延長処理を実行する。"""
        if self._try_rewind_snap_edge(sub_pos, threshold):
            return

        if len(self.op_edges) == 1:
            init_e = self.op_edges[0]
            best_match: tuple[tuple[int, int], tuple[int, int]] | None = None
            best_dist = float("inf")
            for cand_tip, cand_base in ((init_e[0], init_e[1]), (init_e[1], init_e[0])):
                d, nxt = self._find_best_extension_neighbor(
                    cand_tip, cand_base, sub_pos, threshold
                )
                if nxt is not None and d < best_dist:
                    best_dist = d
                    best_match = (cand_tip, nxt)
            if best_match is not None:
                self._append_snap_extension(*best_match)

        elif len(self.op_edges) > 1 and self.drag_tip is not None:
            if len(self.op_vertex_edges.get(self.drag_tip, [])) < 2:
                _, best_next = self._find_best_extension_neighbor(
                    self.drag_tip, self.drag_base, sub_pos, threshold
                )
                if best_next is not None:
                    self._append_snap_extension(self.drag_tip, best_next)

    def _handle_move_grid_adjust(
        self, pos: QPointF, buttons: Qt.MouseButton
    ) -> None:
        """枠調整モードのマウス移動（ドラッグ変形またはカーソル形状更新）を処理する。"""
        if self.adjust_flags is not None and (buttons & Qt.MouseButton.LeftButton):
            self.apply_grid_adjust(pos)
        else:
            self.setCursor(self.adjust_cursor_for(self.find_adjust_target(pos)))

    def _update_hovered_edge(self, pos: QPointF) -> None:
        """カーソル近傍のホバー辺を更新して再描画を要求する。"""
        edge = self.find_nearest_edge(pos)
        if edge != self.hovered_edge:
            self.hovered_edge = edge
            self.update()

    def _handle_move_snap_line(
        self, pos: QPointF, buttons: Qt.MouseButton
    ) -> None:
        """線塗りモードのマウスドラッグ（連続塗り・巻き戻し・連続消去）を処理する。"""
        if not (self.edge_drag_mode and (buttons & Qt.MouseButton.LeftButton)):
            return

        start = self.last_drag_pos if self.last_drag_pos is not None else pos
        dist = math.hypot(pos.x() - start.x(), pos.y() - start.y())
        steps = max(1, int(dist // 8))
        threshold = self.edge_hit_threshold()

        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            sub_pos = QPointF(
                start.x() + (pos.x() - start.x()) * t,
                start.y() + (pos.y() - start.y()) * t,
            )
            if self.edge_drag_mode == "paint":
                self._step_snap_paint(sub_pos, threshold)
            elif self.edge_drag_mode == "erase":
                edge = self.find_nearest_edge(sub_pos)
                if edge is not None:
                    self.apply_edge(edge, "erase")

        self.last_drag_pos = pos

    def _handle_move_freehand(self, pos: QPointF) -> None:
        """自由手書きモードのドラッグ点追加を処理する。"""
        if self.active_stroke:
            self.active_stroke.append(self.widget_to_image_coords(pos))
            self.update()

    def _handle_move_cell_fill(
        self, pos: QPointF, buttons: Qt.MouseButton
    ) -> None:
        """マス目塗りモードのドラッグ連続塗りを処理する。"""
        if buttons & Qt.MouseButton.LeftButton:
            if self._tx_start_state is None:
                self.begin_undo_transaction()
            cell = self.find_cell_at_pos(pos)
            if cell is not None and self.cell_fills.get(cell) != self.current_color:
                self.cell_fills[cell] = QColor(self.current_color)
                self.update()

    def _handle_move_eraser(
        self, pos: QPointF, buttons: Qt.MouseButton
    ) -> None:
        """消しゴムモードのドラッグ連続消去を処理する。"""
        if buttons & Qt.MouseButton.LeftButton:
            if self._tx_start_state is None:
                self.begin_undo_transaction()
            self.erase_at_pos(pos)

    def mouseMoveEvent(self, event):
        pos = event.position()
        buttons = event.buttons()

        if self.draw_mode == DrawMode.GRID_ADJUST:
            self._handle_move_grid_adjust(pos, buttons)
            return

        if self.draw_mode in (DrawMode.SNAP_LINE, DrawMode.ERASER):
            self._update_hovered_edge(pos)

        if self.draw_mode == DrawMode.SNAP_LINE:
            self._handle_move_snap_line(pos, buttons)
        elif self.draw_mode == DrawMode.FREEHAND:
            self._handle_move_freehand(pos)
        elif self.draw_mode == DrawMode.CELL_FILL:
            self._handle_move_cell_fill(pos, buttons)
        elif self.draw_mode == DrawMode.ERASER:
            self._handle_move_eraser(pos, buttons)

    def mouseReleaseEvent(self, event):
        if self.draw_mode == DrawMode.FREEHAND:
            if self.active_stroke:
                if len(self.active_stroke) >= 2:
                    self.freehand_strokes.append(
                        {
                            "points": [QPointF(p) for p in self.active_stroke],
                            "color": QColor(self.current_color),
                            "width": 4,
                        }
                    )
                self.active_stroke.clear()
                self.update()

        self.edge_drag_mode = None
        self.last_drag_pos = None
        self.drag_tip = None
        self.drag_base = None
        self.op_edges.clear()
        self.op_vertex_edges.clear()
        self.adjust_flags = None
        self.adjust_start_pos = None
        self.adjust_start_rect = None
        self.commit_undo_transaction()
