"""
The Witness サンドボックス支援ツール (Witness Sandbox Tool)

【機能】
1. 画面キャプチャ (F4 / ボタン) / 画像読み込み / サンプル画像生成
2. 斜めパズル盤面の四隅指定 & OpenCVによる透視投影変換 (パース補正)
3. 仮想グリッド重畳 (行・列・マージン調整)
4. グリッド吸着 (スナップ) 一筆書き描画機能 (The Witness風の巻き戻し対応)
5. 半透明マス目塗りつぶし機能
6. フリーハンド描画 / 消しゴム / ペン色変更 / Undo / Clear
7. 最前面表示 (TopMost) トグル
"""

import sys
import time
import math
import numpy as np
import cv2
from PIL import Image, ImageGrab

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QPolygonF, QAction, QKeySequence, QCursor, QShortcut
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QSlider, QColorDialog, QFileDialog,
    QStackedWidget, QToolBar, QButtonGroup, QRadioButton, QCheckBox,
    QMessageBox, QGroupBox, QSplitter, QStatusBar, QFrame, QComboBox
)

import ctypes
from ctypes import wintypes

# Windows API 定数
MOD_NONE = 0x0000
VK_F4 = 0x73
WM_HOTKEY = 0x0312
HOTKEY_ID_F4 = 1001


# ==============================================================================
# 幾何・画像処理ユーティリティ
# ==============================================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4つの頂点座標を [左上, 右上, 右下, 左下] の順に幾何学的にソートする。
    pts: shape (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")

    # x + y の合計値: 左上(最小), 右下(最大)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # x - y の差分: 右上(最大), 左下(最小)
    diff = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]

    return rect


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """点 (px, py) と線分 (x1, y1)-(x2, y2) の最短ユークリッド距離を計算する。"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def detect_grid_dimensions(image_bgr: np.ndarray, pts: np.ndarray | None = None) -> tuple[int, int, float]:
    """
    画像内のパズルグリッド線（列数 cols、行数 rows、マージン割合 margin）を画像解析で自動検知する。
    pts が指定された場合は、まず正方形で仮の透視投影変換を行ってからグリッドを解析する。
    戻り値: (cols, rows, margin_percent)
    """
    if pts is not None and len(pts) == 4:
        rect = order_points(pts)
        sample_size = 600
        dst = np.array([
            [0, 0],
            [sample_size - 1, 0],
            [sample_size - 1, sample_size - 1],
            [0, sample_size - 1]
        ], dtype="float32")
        m = cv2.getPerspectiveTransform(rect, dst)
        analysis_img = cv2.warpPerspective(image_bgr, m, (sample_size, sample_size), flags=cv2.INTER_LINEAR)
    else:
        analysis_img = image_bgr

    if analysis_img is None or analysis_img.size == 0:
        return 4, 4, 0.05

    # グレースケール化とSobelフィルタによる水平・垂直エッジ抽出
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

    h, w = gray.shape
    # 外枠以外のノイズを避けるため、中央部でプロファイルを計算
    h_proj = np.mean(sobel_y[:, int(w * 0.15):int(w * 0.85)], axis=1)
    v_proj = np.mean(sobel_x[int(h * 0.15):int(h * 0.85), :], axis=0)

    def evaluate_projection(proj, length) -> tuple[int, float]:
        best_n = 4
        best_m = 0.05
        best_score = -1.0

        # 分割数 1〜10、マージン 0%〜25% を探索
        for n in range(1, 11):
            for m_pct in range(0, 26):
                m_ratio = m_pct / 100.0
                grid_len = length * (1.0 - 2.0 * m_ratio)
                cell_len = grid_len / n

                vals = []
                for k in range(n + 1):
                    pos = int(round(length * m_ratio + k * cell_len))
                    val = np.max(proj[max(0, pos - 3):min(length, pos + 4)])
                    vals.append(val)

                vals = np.array(vals)
                min_v = np.min(vals)
                max_v = np.max(vals)

                # 全てのグリッド線が一定の強度で存在しているかを判定
                if min_v > max_v * 0.35:
                    score = np.sum(vals)
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_m = m_ratio

        return best_n, best_m

    detected_cols, m_cols = evaluate_projection(v_proj, w)
    detected_rows, m_rows = evaluate_projection(h_proj, h)
    avg_margin = round((m_cols + m_rows) / 2.0, 2)

    return detected_cols, detected_rows, avg_margin


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """
    ウィンドウの画面上の矩形 (left, top, right, bottom) を返す。
    影を含まない DWM の拡張フレーム境界を優先し、失敗時は GetWindowRect を使う。
    最小化・サイズ不正・取得失敗時は None。
    """
    try:
        user32 = ctypes.windll.user32
        if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
            return None
        rect = wintypes.RECT()
        # DWMWA_EXTENDED_FRAME_BOUNDS = 9 (ウィンドウ影を除いた見た目通りの枠)
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if res != 0 and not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        l, t, r, b = rect.left, rect.top, rect.right, rect.bottom
        # 仮想デスクトップ範囲 (全モニタ) にクリップ
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
        vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        l, t = max(l, vx), max(t, vy)
        r, b = min(r, vx + vw), min(b, vy + vh)
        if r - l < 50 or b - t < 50:
            return None
        return l, t, r, b
    except Exception:
        return None


def capture_foreground_window_bgr(target_hwnd: int | None = None, exclude_hwnd: int | None = None) -> np.ndarray:
    """
    指定ウィンドウ (省略時は現在の前面ウィンドウ) のみを撮影して BGR 画像で返す。
    exclude_hwnd (自アプリ) が対象になった場合や矩形が取得できない場合は全画面にフォールバックする。
    """
    hwnd = target_hwnd
    try:
        if hwnd is None:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        hwnd = None
    if hwnd and exclude_hwnd and int(hwnd) == int(exclude_hwnd):
        hwnd = None

    rect = get_window_rect(int(hwnd)) if hwnd else None
    if rect is not None:
        shot = ImageGrab.grab(bbox=rect, all_screens=True)
    else:
        shot = ImageGrab.grab(all_screens=True)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def warp_perspective(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    aspect_mode: str = "square",
    cols: int = 4,
    rows: int = 4
) -> np.ndarray:
    """
    指定された4隅をもとに透視投影変換を行い、真正面から見た画像に補正する。
    aspect_mode:
        - "square": 正方形 (1:1) の盤面として補正
        - "grid": マス目が正方形 (1:1) になるよう、列数:行数の比率で補正
        - "original": 元の斜め画面上の推定ユークリッド距離比で補正
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 基準となる幅と高さを計算
    width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
    width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    base_w = max(int(width_a), int(width_b))

    height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
    height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    base_h = max(int(height_a), int(height_b))

    # 基本サイズ
    base_size = max(base_w, base_h)
    base_size = max(400, min(base_size, 1600))

    if aspect_mode == "square":
        # 1:1 の正方形盤面
        dst_w = base_size
        dst_h = base_size
    elif aspect_mode == "grid":
        # マス目が 1:1 正方形になるように cols : rows の比率で生成
        c = max(1, cols)
        r = max(1, rows)
        if c >= r:
            dst_w = base_size
            dst_h = int(round(base_size * (r / c)))
        else:
            dst_h = base_size
            dst_w = int(round(base_size * (c / r)))
        dst_w = max(300, min(dst_w, 1800))
        dst_h = max(300, min(dst_h, 1800))
    else:
        # "original": 従来のピクセル距離比
        dst_w = max(400, min(base_w, 1600))
        dst_h = max(400, min(base_h, 1600))

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype="float32")

    # 変換行列の計算と実行
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, m, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return warped


def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """OpenCVのBGR画像をPyQtのQPixmapに変換する。"""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def create_sample_witness_image() -> np.ndarray:
    """
    The Witness風の斜めパズル画像を自動生成する (動作テスト・デモ用)。
    """
    # 真正面のベース画像 (600x600)
    base_size = 600
    panel = np.full((base_size, base_size, 3), (35, 30, 25), dtype=np.uint8)  # パネル下地 (暗い青/濃紺)

    # 枠線と外枠
    cv2.rectangle(panel, (40, 40), (base_size - 40, base_size - 40), (200, 180, 150), 8)

    # 4x4 グリッド線
    rows, cols = 4, 4
    gx0, gy0 = 60, 60
    gw = (base_size - 120) / cols
    gh = (base_size - 120) / rows

    for r in range(rows + 1):
        y = int(gy0 + r * gh)
        cv2.line(panel, (int(gx0), y), (int(gx0 + cols * gw), y), (140, 130, 110), 4)
    for c in range(cols + 1):
        x = int(gx0 + c * gw)
        cv2.line(panel, (x, int(gy0)), (x, int(gy0 + rows * gh)), (140, 130, 110), 4)

    # スタート地点 (左下の大きな丸)
    start_x = int(gx0)
    start_y = int(gy0 + rows * gh)
    cv2.circle(panel, (start_x, start_y), 18, (230, 220, 200), -1)

    # ゴール地点 (右上の飛び出し線)
    end_x = int(gx0 + cols * gw)
    end_y = int(gy0)
    cv2.line(panel, (end_x, end_y), (end_x + 25, end_y - 25), (200, 180, 150), 6)

    # マス目の記号 (The Witnessでおなじみの白四角・黒四角)
    def draw_symbol(cell_r, cell_c, color_bgr):
        cx = int(gx0 + cell_c * gw + gw / 2)
        cy = int(gy0 + cell_r * gh + gh / 2)
        s = 14
        cv2.rectangle(panel, (cx - s, cy - s), (cx + s, cy + s), color_bgr, -1)
        cv2.rectangle(panel, (cx - s, cy - s), (cx + s, cy + s), (30, 30, 30), 2)

    # 白と黒の記号をいくつか配置
    draw_symbol(0, 0, (240, 240, 240))  # 白
    draw_symbol(1, 1, (240, 240, 240))  # 白
    draw_symbol(2, 2, (30, 30, 30))     # 黒
    draw_symbol(3, 3, (30, 30, 30))     # 黒
    draw_symbol(1, 3, (0, 165, 255))    # オレンジ色の星

    # 背景のある大きなキャンバスに「斜めのパース」をかけて貼り付ける
    bg_w, bg_h = 1000, 700
    scene = np.full((bg_h, bg_w, 3), (60, 50, 45), dtype=np.uint8)  # 周囲の遺跡/岩壁風背景

    # 歪みの変換行列 (ゲーム画面のように斜め右奥に傾いた状態を作る)
    src_pts = np.array([
        [0, 0],
        [base_size, 0],
        [base_size, base_size],
        [0, base_size]
    ], dtype="float32")

    # 斜めの四隅 (歪んだパネル)
    skew_pts = np.array([
        [220, 110],  # 左上
        [780, 70],   # 右上 (奥に向かってすぼまる)
        [890, 580],  # 右下
        [150, 610]   # 左下
    ], dtype="float32")

    m_skew = cv2.getPerspectiveTransform(src_pts, skew_pts)
    warped_panel = cv2.warpPerspective(panel, m_skew, (bg_w, bg_h))

    # マスクを作成して合成
    gray = cv2.cvtColor(warped_panel, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    scene[mask > 0] = warped_panel[mask > 0]

    # 説明テキストを描画
    cv2.putText(
        scene, "Sample Witness Puzzle Screen (Skewed View)",
        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA
    )
    cv2.putText(
        scene, "Click 4 corners of the puzzle panel to correct perspective!",
        (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 200, 220), 2, cv2.LINE_AA
    )

    return scene


# ==============================================================================
# グローバルホットキー通信用ヘルパー
# ==============================================================================

class HotkeySignaler(QObject):
    capture_requested = pyqtSignal()


# ==============================================================================
# モード1: 四隅指定キャンバス (PerspectiveCanvas)
# ==============================================================================

class PerspectiveCanvas(QWidget):
    """
    キャプチャ画像上でパズル盤面の四隅をマウスクリックで指定するキャンバス。
    """
    points_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_bgr: np.ndarray | None = None
        self.pixmap: QPixmap = QPixmap()
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

    def get_display_scale_and_offset(self):
        """ウィジェットサイズに応じた画像のスケーリング倍率と描画オフセットを計算する。"""
        if self.pixmap.isNull():
            return 1.0, 0, 0
        w_ratio = self.width() / self.pixmap.width()
        h_ratio = self.height() / self.pixmap.height()
        scale = min(w_ratio, h_ratio)
        offset_x = (self.width() - self.pixmap.width() * scale) / 2
        offset_y = (self.height() - self.pixmap.height() * scale) / 2
        return scale, offset_x, offset_y

    def widget_to_image_coords(self, pt: QPointF) -> QPointF:
        """ウィジェット上のマウス座標を画像内の元座標に変換する。"""
        scale, ox, oy = self.get_display_scale_and_offset()
        img_x = (pt.x() - ox) / scale
        img_y = (pt.y() - oy) / scale
        return QPointF(img_x, img_y)

    def image_to_widget_coords(self, pt: QPointF) -> QPointF:
        """画像内の元座標をウィジェット上の表示座標に変換する。"""
        scale, ox, oy = self.get_display_scale_and_offset()
        return QPointF(pt.x() * scale + ox, pt.y() * scale + oy)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景の黒塗り
        painter.fillRect(self.rect(), QColor(25, 25, 28))

        if self.pixmap.isNull():
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "画像がありません。\n[画面キャプチャ(F4)] または [サンプル画像読み込み] をクリックしてください。"
            )
            return

        scale, ox, oy = self.get_display_scale_and_offset()
        dest_rect = QRectF(ox, oy, self.pixmap.width() * scale, self.pixmap.height() * scale)
        painter.drawPixmap(dest_rect.toRect(), self.pixmap)

        # 選択済みの点を接続する多角形の描画
        if len(self.points) > 1:
            poly = QPolygonF([self.image_to_widget_coords(p) for p in self.points])
            # 4点揃っている場合は半透明塗りつぶし
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

            # 外枠円
            radius = 12 if is_hover else 9
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(255, 80, 80) if is_hover else QColor(0, 200, 255)))
            painter.drawEllipse(w_pt, radius, radius)

            # 番号ラベル
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(
                QRectF(w_pt.x() - 15, w_pt.y() - 30, 30, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"P{idx + 1}"
            )

    def mousePressEvent(self, event):
        if self.pixmap.isNull():
            return

        pos = event.position()
        # 既存の点をつかんでドラッグ移動できるかチェック
        hit_radius = 15
        for idx, pt in enumerate(self.points):
            w_pt = self.image_to_widget_coords(pt)
            if math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y()) <= hit_radius:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragged_point_idx = idx
                    self.update()
                    return
                elif event.button() == Qt.MouseButton.RightButton:
                    # 右クリックでその点を削除
                    self.points.pop(idx)
                    self.points_updated.emit(self.points)
                    self.update()
                    return

        # 左クリックで新しい点を追加 (最大4点)
        if event.button() == Qt.MouseButton.LeftButton and len(self.points) < 4:
            img_pt = self.widget_to_image_coords(pos)
            # 画像の境界内に収める
            img_pt.setX(max(0, min(img_pt.x(), self.pixmap.width() - 1)))
            img_pt.setY(max(0, min(img_pt.y(), self.pixmap.height() - 1)))
            self.points.append(img_pt)
            self.points_updated.emit(self.points)
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.dragged_point_idx >= 0:
            # ドラッグ移動
            img_pt = self.widget_to_image_coords(pos)
            img_pt.setX(max(0, min(img_pt.x(), self.pixmap.width() - 1)))
            img_pt.setY(max(0, min(img_pt.y(), self.pixmap.height() - 1)))
            self.points[self.dragged_point_idx] = img_pt
            self.points_updated.emit(self.points)
            self.update()
            return

        # ホバー判定
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
    SNAP_LINE = 0   # グリッドスナップ一筆書き
    CELL_FILL = 1   # マス目塗りつぶし
    FREEHAND = 2    # 自由手書き
    ERASER = 3      # 消しゴム
    GRID_ADJUST = 4  # グリッド外枠の位置・サイズ調整


class SandboxCanvas(QWidget):
    """
    パース補正後の画像上に仮想グリッドを重ね、スナップ描画や塗りつぶしを行うキャンバス。
    """
    # グリッド外枠が (ドラッグ等で) 変更されたとき
    grid_rect_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_bgr: np.ndarray | None = None
        self.pixmap: QPixmap = QPixmap()

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
        # 辺ドラッグ塗り/消しのモード: None / "paint" / "erase"
        self.edge_drag_mode: str | None = None
        self.last_drag_pos: QPointF | None = None
        # 1操作での連続線伸ばし・巻き戻し追跡
        self.drag_tip: tuple[int, int] | None = None
        self.drag_base: tuple[int, int] | None = None
        self.op_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.op_vertex_edges: dict[tuple[int, int], list] = {}
        # ホバー中の辺 (ハイライト用)
        self.hovered_edge: tuple[tuple[int, int], tuple[int, int]] | None = None

        # 2. セル塗りつぶしデータ: {(col, row): QColor}
        self.cell_fills: dict[tuple[int, int], QColor] = {}

        # 3. フリーハンドストローク: list[dict] ({"points": [QPointF], "color": QColor, "width": int})
        self.freehand_strokes: list[dict] = []
        self.active_stroke: list[QPointF] = []

        # 4. 操作履歴 (Undo用)
        # 各要素: {"type": "path"|"fill"|"stroke", "data": ...}
        self.undo_stack: list[dict] = []

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

    def set_margin_percent(self, percent: float):
        """マージン割合を更新する (0.0 ~ 0.3)。"""
        self.margin_percent = max(0.0, min(0.35, percent))
        m = self.margin_percent
        self.grid_rect = [m, m, 1.0 - m, 1.0 - m]
        self.update()

    def set_draw_mode(self, mode: int):
        """描画モードを切り替える。"""
        self.draw_mode = mode
        self.edge_drag_mode = None
        self.adjust_flags = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.active_stroke.clear()
        self.update()

    def set_color(self, color: QColor):
        """描画色を設定する。"""
        self.current_color = color

    def clear_all(self, clear_history=False):
        """描画内容を全消去する。"""
        self.edges.clear()
        self.cell_fills.clear()
        self.freehand_strokes.clear()
        self.active_stroke.clear()
        if clear_history:
            self.undo_stack.clear()
        else:
            self.undo_stack.append({"type": "clear_snapshot"})
        self.update()

    def undo(self):
        """直前の操作を1つ取り消す。"""
        if not self.undo_stack:
            return

        action = self.undo_stack.pop()
        action_type = action["type"]

        if action_type == "edge":
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
        elif action_type in ("add_stroke", "stroke"):
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
        """辺に塗り/消去を適用し、変化があれば Undo に記録する。"""
        prev = self.edges.get(key)
        if mode == "paint":
            if prev is not None and prev == self.current_color:
                return False
            self.edges[key] = self.current_color
        else:
            if prev is None:
                return False
            self.edges.pop(key)
        self.undo_stack.append({"type": "edge", "edge": key, "prev_color": prev})
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
                p1 = pts[i]
                p2 = pts[i + 1]
                dist = point_to_segment_distance(px, py, p1.x(), p1.y(), p2.x(), p2.y())
                if dist <= threshold_px:
                    return idx
        return None

    def delete_stroke_at_index(self, idx: int) -> bool:
        """指定インデックスの手書きストロークを削除し、Undoスタックに記録する。"""
        if 0 <= idx < len(self.freehand_strokes):
            data = self.freehand_strokes.pop(idx)
            self.undo_stack.append({
                "type": "remove_stroke",
                "index": idx,
                "data": data
            })
            self.update()
            return True
        return False

    def erase_at_pos(self, pos: QPointF) -> bool:
        """
        消しゴム機能: 指定座標にある塗り辺、セル塗り、または手書きストロークを消去する。
        """
        erased = False

        # 1. 塗られた辺の消去
        edge = self.find_nearest_edge(pos)
        if edge is not None and self.apply_edge(edge, "erase"):
            erased = True

        # 2. セル塗りの消去
        cell = self.find_cell_at_pos(pos)
        if cell in self.cell_fills:
            prev = self.cell_fills.pop(cell)
            self.undo_stack.append({
                "type": "fill",
                "cell": cell,
                "prev_color": prev,
                "new_color": None
            })
            erased = True

        # 3. 自由手書きストロークの消去
        hit_stroke_idx = self.find_stroke_at_pos(pos)
        if hit_stroke_idx is not None:
            self.delete_stroke_at_index(hit_stroke_idx)
            erased = True

        if erased:
            self.update()
        return erased

    # --------------------------------------------------------------------------
    # 座標変換・グリッド計算
    # --------------------------------------------------------------------------

    def get_display_scale_and_offset(self):
        if self.pixmap.isNull():
            return 1.0, 0, 0
        w_ratio = self.width() / self.pixmap.width()
        h_ratio = self.height() / self.pixmap.height()
        scale = min(w_ratio, h_ratio)
        offset_x = (self.width() - self.pixmap.width() * scale) / 2
        offset_y = (self.height() - self.pixmap.height() * scale) / 2
        return scale, offset_x, offset_y

    def get_grid_bounds(self) -> tuple[float, float, float, float]:
        """
        画像座標系におけるグリッドの (min_x, min_y, max_x, max_y) を計算する。
        """
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
        img_pt = self.vertex_to_image_coords(col, row)
        scale, ox, oy = self.get_display_scale_and_offset()
        return QPointF(img_pt.x() * scale + ox, img_pt.y() * scale + oy)

    def find_nearest_vertex(self, widget_pos: QPointF, threshold_px: float = 28.0) -> tuple[int, int] | None:
        """
        指定されたウィジェット座標に最も近いグリッド頂点 (col, row) を検出する。
        """
        best_v = None
        best_dist = float("inf")

        for c in range(self.cols + 1):
            for r in range(self.rows + 1):
                wpt = self.vertex_to_widget_coords(c, r)
                dist = math.hypot(widget_pos.x() - wpt.x(), widget_pos.y() - wpt.y())
                if dist < best_dist and dist <= threshold_px:
                    best_dist = dist
                    best_v = (c, r)
        return best_v

    # --------------------------------------------------------------------------
    # グリッド外枠の調整 (枠調整ツール)
    # --------------------------------------------------------------------------

    # ハンドル名 -> 動かす辺 (left, top, right, bottom) のフラグ
    ADJUST_FLAGS = {
        "tl": (1, 1, 0, 0), "tr": (0, 1, 1, 0), "br": (0, 0, 1, 1), "bl": (1, 0, 0, 1),
        "t": (0, 1, 0, 0), "b": (0, 0, 0, 1), "l": (1, 0, 0, 0), "r": (0, 0, 1, 0),
    }

    def grid_handle_positions(self) -> dict[str, QPointF]:
        """枠調整ハンドル (四隅 + 各辺中点) のウィジェット座標を返す。"""
        tl = self.vertex_to_widget_coords(0, 0)
        br = self.vertex_to_widget_coords(self.cols, self.rows)
        cx = (tl.x() + br.x()) / 2.0
        cy = (tl.y() + br.y()) / 2.0
        return {
            "tl": QPointF(tl.x(), tl.y()), "tr": QPointF(br.x(), tl.y()),
            "br": QPointF(br.x(), br.y()), "bl": QPointF(tl.x(), br.y()),
            "t": QPointF(cx, tl.y()), "b": QPointF(cx, br.y()),
            "l": QPointF(tl.x(), cy), "r": QPointF(br.x(), cy),
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
        if self.adjust_flags is None or self.adjust_start_pos is None or self.adjust_start_rect is None:
            return
        scale, _, _ = self.get_display_scale_and_offset()
        iw, ih = self.pixmap.width(), self.pixmap.height()
        dx = (pos.x() - self.adjust_start_pos.x()) / scale / iw
        dy = (pos.y() - self.adjust_start_pos.y()) / scale / ih
        l0, t0, r0, b0 = self.adjust_start_rect
        fl, ft, fr, fb = self.adjust_flags
        min_size = 0.05

        if (fl, ft, fr, fb) == (1, 1, 1, 1):
            # 全体移動: サイズを保ったまま画像内に収める
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
        """
        指定されたウィジェット座標が位置するマス目 (col, row) を検出する。
        """
        scale, ox, oy = self.get_display_scale_and_offset()
        img_x = (widget_pos.x() - ox) / scale
        img_y = (widget_pos.y() - oy) / scale

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 28))

        if self.pixmap.isNull():
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "補正後画像がありません。\nまずは四隅を指定して「補正してサンドボックスへ」を実行してください。"
            )
            return

        scale, ox, oy = self.get_display_scale_and_offset()
        dest_rect = QRectF(ox, oy, self.pixmap.width() * scale, self.pixmap.height() * scale)
        painter.drawPixmap(dest_rect.toRect(), self.pixmap)

        # 1. マス目の塗りつぶし描画 (半透明)
        min_x, min_y, max_x, max_y = self.get_grid_bounds()
        cell_w = (max_x - min_x) / self.cols
        cell_h = (max_y - min_y) / self.rows

        for (c, r), col in self.cell_fills.items():
            # ウィジェット座標でのセル矩形
            cx = (min_x + c * cell_w) * scale + ox
            cy = (min_y + r * cell_h) * scale + oy
            cw = cell_w * scale
            ch = cell_h * scale
            fill_rect = QRectF(cx, cy, cw, ch)

            # 半透明カラーで塗る (アルファ値110で背景記号が透けて見えるように)
            alpha_color = QColor(col.red(), col.green(), col.blue(), 110)
            painter.fillRect(fill_rect, alpha_color)

            # 境界のハイライト線
            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 180), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(fill_rect)

        # 2. 仮想グリッド線の描画 (半透明の白・グレー)
        grid_pen = QPen(QColor(255, 255, 255, 80), 1.5, Qt.PenStyle.DashLine)
        outer_pen = QPen(QColor(255, 255, 255, 140), 2.5)

        # 内枠線
        painter.setPen(grid_pen)
        for r in range(self.rows + 1):
            p1 = self.vertex_to_widget_coords(0, r)
            p2 = self.vertex_to_widget_coords(self.cols, r)
            painter.drawLine(p1, p2)
        for c in range(self.cols + 1):
            p1 = self.vertex_to_widget_coords(c, 0)
            p2 = self.vertex_to_widget_coords(c, self.rows)
            painter.drawLine(p1, p2)

        # 外枠
        painter.setPen(outer_pen)
        tl = self.vertex_to_widget_coords(0, 0)
        br = self.vertex_to_widget_coords(self.cols, self.rows)
        painter.drawRect(QRectF(tl, br))

        # 3. 格子頂点の小さなドット描画
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
        dot_r = 3.5
        for c in range(self.cols + 1):
            for r in range(self.rows + 1):
                wpt = self.vertex_to_widget_coords(c, r)
                painter.drawEllipse(wpt, dot_r, dot_r)

        # 4. フリーハンドストロークの描画
        all_strokes = list(self.freehand_strokes)
        if self.active_stroke:
            all_strokes.append({
                "points": self.active_stroke,
                "color": self.current_color,
                "width": 4
            })
        for stroke in all_strokes:
            pts = stroke["points"]
            if len(pts) < 2:
                continue
            painter.setPen(QPen(
                stroke["color"],
                stroke["width"],
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            ))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # 5. 塗られた辺の描画 (The Witness風の太いライン)
        line_w = self.line_width
        for key, color in self.edges.items():
            a = self.vertex_to_widget_coords(*key[0])
            b = self.vertex_to_widget_coords(*key[1])
            painter.setPen(QPen(
                QColor(color.red(), color.green(), color.blue(), 70),
                line_w + 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            ))
            painter.drawLine(a, b)
        for key, color in self.edges.items():
            a = self.vertex_to_widget_coords(*key[0])
            b = self.vertex_to_widget_coords(*key[1])
            painter.setPen(QPen(
                color, line_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            ))
            painter.drawLine(a, b)

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

        # 6. ホバー中の辺のハイライト
        if self.draw_mode in (DrawMode.SNAP_LINE, DrawMode.ERASER) and self.hovered_edge:
            a = self.vertex_to_widget_coords(*self.hovered_edge[0])
            b = self.vertex_to_widget_coords(*self.hovered_edge[1])
            painter.setPen(QPen(
                QColor(255, 255, 255, 120), line_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            ))
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
                    self.adjust_flags = flags
                    self.adjust_start_pos = pos
                    self.adjust_start_rect = list(self.grid_rect)
            return

        if self.draw_mode == DrawMode.SNAP_LINE:
            edge = self.find_nearest_edge(pos)
            if edge is None:
                return
            if event.button() == Qt.MouseButton.RightButton:
                # 右クリック: その辺を消去
                self.apply_edge(edge, "erase")
            elif event.button() == Qt.MouseButton.LeftButton:
                # 左クリック: トグル。以降のドラッグは同じモード(塗り/消し)で辺をなぞる
                self.edge_drag_mode = self.toggle_edge(edge)
                self.last_drag_pos = pos

                # 1操作でのドラッグ追跡状態を初期化
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
            # マス目塗りつぶし
            cell = self.find_cell_at_pos(pos)
            if cell is not None:
                prev_color = self.cell_fills.get(cell)
                if event.button() == Qt.MouseButton.RightButton or prev_color == self.current_color:
                    # すでに同じ色なら消去 (トグル消去)
                    self.cell_fills.pop(cell, None)
                    new_color = None
                else:
                    self.cell_fills[cell] = self.current_color
                    new_color = self.current_color

                self.undo_stack.append({
                    "type": "fill",
                    "cell": cell,
                    "prev_color": prev_color,
                    "new_color": new_color
                })
                self.update()

        elif self.draw_mode == DrawMode.FREEHAND:
            # フリーハンド描画開始
            if event.button() == Qt.MouseButton.LeftButton:
                self.active_stroke = [pos]
                self.update()

        elif self.draw_mode == DrawMode.ERASER:
            # 消しゴム: クリック箇所の塗り辺、セル塗り、手書きを消去
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self.erase_at_pos(pos)

    def mouseMoveEvent(self, event):
        pos = event.position()

        if self.draw_mode == DrawMode.GRID_ADJUST:
            if self.adjust_flags is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                self.apply_grid_adjust(pos)
            else:
                self.setCursor(self.adjust_cursor_for(self.find_adjust_target(pos)))
            return

        if self.draw_mode in (DrawMode.SNAP_LINE, DrawMode.ERASER):
            # ホバー辺の更新
            edge = self.find_nearest_edge(pos)
            if edge != self.hovered_edge:
                self.hovered_edge = edge
                self.update()

        if self.draw_mode == DrawMode.SNAP_LINE:
            if self.edge_drag_mode and (event.buttons() & Qt.MouseButton.LeftButton):
                start = self.last_drag_pos if self.last_drag_pos is not None else pos
                dist = math.hypot(pos.x() - start.x(), pos.y() - start.y())
                steps = max(1, int(dist // 8))

                def get_neighbors(v):
                    tc, tr = v
                    res = []
                    if tc + 1 <= self.cols: res.append((tc + 1, tr))
                    if tc - 1 >= 0: res.append((tc - 1, tr))
                    if tr + 1 <= self.rows: res.append((tc, tr + 1))
                    if tr - 1 >= 0: res.append((tc, tr - 1))
                    return res

                threshold = self.edge_hit_threshold()

                for step_idx in range(1, steps + 1):
                    t = step_idx / steps
                    sub_pos = QPointF(start.x() + (pos.x() - start.x()) * t, start.y() + (pos.y() - start.y()) * t)

                    if self.edge_drag_mode == "paint":
                        # 1. 巻き戻し判定: マウスが1つ前の頂点 (drag_base) に戻った場合
                        if len(self.op_edges) > 1 and self.drag_base is not None:
                            base_pt = self.vertex_to_widget_coords(*self.drag_base)
                            if math.hypot(sub_pos.x() - base_pt.x(), sub_pos.y() - base_pt.y()) <= threshold:
                                last_edge = self.op_edges.pop()
                                self.apply_edge(last_edge, "erase")
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
                                continue

                        # 2. 次の辺へ伸ばす判定 (1つの交点からは1操作で1方向にしか伸ばせない)
                        if len(self.op_edges) == 1:
                            # 最初の1辺のみ: 両端点のどちらからでも伸ばし開始可能
                            init_e = self.op_edges[0]
                            cand_pairs = [(init_e[0], init_e[1]), (init_e[1], init_e[0])]
                            best_match = None
                            best_dist = float("inf")

                            for cand_tip, cand_base in cand_pairs:
                                p_tip = self.vertex_to_widget_coords(*cand_tip)
                                dist_to_tip = math.hypot(sub_pos.x() - p_tip.x(), sub_pos.y() - p_tip.y())
                                for nxt in get_neighbors(cand_tip):
                                    if nxt == cand_base:
                                        continue
                                    p_nxt = self.vertex_to_widget_coords(*nxt)
                                    d = point_to_segment_distance(sub_pos.x(), sub_pos.y(), p_tip.x(), p_tip.y(), p_nxt.x(), p_nxt.y())
                                    dist_to_nxt = math.hypot(sub_pos.x() - p_nxt.x(), sub_pos.y() - p_nxt.y())
                                    if d <= threshold and d < best_dist and dist_to_nxt < dist_to_tip + 20:
                                        best_dist = d
                                        best_match = (cand_tip, nxt)

                            if best_match is not None:
                                c_tip, nxt = best_match
                                new_edge = self.make_edge_key(c_tip, nxt)
                                if new_edge not in self.op_edges:
                                    self.apply_edge(new_edge, "paint")
                                    self.op_edges.append(new_edge)
                                    self.op_vertex_edges.setdefault(c_tip, []).append(new_edge)
                                    self.op_vertex_edges.setdefault(nxt, []).append(new_edge)
                                    self.drag_base = c_tip
                                    self.drag_tip = nxt

                        elif len(self.op_edges) > 1 and self.drag_tip is not None:
                            # 2本目以降: 現在の先端からのみ次の方向へ伸ばす (先端からの次数を1に制限)
                            tip_edges_in_op = self.op_vertex_edges.get(self.drag_tip, [])
                            if len(tip_edges_in_op) < 2:
                                best_next = None
                                best_dist = float("inf")
                                p_tip = self.vertex_to_widget_coords(*self.drag_tip)
                                dist_to_tip = math.hypot(sub_pos.x() - p_tip.x(), sub_pos.y() - p_tip.y())

                                for nxt in get_neighbors(self.drag_tip):
                                    if nxt == self.drag_base:
                                        continue
                                    p_nxt = self.vertex_to_widget_coords(*nxt)
                                    d = point_to_segment_distance(sub_pos.x(), sub_pos.y(), p_tip.x(), p_tip.y(), p_nxt.x(), p_nxt.y())
                                    dist_to_nxt = math.hypot(sub_pos.x() - p_nxt.x(), sub_pos.y() - p_nxt.y())

                                    if d <= threshold and d < best_dist and dist_to_nxt < dist_to_tip + 20:
                                        best_dist = d
                                        best_next = nxt

                                if best_next is not None:
                                    new_edge = self.make_edge_key(self.drag_tip, best_next)
                                    if new_edge not in self.op_edges:
                                        self.apply_edge(new_edge, "paint")
                                        self.op_edges.append(new_edge)
                                        self.op_vertex_edges.setdefault(self.drag_tip, []).append(new_edge)
                                        self.op_vertex_edges.setdefault(best_next, []).append(new_edge)
                                        self.drag_base = self.drag_tip
                                        self.drag_tip = best_next

                    elif self.edge_drag_mode == "erase":
                        edge = self.find_nearest_edge(sub_pos)
                        if edge is not None:
                            self.apply_edge(edge, "erase")

                self.last_drag_pos = pos

        elif self.draw_mode == DrawMode.FREEHAND:
            if self.active_stroke:
                self.active_stroke.append(pos)
                self.update()

        elif self.draw_mode == DrawMode.CELL_FILL:
            # ドラッグで連続塗りつぶし (左ボタン押下時)
            if event.buttons() & Qt.MouseButton.LeftButton:
                cell = self.find_cell_at_pos(pos)
                if cell is not None and self.cell_fills.get(cell) != self.current_color:
                    prev = self.cell_fills.get(cell)
                    self.cell_fills[cell] = self.current_color
                    self.undo_stack.append({
                        "type": "fill", "cell": cell, "prev_color": prev, "new_color": self.current_color
                    })
                    self.update()

        elif self.draw_mode == DrawMode.ERASER:
            # 消しゴムのドラッグなぞり消去
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.erase_at_pos(pos)

    def mouseReleaseEvent(self, event):
        self.edge_drag_mode = None
        self.last_drag_pos = None
        self.drag_tip = None
        self.drag_base = None
        self.op_edges.clear()
        self.op_vertex_edges.clear()
        self.adjust_flags = None
        self.adjust_start_pos = None
        self.adjust_start_rect = None

        if self.draw_mode == DrawMode.FREEHAND:
            if self.active_stroke:
                if len(self.active_stroke) >= 2:
                    self.freehand_strokes.append({
                        "points": list(self.active_stroke),
                        "color": self.current_color,
                        "width": 4
                    })
                    self.undo_stack.append({"type": "add_stroke"})
                self.active_stroke.clear()
                self.update()


# ==============================================================================
# メインウィンドウ (MainWindow)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Witness パズル考察サンドボックス (Witness Sandbox)")
        self.resize(1100, 800)

        # 現在保持しているキャプチャ画像 (BGR) と補正後画像 (BGR)
        self.captured_bgr: np.ndarray | None = None
        self.warped_bgr: np.ndarray | None = None

        # ホットキーシグナラー
        self.hotkey_signaler = HotkeySignaler()
        self.hotkey_signaler.capture_requested.connect(self.action_capture_screen)

        # UI構築
        self._init_ui()

        # アプリ内ショートカットの登録 (F4)
        self.shortcut_f4 = QShortcut(QKeySequence("F4"), self)
        self.shortcut_f4.activated.connect(self.action_capture_screen)

        # 初期サンプル画像をロード
        self.action_load_sample()

        # グローバルホットキー (F4) はUIの描画完了後に安全に登録
        QTimer.singleShot(600, self._register_global_hotkey)

    def center_on_screen(self):
        """ウィンドウをデスクトップ画面の中央に配置する。"""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            geo = self.geometry()
            geo.moveCenter(screen_geo.center())
            self.setGeometry(geo)

    def _init_ui(self):
        # メインコンテナとスタックウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. 画面スタック: [0] 四隅指定画面 (Perspective)  [1] サンドボックス画面 (Sandbox)
        self.stack = QStackedWidget()
        self.perspective_canvas = PerspectiveCanvas()
        self.sandbox_canvas = SandboxCanvas()

        self.stack.addWidget(self.perspective_canvas)
        self.stack.addWidget(self.sandbox_canvas)

        # 2. トップツールバー
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 3. サンドボックス用コントロールバー (スタックがサンドボックスの時に表示)
        self.sandbox_ctrl_panel = self._create_sandbox_controls()

        main_layout.addWidget(self.sandbox_ctrl_panel)
        main_layout.addWidget(self.stack, stretch=1)

        # 4. ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status_guide()

        # シグナル接続
        self.perspective_canvas.points_updated.connect(self._on_perspective_points_updated)
        self.sandbox_canvas.grid_rect_changed.connect(self._on_grid_rect_changed)

    def _create_top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # キャプチャ関連ボタン
        self.btn_capture = QPushButton("📷 画面キャプチャ (F4)")
        self.btn_capture.setStyleSheet("font-weight: bold; background-color: #2b5b84; color: white; padding: 5px 10px;")
        self.btn_capture.clicked.connect(self.action_capture_screen)
        layout.addWidget(self.btn_capture)

        btn_open = QPushButton("📁 画像を開く...")
        btn_open.clicked.connect(self.action_open_file)
        layout.addWidget(btn_open)

        btn_sample = QPushButton("🧩 サンプルパズル読込")
        btn_sample.clicked.connect(self.action_load_sample)
        layout.addWidget(btn_sample)

        # セパレータ
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line)

        # モード切替
        self.btn_goto_perspective = QPushButton("1. 四隅指定モード")
        self.btn_goto_perspective.setCheckable(True)
        self.btn_goto_perspective.setChecked(True)
        self.btn_goto_perspective.clicked.connect(lambda: self.switch_mode(0))
        layout.addWidget(self.btn_goto_perspective)

        btn_reset_pts = QPushButton("↺ 四隅リセット")
        btn_reset_pts.clicked.connect(self.perspective_canvas.reset_points)
        layout.addWidget(btn_reset_pts)

        # 比率選択と自動検知
        layout.addWidget(QLabel("<b>比率:</b>"))
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItem("正方形 (1:1)", "square")
        self.combo_aspect.addItem("マス目比 (列:行)", "grid")
        self.combo_aspect.addItem("元画像比", "original")
        self.combo_aspect.currentIndexChanged.connect(self._on_aspect_changed)
        layout.addWidget(self.combo_aspect)

        self.btn_detect_grid = QPushButton("🔍 マス目検知")
        self.btn_detect_grid.setToolTip("盤面のグリッド線からマス目の数(列・行)と余白を自動推定します")
        self.btn_detect_grid.clicked.connect(self.action_detect_grid)
        layout.addWidget(self.btn_detect_grid)

        self.btn_warp = QPushButton("➡️ 補正してサンドボックスへ")
        self.btn_warp.setStyleSheet("font-weight: bold; background-color: #2d8a4e; color: white; padding: 5px 12px;")
        self.btn_warp.setEnabled(False)
        self.btn_warp.clicked.connect(self.action_execute_warp)
        layout.addWidget(self.btn_warp)

        self.btn_goto_sandbox = QPushButton("2. サンドボックス")
        self.btn_goto_sandbox.setCheckable(True)
        self.btn_goto_sandbox.setEnabled(False)
        self.btn_goto_sandbox.clicked.connect(lambda: self.switch_mode(1))
        layout.addWidget(self.btn_goto_sandbox)

        # モードボタングループ
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.addButton(self.btn_goto_perspective, 0)
        self.mode_btn_group.addButton(self.btn_goto_sandbox, 1)

        layout.addStretch(1)

        # 最前面表示トグル
        self.chk_topmost = QCheckBox("常に最前面表示 (TopMost)")
        self.chk_topmost.toggled.connect(self.action_toggle_topmost)
        layout.addWidget(self.chk_topmost)

        return frame

    def _create_sandbox_controls(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        # ツール選択
        layout.addWidget(QLabel("<b>ツール:</b>"))
        self.btn_tool_snap = QRadioButton("線塗り")
        self.btn_tool_snap.setChecked(True)
        self.btn_tool_snap.toggled.connect(lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.SNAP_LINE))
        layout.addWidget(self.btn_tool_snap)

        self.btn_tool_fill = QRadioButton("マス目塗り")
        self.btn_tool_fill.toggled.connect(lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.CELL_FILL))
        layout.addWidget(self.btn_tool_fill)

        self.btn_tool_free = QRadioButton("自由手書き")
        self.btn_tool_free.toggled.connect(lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.FREEHAND))
        layout.addWidget(self.btn_tool_free)

        self.btn_tool_eraser = QRadioButton("消しゴム")
        self.btn_tool_eraser.toggled.connect(lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.ERASER))
        layout.addWidget(self.btn_tool_eraser)

        self.btn_tool_adjust = QRadioButton("枠調整")
        self.btn_tool_adjust.setToolTip("グリッドの外枠をドラッグして位置・サイズを合わせます (角/辺/内側ドラッグ)")
        self.btn_tool_adjust.toggled.connect(lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.GRID_ADJUST))
        layout.addWidget(self.btn_tool_adjust)

        tool_group = QButtonGroup(self)
        tool_group.addButton(self.btn_tool_snap)
        tool_group.addButton(self.btn_tool_fill)
        tool_group.addButton(self.btn_tool_free)
        tool_group.addButton(self.btn_tool_eraser)
        tool_group.addButton(self.btn_tool_adjust)
        # ツール切替時にステータスバーのガイドも更新 (SandboxCanvas側のモード変更後に呼ばれるよう遅延実行)
        tool_group.buttonToggled.connect(lambda _b, checked: checked and QTimer.singleShot(0, self._update_status_guide))

        # カラー選択
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line1)

        layout.addWidget(QLabel("<b>色:</b>"))
        self.btn_color_indicator = QPushButton()
        self.btn_color_indicator.setFixedSize(24, 24)
        self.btn_color_indicator.setStyleSheet("background-color: #FFAA00; border: 1px solid #ffffff; border-radius: 4px;")
        self.btn_color_indicator.clicked.connect(self.action_choose_color)
        layout.addWidget(self.btn_color_indicator)

        # プリセットカラー
        presets = [
            ("#FFAA00", "オレンジ"),
            ("#FFFF00", "イエロー"),
            ("#00D0FF", "シアン"),
            ("#00FF66", "グリーン"),
            ("#FF3366", "ピンク"),
            ("#FFFFFF", "ホワイト"),
        ]
        for hex_col, name in presets:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setToolTip(name)
            btn.setStyleSheet(f"background-color: {hex_col}; border: 1px solid #555555; border-radius: 3px;")
            btn.clicked.connect(lambda checked, c=hex_col: self.action_set_preset_color(c))
            layout.addWidget(btn)

        # グリッド設定
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line2)

        layout.addWidget(QLabel("<b>グリッド:</b>"))

        btn_detect_sb = QPushButton("🔍 自動検知")
        btn_detect_sb.setToolTip("盤面のマス目を自動検出します")
        btn_detect_sb.clicked.connect(self.action_detect_grid)
        layout.addWidget(btn_detect_sb)

        layout.addWidget(QLabel("列:"))
        self.slider_cols = QSlider(Qt.Orientation.Horizontal)
        self.slider_cols.setRange(1, 20)
        self.slider_cols.setValue(4)
        self.slider_cols.setFixedWidth(70)
        self.slider_cols.setToolTip("盤面の列数を変更します (1〜20)")
        self.lbl_cols_val = QLabel("4")
        self.lbl_cols_val.setFixedWidth(16)
        self.slider_cols.valueChanged.connect(self._on_cols_slider_changed)
        layout.addWidget(self.slider_cols)
        layout.addWidget(self.lbl_cols_val)

        layout.addWidget(QLabel("行:"))
        self.slider_rows = QSlider(Qt.Orientation.Horizontal)
        self.slider_rows.setRange(1, 20)
        self.slider_rows.setValue(4)
        self.slider_rows.setFixedWidth(70)
        self.slider_rows.setToolTip("盤面の行数を変更します (1〜20)")
        self.lbl_rows_val = QLabel("4")
        self.lbl_rows_val.setFixedWidth(16)
        self.slider_rows.valueChanged.connect(self._on_rows_slider_changed)
        layout.addWidget(self.slider_rows)
        layout.addWidget(self.lbl_rows_val)


        layout.addWidget(QLabel("マージン"))
        self.slider_margin = QSlider(Qt.Orientation.Horizontal)
        self.slider_margin.setRange(0, 30)
        self.slider_margin.setValue(5)
        self.slider_margin.setFixedWidth(70)
        self.slider_margin.valueChanged.connect(lambda v: self.sandbox_canvas.set_margin_percent(v / 100.0))
        layout.addWidget(self.slider_margin)

        layout.addStretch(1)

        # 編集アクション (Undo, Clear, Save)
        btn_undo = QPushButton("↩ 1つ戻る (Ctrl+Z)")
        btn_undo.clicked.connect(self.sandbox_canvas.undo)
        layout.addWidget(btn_undo)

        btn_clear = QPushButton("🗑 全消去")
        btn_clear.clicked.connect(self.sandbox_canvas.clear_all)
        layout.addWidget(btn_clear)

        btn_save = QPushButton("💾 保存")
        btn_save.clicked.connect(self.action_save_sandbox_image)
        layout.addWidget(btn_save)

        # 初期状態は四隅指定モードなのでコントロールバーは非表示
        frame.setVisible(False)
        return frame

    def _register_global_hotkey(self):
        """グローバルホットキー (F4) の安全な登録"""
        try:
            import keyboard
            keyboard.add_hotkey("F4", lambda: self.hotkey_signaler.capture_requested.emit())
            print("[Info] グローバルホットキー (F4) を登録しました。", flush=True)
        except Exception as e:
            print(f"[Info] グローバルホットキーの登録をスキップしました (アプリ内F4キーやボタンで操作可能): {e}", flush=True)

    def closeEvent(self, event):
        """アプリ終了処理"""
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        event.accept()

    def keyPressEvent(self, event):
        """ウィンドウ内ショートカットキーの処理"""
        # F4: キャプチャ
        if event.key() == Qt.Key.Key_F4:
            self.action_capture_screen()
            return
        # Ctrl+Z: Undo
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if self.stack.currentIndex() == 1:
                self.sandbox_canvas.undo()
                return
        super().keyPressEvent(event)

    def _update_status_guide(self):
        """現在の状態に応じたユーザーガイダンスをステータスバーに表示する。"""
        if self.stack.currentIndex() == 0:
            count = len(self.perspective_canvas.points)
            if count < 4:
                self.status_bar.showMessage(
                    f"【四隅指定モード】パズル盤面の4隅を順にクリックしてください (指定済み: {count}/4 点)。ドラッグで微調整、右クリックで点削除が可能です。"
                )
            else:
                self.status_bar.showMessage(
                    "【四隅指定完了】4点が指定されました。「マス目検知」でグリッド数を推定するか、「補正してサンドボックスへ」を実行してください。"
                )
        else:
            mode = self.sandbox_canvas.draw_mode
            if mode == DrawMode.SNAP_LINE:
                self.status_bar.showMessage(
                    "【サンドボックス: 線塗り】グリッドの線をクリックで塗り、塗り済みの線をもう一度クリックで消去。ドラッグでなぞって連続塗り/消去できます。"
                )
            elif mode == DrawMode.CELL_FILL:
                self.status_bar.showMessage(
                    "【サンドボックス: マス目塗り】マス目を左クリックして半透明塗りつぶし。もう一度クリックすると消去します。"
                )
            elif mode == DrawMode.FREEHAND:
                self.status_bar.showMessage(
                    "【サンドボックス: 自由手書き】マウスをドラッグしてメモや印を自由に書き込めます。"
                )
            elif mode == DrawMode.GRID_ADJUST:
                self.status_bar.showMessage(
                    "【サンドボックス: 枠調整】オレンジの枠の角・辺をドラッグでサイズ変更、内側をドラッグで全体を移動してマス目に合わせます。"
                )
            elif mode == DrawMode.ERASER:
                self.status_bar.showMessage(
                    "【サンドボックス: 消しゴム】マス目塗り、塗った線、手書きをクリックまたはドラッグして消去します。"
                )

    # --------------------------------------------------------------------------
    # アクション・イベントハンドラ
    # --------------------------------------------------------------------------

    def switch_mode(self, index: int):
        """四隅指定画面 (0) と サンドボックス画面 (1) を切り替える。"""
        self.stack.setCurrentIndex(index)
        self.sandbox_ctrl_panel.setVisible(index == 1)
        self.btn_goto_perspective.setChecked(index == 0)
        self.btn_goto_sandbox.setChecked(index == 1)
        self._update_status_guide()

    def _on_perspective_points_updated(self, pts):
        """四隅の指定数が変わった時の処理"""
        self.btn_warp.setEnabled(len(pts) == 4)
        self.btn_detect_grid.setEnabled(len(pts) == 4)
        self._update_status_guide()

    def action_capture_screen(self):
        """
        F4 / ボタンで、前面にあるウィンドウ (ゲーム画面) のみをキャプチャする。
        本アプリは撮影前に非表示にし、描画から確実に消えるまで待ってから撮る。
        """
        own_hwnd = 0
        target_hwnd = 0
        try:
            own_hwnd = int(self.winId())
            # グローバルホットキー経由ではゲームが前面のまま。ここで対象を確定する
            target_hwnd = int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            pass

        was_visible = self.isVisible()
        if was_visible:
            self.hide()

        # 非表示が画面に反映されるまで待つ (processEvents のみでは写り込む)
        deadline = time.time() + 0.25
        while time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)

        try:
            # 自アプリが前面だった場合は、隠した後に前面になったウィンドウを対象にする
            if not target_hwnd or target_hwnd == own_hwnd:
                target_hwnd = int(ctypes.windll.user32.GetForegroundWindow())

            bgr = capture_foreground_window_bgr(target_hwnd, exclude_hwnd=own_hwnd)

            self.captured_bgr = bgr
            self.perspective_canvas.set_image(bgr)
            self.switch_mode(0)
            self.btn_goto_sandbox.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "キャプチャエラー", f"画面キャプチャに失敗しました:\n{e}")
        finally:
            if was_visible:
                self.show()
                self.raise_()
                self.activateWindow()

    def action_open_file(self):
        """ローカル画像ファイルを開く。"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "パズル画像を開く", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not filepath:
            return

        # 日本語パス対応の読み込み
        stream = open(filepath, "rb")
        bytes_data = bytearray(stream.read())
        stream.close()
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img_bgr = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            QMessageBox.warning(self, "エラー", "画像の読み込みに失敗しました。")
            return

        self.captured_bgr = img_bgr
        self.perspective_canvas.set_image(img_bgr)
        self.switch_mode(0)
        self.btn_goto_sandbox.setEnabled(False)

    def action_load_sample(self):
        """The Witness風の斜めパズルサンプル画像を生成してロードする。"""
        sample_bgr = create_sample_witness_image()
        self.captured_bgr = sample_bgr
        self.perspective_canvas.set_image(sample_bgr)
        # サンプル用の斜め四隅をプリセット (すぐに補正を体験可能)
        self.perspective_canvas.points = [
            QPointF(220, 110),
            QPointF(780, 70),
            QPointF(890, 580),
            QPointF(150, 610)
        ]
        self.perspective_canvas.points_updated.emit(self.perspective_canvas.points)
        self.perspective_canvas.update()
        self.switch_mode(0)
        self.btn_goto_sandbox.setEnabled(False)

    def action_detect_grid(self):
        """
        現在の画像と指定隅からパズルのマス目（列数・行数・マージン）を自動検知して設定する。
        """
        pts = None
        if len(self.perspective_canvas.points) == 4:
            pts = np.array([[p.x(), p.y()] for p in self.perspective_canvas.points], dtype="float32")

        img_to_analyze = self.captured_bgr
        if pts is None and self.warped_bgr is not None:
            img_to_analyze = self.warped_bgr

        if img_to_analyze is None:
            QMessageBox.information(self, "マス目検知", "検知対象の画像が読み込まれていません。")
            return

        cols, rows, margin = detect_grid_dimensions(img_to_analyze, pts)
        self.slider_cols.blockSignals(True)
        self.slider_rows.blockSignals(True)
        self.slider_cols.setValue(cols)
        self.slider_rows.setValue(rows)
        self.slider_cols.blockSignals(False)
        self.slider_rows.blockSignals(False)
        self.lbl_cols_val.setText(str(cols))
        self.lbl_rows_val.setText(str(rows))
        self.slider_margin.setValue(int(round(margin * 100)))
        self.sandbox_canvas.set_grid_dimensions(rows, cols)
        self.sandbox_canvas.set_margin_percent(margin)

        msg = f"✅ マス目を自動検知しました: {cols}列 × {rows}行 (マージン: {int(round(margin * 100))}%)"
        self.status_bar.showMessage(msg, 6000)

    def _on_aspect_changed(self):
        """アスペクト比選択が変更されたときの処理"""
        # サンドボックス表示中で、四隅データがあれば即座に再補正を適用
        if self.stack.currentIndex() == 1 and self.captured_bgr is not None and len(self.perspective_canvas.points) == 4:
            self.action_execute_warp()

    def action_execute_warp(self):
        """
        指定された4隅をもとに透視投影変換 (パース補正) を実行し、
        サンドボックス画面へ遷移する。
        """
        if self.captured_bgr is None or len(self.perspective_canvas.points) != 4:
            return

        pts = np.array([[p.x(), p.y()] for p in self.perspective_canvas.points], dtype="float32")

        try:
            # 補正前に自動でグリッド線（マス目数）を自動検知
            detected_cols, detected_rows, detected_margin = detect_grid_dimensions(self.captured_bgr, pts)
            self.slider_cols.blockSignals(True)
            self.slider_rows.blockSignals(True)
            self.slider_cols.setValue(detected_cols)
            self.slider_rows.setValue(detected_rows)
            self.slider_cols.blockSignals(False)
            self.slider_rows.blockSignals(False)
            self.lbl_cols_val.setText(str(detected_cols))
            self.lbl_rows_val.setText(str(detected_rows))
            self.slider_margin.setValue(int(round(detected_margin * 100)))

            aspect_mode = self.combo_aspect.currentData() or "square"
            warped = warp_perspective(
                self.captured_bgr,
                pts,
                aspect_mode=aspect_mode,
                cols=detected_cols,
                rows=detected_rows
            )
            self.warped_bgr = warped
            self.sandbox_canvas.set_warped_image(warped)
            self.sandbox_canvas.set_grid_dimensions(detected_rows, detected_cols)
            self.sandbox_canvas.set_margin_percent(detected_margin)

            # サンドボックスモードへ切り替え
            self.btn_goto_sandbox.setEnabled(True)
            self.switch_mode(1)
            self.status_bar.showMessage(
                f"✅ 補正完了: {detected_cols}×{detected_rows} マスを検出 (比率: {self.combo_aspect.currentText()})",
                6000
            )
        except Exception as e:
            QMessageBox.critical(self, "補正エラー", f"パース補正の実行に失敗しました:\n{e}")

    def _on_grid_rect_changed(self):
        """枠調整で外枠が変わったとき、マージンスライダーを左マージンの値に同期する (信号は発火させない)。"""
        self.slider_margin.blockSignals(True)
        self.slider_margin.setValue(int(round(self.sandbox_canvas.grid_rect[0] * 100)))
        self.slider_margin.blockSignals(False)

    @property
    def spin_cols(self):
        """後方互換用エイリアス"""
        return self.slider_cols

    @property
    def spin_rows(self):
        """後方互換用エイリアス"""
        return self.slider_rows

    def _on_cols_slider_changed(self, val: int):
        """列スライダー変更ハンドラ"""
        self.lbl_cols_val.setText(str(val))
        self.sandbox_canvas.set_grid_dimensions(self.slider_rows.value(), val)

    def _on_rows_slider_changed(self, val: int):
        """行スライダー変更ハンドラ"""
        self.lbl_rows_val.setText(str(val))
        self.sandbox_canvas.set_grid_dimensions(val, self.slider_cols.value())

    def _on_grid_size_changed(self):
        """グリッドサイズ変更ハンドラ (外部・互換用)"""
        cols = self.slider_cols.value()
        rows = self.slider_rows.value()
        self.lbl_cols_val.setText(str(cols))
        self.lbl_rows_val.setText(str(rows))
        self.sandbox_canvas.set_grid_dimensions(rows, cols)


    def action_toggle_topmost(self, checked: bool):
        """最前面表示トグル"""
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def action_choose_color(self):
        """カラーピッカーダイアログを表示する。"""
        col = QColorDialog.getColor(self.sandbox_canvas.current_color, self, "ペンの色を選択")
        if col.isValid():
            self.sandbox_canvas.set_color(col)
            self.btn_color_indicator.setStyleSheet(
                f"background-color: {col.name()}; border: 1px solid #ffffff; border-radius: 4px;"
            )

    def action_set_preset_color(self, hex_color: str):
        """プリセット色を設定する。"""
        col = QColor(hex_color)
        self.sandbox_canvas.set_color(col)
        self.btn_color_indicator.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #ffffff; border-radius: 4px;"
        )

    def action_save_sandbox_image(self):
        """現在のサンドボックスキャンバスの内容を画像ファイルに保存する。"""
        if self.sandbox_canvas.pixmap.isNull():
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "考察画像を保存", "witness_sandbox_solution.png", "PNG画像 (*.png);;JPEG画像 (*.jpg)"
        )
        if not filepath:
            return

        # キャンバス全体をレンダリングして保存
        img = self.sandbox_canvas.grab()
        if img.save(filepath):
            QMessageBox.information(self, "保存完了", f"画像を保存しました:\n{filepath}")
        else:
            QMessageBox.warning(self, "保存失敗", "画像の保存に失敗しました。")


# ==============================================================================
# メインエントリーポイント
# ==============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ダークテーマ調のスタイルシート設定
    dark_style = """
        QMainWindow, QWidget {
            background-color: #1e1e24;
            color: #e0e0e0;
            font-family: 'Segoe UI', Meiryo, sans-serif;
        }
        QPushButton {
            background-color: #33333d;
            color: #ffffff;
            border: 1px solid #4f4f5c;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #444452;
            border-color: #00bcd4;
        }
        QPushButton:pressed {
            background-color: #25252d;
        }
        QPushButton:disabled {
            background-color: #26262c;
            color: #666670;
            border-color: #363640;
        }
        QRadioButton, QCheckBox {
            color: #e0e0e0;
            spacing: 5px;
        }
        QSpinBox {
            background-color: #2b2b36;
            color: #ffffff;
            border: 1px solid #4f4f5c;
            border-radius: 3px;
            padding: 2px 4px;
        }
        QStatusBar {
            background-color: #18181c;
            color: #aaaaaa;
        }
        QFrame[frameShape="4"], QFrame[frameShape="5"] {
            color: #444450;
        }
    """
    app.setStyleSheet(dark_style)

    window = MainWindow()
    window.center_on_screen()
    window.show()
    window.showNormal()
    window.raise_()
    window.activateWindow()
    app.processEvents()

    # Windows API で確実に最前面・アクティブ化
    try:
        hwnd = int(window.winId())
        ctypes.windll.user32.ShowWindow(hwnd, 1)  # 1 = SW_SHOWNORMAL
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f"[Warn] フォアグラウンド設定警告: {e}", flush=True)

    print("[Info] ウィンドウを表示しました。GUIイベントループを開始します。", flush=True)
    exit_code = app.exec()
    print(f"[Info] アプリケーションが正常終了しました (exit code: {exit_code})", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
