"""
メインウィンドウ UI・ホットキーイベント中継・アプリケーション起動モジュール。
"""

import sys
import time

import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app import capture, vision
from app.canvas import DrawMode, PerspectiveCanvas, SandboxCanvas

DARK_STYLE_QSS = """
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
    QStatusBar {
        background-color: #18181c;
        color: #aaaaaa;
    }
    QFrame[frameShape="4"], QFrame[frameShape="5"] {
        color: #444450;
    }
"""


# ==============================================================================
# グローバルホットキー通信用ヘルパー
# ==============================================================================


class HotkeySignaler(QObject):
    """ネイティブホットキー受信を Qt シグナルへ中継するヘルパー。"""

    capture_requested = pyqtSignal()


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
        self._registered_hotkey_hwnd: int | None = None
        self._is_closed: bool = False

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
        self._hotkey_init_timer = QTimer(self)
        self._hotkey_init_timer.setSingleShot(True)
        self._hotkey_init_timer.timeout.connect(self._register_global_hotkey)
        self._hotkey_init_timer.start(600)

    def center_on_screen(self):
        """ウィンドウをデスクトップ画面の中央に配置する。"""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            geo = self.geometry()
            geo.moveCenter(screen_geo.center())
            self.setGeometry(geo)

    def _init_ui(self):
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
        self.perspective_canvas.points_updated.connect(
            self._on_perspective_points_updated
        )
        self.sandbox_canvas.grid_rect_changed.connect(self._on_grid_rect_changed)

    def _create_top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.btn_capture = QPushButton("📷 画面キャプチャ (F4)")
        self.btn_capture.setStyleSheet(
            "font-weight: bold; background-color: #2b5b84; color: white; padding: 5px 10px;"
        )
        self.btn_capture.clicked.connect(self.action_capture_screen)
        layout.addWidget(self.btn_capture)

        btn_open = QPushButton("📁 画像を開く...")
        btn_open.clicked.connect(self.action_open_file)
        layout.addWidget(btn_open)

        btn_sample = QPushButton("🧩 サンプルパズル読込")
        btn_sample.clicked.connect(self.action_load_sample)
        layout.addWidget(btn_sample)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line)

        self.btn_goto_perspective = QPushButton("1. 四隅指定モード")
        self.btn_goto_perspective.setCheckable(True)
        self.btn_goto_perspective.setChecked(True)
        self.btn_goto_perspective.clicked.connect(lambda: self.switch_mode(0))
        layout.addWidget(self.btn_goto_perspective)

        btn_reset_pts = QPushButton("↺ 四隅リセット")
        btn_reset_pts.clicked.connect(self.perspective_canvas.reset_points)
        layout.addWidget(btn_reset_pts)

        layout.addWidget(QLabel("<b>比率:</b>"))
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItem("正方形 (1:1)", "square")
        self.combo_aspect.addItem("マス目比 (列:行)", "grid")
        self.combo_aspect.addItem("元画像比", "original")
        self.combo_aspect.currentIndexChanged.connect(self._on_aspect_changed)
        layout.addWidget(self.combo_aspect)

        self.btn_detect_grid = QPushButton("🔍 マス目検知")
        self.btn_detect_grid.setToolTip(
            "盤面のグリッド線からマス目の数(列・行)と余白を自動推定します"
        )
        self.btn_detect_grid.clicked.connect(self.action_detect_grid)
        layout.addWidget(self.btn_detect_grid)

        self.btn_warp = QPushButton("➡️ 補正してサンドボックスへ")
        self.btn_warp.setStyleSheet(
            "font-weight: bold; background-color: #2d8a4e; color: white; padding: 5px 12px;"
        )
        self.btn_warp.setEnabled(False)
        self.btn_warp.clicked.connect(self.action_execute_warp)
        layout.addWidget(self.btn_warp)

        self.btn_goto_sandbox = QPushButton("2. サンドボックス")
        self.btn_goto_sandbox.setCheckable(True)
        self.btn_goto_sandbox.setEnabled(False)
        self.btn_goto_sandbox.clicked.connect(lambda: self.switch_mode(1))
        layout.addWidget(self.btn_goto_sandbox)

        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.addButton(self.btn_goto_perspective, 0)
        self.mode_btn_group.addButton(self.btn_goto_sandbox, 1)

        layout.addStretch(1)

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

        layout.addWidget(QLabel("<b>ツール:</b>"))
        self.btn_tool_snap = QRadioButton("線塗り")
        self.btn_tool_snap.setChecked(True)
        self.btn_tool_snap.toggled.connect(
            lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.SNAP_LINE)
        )
        layout.addWidget(self.btn_tool_snap)

        self.btn_tool_fill = QRadioButton("マス目塗り")
        self.btn_tool_fill.toggled.connect(
            lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.CELL_FILL)
        )
        layout.addWidget(self.btn_tool_fill)

        self.btn_tool_free = QRadioButton("自由手書き")
        self.btn_tool_free.toggled.connect(
            lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.FREEHAND)
        )
        layout.addWidget(self.btn_tool_free)

        self.btn_tool_eraser = QRadioButton("消しゴム")
        self.btn_tool_eraser.toggled.connect(
            lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.ERASER)
        )
        layout.addWidget(self.btn_tool_eraser)

        self.btn_tool_adjust = QRadioButton("枠調整")
        self.btn_tool_adjust.setToolTip(
            "グリッドの外枠をドラッグして位置・サイズを合わせます (角/辺/内側ドラッグ)"
        )
        self.btn_tool_adjust.toggled.connect(
            lambda c: c and self.sandbox_canvas.set_draw_mode(DrawMode.GRID_ADJUST)
        )
        layout.addWidget(self.btn_tool_adjust)

        tool_group = QButtonGroup(self)
        tool_group.addButton(self.btn_tool_snap)
        tool_group.addButton(self.btn_tool_fill)
        tool_group.addButton(self.btn_tool_free)
        tool_group.addButton(self.btn_tool_eraser)
        tool_group.addButton(self.btn_tool_adjust)
        tool_group.buttonToggled.connect(
            lambda _b, checked: checked and QTimer.singleShot(0, self._update_status_guide)
        )

        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line1)

        layout.addWidget(QLabel("<b>色:</b>"))
        self.btn_color_indicator = QPushButton()
        self.btn_color_indicator.setFixedSize(24, 24)
        self.btn_color_indicator.setStyleSheet(
            "background-color: #FFAA00; border: 1px solid #ffffff; border-radius: 4px;"
        )
        self.btn_color_indicator.clicked.connect(self.action_choose_color)
        layout.addWidget(self.btn_color_indicator)

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
            btn.setStyleSheet(
                f"background-color: {hex_col}; border: 1px solid #555555; border-radius: 3px;"
            )
            btn.clicked.connect(
                lambda checked, c=hex_col: self.action_set_preset_color(c)
            )
            layout.addWidget(btn)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(line2)

        layout.addWidget(QLabel("<b>グリッド:</b>"))

        btn_detect_sb = QPushButton("🔍 自動検知")
        btn_detect_sb.setToolTip("盤面のマス目を自動検出します")
        btn_detect_sb.clicked.connect(self.action_detect_grid)
        layout.addWidget(btn_detect_sb)

        self.chk_unlock_size = QCheckBox("制限解除")
        self.chk_unlock_size.setToolTip(
            "チェックを入れると9以上のサイズや1〜2のサイズが選択可能になります"
        )
        self.chk_unlock_size.stateChanged.connect(self._on_unlock_size_changed)
        layout.addWidget(self.chk_unlock_size)

        layout.addWidget(QLabel("列:"))
        self.slider_cols = QSlider(Qt.Orientation.Horizontal)
        self.slider_cols.setRange(3, 8)
        self.slider_cols.setValue(4)
        self.slider_cols.setFixedWidth(70)
        self.slider_cols.setToolTip("盤面の列数を変更します (3〜8)")
        self.lbl_cols_val = QLabel("4")
        self.lbl_cols_val.setFixedWidth(16)
        self.slider_cols.valueChanged.connect(self._on_cols_slider_changed)
        layout.addWidget(self.slider_cols)
        layout.addWidget(self.lbl_cols_val)

        layout.addWidget(QLabel("行:"))
        self.slider_rows = QSlider(Qt.Orientation.Horizontal)
        self.slider_rows.setRange(3, 8)
        self.slider_rows.setValue(4)
        self.slider_rows.setFixedWidth(70)
        self.slider_rows.setToolTip("盤面の行数を変更します (3〜8)")
        self.lbl_rows_val = QLabel("4")
        self.lbl_rows_val.setFixedWidth(16)
        self.slider_rows.valueChanged.connect(self._on_rows_slider_changed)
        layout.addWidget(self.slider_rows)
        layout.addWidget(self.lbl_rows_val)

        layout.addWidget(QLabel("マージン"))
        self._margin_slider_dragging = False
        self.slider_margin = QSlider(Qt.Orientation.Horizontal)
        self.slider_margin.setRange(0, 30)
        self.slider_margin.setValue(5)
        self.slider_margin.setFixedWidth(70)
        self.slider_margin.sliderPressed.connect(self._on_margin_slider_pressed)
        self.slider_margin.valueChanged.connect(self._on_margin_slider_changed)
        self.slider_margin.sliderReleased.connect(self._on_margin_slider_released)
        layout.addWidget(self.slider_margin)

        layout.addStretch(1)

        btn_undo = QPushButton("↩ 1つ戻る (Ctrl+Z)")
        btn_undo.clicked.connect(self.sandbox_canvas.undo)
        layout.addWidget(btn_undo)

        self.btn_clear = QPushButton("🗑 全消去")
        self.btn_clear.clicked.connect(
            lambda _checked=False: self.sandbox_canvas.clear_all(clear_history=False)
        )
        layout.addWidget(self.btn_clear)

        btn_save = QPushButton("💾 保存")
        btn_save.clicked.connect(self.action_save_sandbox_image)
        layout.addWidget(btn_save)

        frame.setVisible(False)
        return frame

    def _register_global_hotkey(self) -> bool:
        """Win32 RegisterHotKey によりグローバルホットキー (F4) を安全に登録する。"""
        self._unregister_global_hotkey()
        if self._is_closed:
            return False
        hwnd = 0
        try:
            hwnd = int(self.winId())
        except Exception:
            hwnd = 0
        ok = capture.register_f4_hotkey(hwnd)
        if ok:
            self._registered_hotkey_hwnd = hwnd
        return ok

    def _unregister_global_hotkey(self):
        """登録済みの Win32 グローバルホットキーを解除する。"""
        hwnd = self._registered_hotkey_hwnd
        self._registered_hotkey_hwnd = None
        if hwnd:
            capture.unregister_f4_hotkey(hwnd)

    def nativeEvent(self, eventType, message):
        """Windows ネイティブメッセージ (WM_HOTKEY) を受信して画面キャプチャを発火する。"""
        try:
            if capture.is_f4_hotkey_message(bytes(eventType), int(message)):
                QTimer.singleShot(0, self.hotkey_signaler.capture_requested.emit)
                return True, 0
        except Exception:
            pass
        return super().nativeEvent(eventType, message)

    def closeEvent(self, event):
        """アプリ終了処理"""
        self._is_closed = True
        self._hotkey_init_timer.stop()
        self._unregister_global_hotkey()
        event.accept()

    def keyPressEvent(self, event):
        """ウィンドウ内ショートカットキーの処理"""
        if event.key() == Qt.Key.Key_F4:
            self.action_capture_screen()
            return
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_Z
        ):
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
        try:
            own_hwnd = int(self.winId())
        except Exception:
            own_hwnd = 0
        target_hwnd = capture.get_foreground_window_hwnd()

        was_visible = self.isVisible()
        if was_visible:
            self.hide()

        deadline = time.time() + 0.25
        while time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)

        try:
            if not target_hwnd or target_hwnd == own_hwnd:
                target_hwnd = capture.get_foreground_window_hwnd()

            bgr = capture.capture_foreground_window_bgr(
                target_hwnd or None, exclude_hwnd=own_hwnd or None
            )
            self._load_captured_image_to_perspective(bgr)
        except Exception as e:
            QMessageBox.critical(
                self, "キャプチャエラー", f"画面キャプチャに失敗しました:\n{e}"
            )
        finally:
            if was_visible:
                self.show()
                self.raise_()
                self.activateWindow()
                if (
                    self._registered_hotkey_hwnd is not None
                    and int(self.winId()) != self._registered_hotkey_hwnd
                ):
                    self._register_global_hotkey()

    def _load_captured_image_to_perspective(
        self, img_bgr: np.ndarray, preset_points: list[QPointF] | None = None
    ):
        """読み込んだ画像を四隅指定キャンバスへセットしてモードを初期化する。"""
        self.captured_bgr = img_bgr
        self.perspective_canvas.set_image(img_bgr)
        if preset_points:
            self.perspective_canvas.points = list(preset_points)
            self.perspective_canvas.points_updated.emit(self.perspective_canvas.points)
            self.perspective_canvas.update()
        self.switch_mode(0)
        self.btn_goto_sandbox.setEnabled(False)

    def action_open_file(self):
        """ローカル画像ファイルを開く。"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "パズル画像を開く", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not filepath:
            return

        try:
            img_bgr = vision.load_image_bgr(filepath)
        except (OSError, ValueError) as e:
            QMessageBox.warning(
                self, "画像読込エラー", f"画像の読み込みに失敗しました:\n{e}"
            )
            return

        self._load_captured_image_to_perspective(img_bgr)

    def action_load_sample(self):
        """The Witness風の斜めパズルサンプル画像を生成してロードする。"""
        sample_bgr = vision.create_sample_witness_image()
        preset_points = [
            QPointF(220, 110),
            QPointF(780, 70),
            QPointF(890, 580),
            QPointF(150, 610),
        ]
        self._load_captured_image_to_perspective(sample_bgr, preset_points=preset_points)

    def _apply_detected_grid_settings(self, cols: int, rows: int, margin: float):
        """自動検知された列数・行数・マージン割合をUIスライダーとキャンバスへ一括反映する。"""
        if cols > 8 or rows > 8 or cols < 3 or rows < 3:
            self.chk_unlock_size.setChecked(True)
        self.slider_cols.blockSignals(True)
        self.slider_rows.blockSignals(True)
        self.slider_cols.setValue(cols)
        self.slider_rows.setValue(rows)
        self.slider_cols.blockSignals(False)
        self.slider_rows.blockSignals(False)
        self.lbl_cols_val.setText(str(cols))
        self.lbl_rows_val.setText(str(rows))
        self.slider_margin.blockSignals(True)
        self.slider_margin.setValue(int(round(margin * 100)))
        self.slider_margin.blockSignals(False)
        self.sandbox_canvas.set_grid_dimensions(rows, cols)
        self.sandbox_canvas.set_margin_percent(margin)

    def action_detect_grid(self):
        """現在の画像と指定隅からパズルのマス目（列数・行数・マージン）を自動検知して設定する。"""
        pts = None
        if len(self.perspective_canvas.points) == 4:
            pts = np.array(
                [[p.x(), p.y()] for p in self.perspective_canvas.points],
                dtype="float32",
            )
            try:
                vision.validate_quad_points(pts)
            except ValueError as e:
                QMessageBox.warning(
                    self, "四隅指定エラー", f"四隅の指定が不正です:\n{e}"
                )
                return

        img_to_analyze = self.captured_bgr
        if pts is None and self.warped_bgr is not None:
            img_to_analyze = self.warped_bgr

        if img_to_analyze is None:
            QMessageBox.information(
                self, "マス目検知", "検知対象の画像が読み込まれていません。"
            )
            return

        cols, rows, margin = vision.detect_grid_dimensions(img_to_analyze, pts)
        self._apply_detected_grid_settings(cols, rows, margin)

        msg = f"✅ マス目を自動検知しました: {cols}列 × {rows}行 (マージン: {int(round(margin * 100))}%)"
        self.status_bar.showMessage(msg, 6000)

    def _on_aspect_changed(self):
        """アスペクト比選択が変更されたときの処理"""
        if (
            self.stack.currentIndex() == 1
            and self.captured_bgr is not None
            and len(self.perspective_canvas.points) == 4
        ):
            self.action_execute_warp()

    def action_execute_warp(self):
        """指定された4隅をもとに透視投影変換 (パース補正) を実行し、サンドボックス画面へ遷移する。"""
        if self.captured_bgr is None or len(self.perspective_canvas.points) != 4:
            return

        pts = np.array(
            [[p.x(), p.y()] for p in self.perspective_canvas.points], dtype="float32"
        )

        try:
            vision.validate_quad_points(pts)
            detected_cols, detected_rows, detected_margin = vision.detect_grid_dimensions(
                self.captured_bgr, pts
            )

            aspect_mode = self.combo_aspect.currentData() or "square"
            warped = vision.warp_perspective(
                self.captured_bgr,
                pts,
                aspect_mode=aspect_mode,
                cols=detected_cols,
                rows=detected_rows,
            )
            self.warped_bgr = warped
            self.sandbox_canvas.set_warped_image(warped)
            self._apply_detected_grid_settings(
                detected_cols, detected_rows, detected_margin
            )

            self.btn_goto_sandbox.setEnabled(True)
            self.switch_mode(1)
            self.status_bar.showMessage(
                f"✅ 補正完了: {detected_cols}×{detected_rows} マスを検出 (比率: {self.combo_aspect.currentText()})",
                6000,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "補正エラー", f"パース補正の実行に失敗しました:\n{e}"
            )

    def _on_margin_slider_pressed(self):
        """マージンスライダーのドラッグ開始時に Undo トランザクションを開始する。"""
        self._margin_slider_dragging = True
        self.sandbox_canvas.begin_undo_transaction()

    def _on_margin_slider_changed(self, val: int):
        """マージンスライダー値変更ハンドラ (ドラッグ中はトランザクション継続、単発変更は個別記録)。"""
        self.sandbox_canvas.set_margin_percent(
            val / 100.0, record_undo=not self._margin_slider_dragging
        )

    def _on_margin_slider_released(self):
        """マージンスライダーのドラッグ終了時に Undo トランザクションを確定する。"""
        self._margin_slider_dragging = False
        self.sandbox_canvas.commit_undo_transaction()

    def _on_grid_rect_changed(self):
        """枠調整やUndoで外枠が変わったとき、4辺の平均マージンをスライダー表示に同期する (信号は発火させない)。"""
        l, t, r, b = self.sandbox_canvas.grid_rect
        avg_margin_pct = int(round(((l + t + (1.0 - r) + (1.0 - b)) / 4.0) * 100))
        clamped = max(
            self.slider_margin.minimum(),
            min(self.slider_margin.maximum(), avg_margin_pct),
        )
        self.slider_margin.blockSignals(True)
        self.slider_margin.setValue(clamped)
        self.slider_margin.blockSignals(False)

    def _on_unlock_size_changed(self, state: int):
        """制限解除チェックボックスの変更ハンドラ"""
        if state == Qt.CheckState.Checked.value:
            self.slider_cols.setRange(1, 20)
            self.slider_rows.setRange(1, 20)
            self.slider_cols.setToolTip("盤面の列数を変更します (1〜20)")
            self.slider_rows.setToolTip("盤面の行数を変更します (1〜20)")
        else:
            self.slider_cols.setRange(3, 8)
            self.slider_rows.setRange(3, 8)
            self.slider_cols.setToolTip("盤面の列数を変更します (3〜8)")
            self.slider_rows.setToolTip("盤面の行数を変更します (3〜8)")

    def _on_cols_slider_changed(self, val: int):
        """列スライダー変更ハンドラ"""
        self.lbl_cols_val.setText(str(val))
        self.sandbox_canvas.set_grid_dimensions(self.slider_rows.value(), val)

    def _on_rows_slider_changed(self, val: int):
        """行スライダー変更ハンドラ"""
        self.lbl_rows_val.setText(str(val))
        self.sandbox_canvas.set_grid_dimensions(val, self.slider_cols.value())

    def action_toggle_topmost(self, checked: bool):
        """最前面表示トグル (setWindowFlags による HWND 再生成時もホットキーを維持)"""
        was_hotkey_registered = self._registered_hotkey_hwnd is not None
        if was_hotkey_registered:
            self._unregister_global_hotkey()

        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

        if was_hotkey_registered:
            self._register_global_hotkey()

    def _apply_pen_color(self, color: QColor):
        """選択されたペン色をキャンバスと色インジケータボタンへ反映する。"""
        self.sandbox_canvas.set_color(color)
        self.btn_color_indicator.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #ffffff; border-radius: 4px;"
        )

    def action_choose_color(self):
        """カラーピッカーダイアログを表示する。"""
        col = QColorDialog.getColor(
            self.sandbox_canvas.current_color, self, "ペンの色を選択"
        )
        if col.isValid():
            self._apply_pen_color(col)

    def action_set_preset_color(self, hex_color: str):
        """プリセット色を設定する。"""
        self._apply_pen_color(QColor(hex_color))

    def action_save_sandbox_image(self):
        """現在のサンドボックスキャンバスの内容を元解像度画像ファイルとして保存する。"""
        if self.sandbox_canvas.pixmap.isNull():
            return

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "考察画像を保存",
            "witness_sandbox_solution.png",
            "PNG画像 (*.png);;JPEG画像 (*.jpg *.jpeg)",
        )
        if not filepath:
            return

        filepath = vision.normalize_save_image_path(filepath, selected_filter)
        export_img = self.sandbox_canvas.render_export_image()
        if export_img.isNull():
            QMessageBox.warning(self, "保存失敗", "保存用画像の生成に失敗しました。")
            return

        try:
            saved = export_img.save(filepath)
        except Exception as e:
            QMessageBox.warning(
                self, "保存失敗", f"画像の保存中にエラーが発生しました:\n{e}"
            )
            return

        if saved:
            QMessageBox.information(self, "保存完了", f"画像を保存しました:\n{filepath}")
        else:
            QMessageBox.warning(self, "保存失敗", "画像の保存に失敗しました。")


# ==============================================================================
# メインエントリーポイント
# ==============================================================================


def main():
    """アプリケーションを初期化して GUI イベントループを開始する。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE_QSS)

    window = MainWindow()
    window.center_on_screen()
    window.show()
    window.showNormal()
    window.raise_()
    window.activateWindow()
    app.processEvents()

    try:
        capture.activate_window_native(int(window.winId()))
    except Exception as e:
        print(f"[Warn] フォアグラウンド設定警告: {e}", flush=True)

    print("[Info] ウィンドウを表示しました。GUIイベントループを開始します。", flush=True)
    exit_code = app.exec()
    print(f"[Info] アプリケーションが正常終了しました (exit code: {exit_code})", flush=True)
    sys.exit(exit_code)
