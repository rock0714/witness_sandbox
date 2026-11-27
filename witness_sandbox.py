"""
The Witness サンドボックス支援ツール (Witness Sandbox Tool)

起動エントリポイントおよび主要クラス・関数の公開インターフェース。
実装は以下の責務別モジュールに分割されている:
- `app.vision`: 幾何計算・グリッド自動検知・透視投影変換・ローカル画像入出力検証
- `app.capture`: 前面ウィンドウ取得・画面キャプチャ・Win32 グローバルホットキー制御
- `app.canvas`: 共通キャンバス基底 (`ImageCanvasBase`)・四隅指定・サンドボックス描画キャンバス
- `app.window`: メインウィンドウ (`MainWindow`)・ホットキー中継・アプリケーション起動
"""

from app.canvas import (
    DrawMode,
    ImageCanvasBase,
    PerspectiveCanvas,
    SandboxCanvas,
    cv_to_qpixmap,
)
from app.capture import (
    HOTKEY_ID_F4,
    MOD_NONE,
    MOD_NOREPEAT,
    VK_F4,
    WM_HOTKEY,
    activate_window_native,
    capture_foreground_window_bgr,
    get_foreground_window_hwnd,
    get_window_rect,
    is_f4_hotkey_message,
    register_f4_hotkey,
    unregister_f4_hotkey,
)
from app.vision import (
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE_PX,
    create_sample_witness_image,
    detect_grid_dimensions,
    load_image_bgr,
    normalize_save_image_path,
    order_points,
    point_to_segment_distance,
    validate_quad_points,
    warp_perspective,
)
from app.window import DARK_STYLE_QSS, HotkeySignaler, MainWindow, main

__all__ = [
    "DARK_STYLE_QSS",
    "DrawMode",
    "HOTKEY_ID_F4",
    "HotkeySignaler",
    "ImageCanvasBase",
    "MAX_IMAGE_FILE_BYTES",
    "MAX_IMAGE_PIXELS",
    "MAX_IMAGE_SIDE_PX",
    "MOD_NONE",
    "MOD_NOREPEAT",
    "MainWindow",
    "PerspectiveCanvas",
    "SandboxCanvas",
    "VK_F4",
    "WM_HOTKEY",
    "activate_window_native",
    "capture_foreground_window_bgr",
    "create_sample_witness_image",
    "cv_to_qpixmap",
    "detect_grid_dimensions",
    "get_foreground_window_hwnd",
    "get_window_rect",
    "is_f4_hotkey_message",
    "load_image_bgr",
    "main",
    "normalize_save_image_path",
    "order_points",
    "point_to_segment_distance",
    "register_f4_hotkey",
    "unregister_f4_hotkey",
    "validate_quad_points",
    "warp_perspective",
]


if __name__ == "__main__":
    main()
