"""
OS 連携・画面キャプチャ・Win32 グローバルホットキー制御モジュール。
Windows API (user32 / dwmapi) 呼び出しを本モジュールに隔離し、非 Windows 環境でも安全にフォールバックする。
"""

import ctypes

try:
    from ctypes import wintypes
except ImportError:
    wintypes = None  # type: ignore[assignment]

import cv2
import numpy as np
from PIL import ImageGrab

# Windows API 定数 (グローバルホットキー RegisterHotKey / WM_HOTKEY 用)
MOD_NONE = 0x0000
MOD_NOREPEAT = 0x4000
VK_F4 = 0x73
WM_HOTKEY = 0x0312
HOTKEY_ID_F4 = 1001


def get_foreground_window_hwnd() -> int:
    """現在の前面ウィンドウの HWND を返す。非 Windows 環境や取得失敗時は 0 を返す。"""
    try:
        if hasattr(ctypes, "windll"):
            get_fg = ctypes.windll.user32.GetForegroundWindow
            if wintypes is not None:
                get_fg.restype = wintypes.HWND
            fg = get_fg()
            return int(fg) if fg else 0
    except Exception:
        pass
    return 0


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """
    ウィンドウの画面上の矩形 (left, top, right, bottom) を返す。
    影を含まない DWM の拡張フレーム境界を優先し、失敗時は GetWindowRect を使う。
    最小化・サイズ不正・取得失敗時は None。
    """
    try:
        if not hwnd or not hasattr(ctypes, "windll") or wintypes is None:
            return None
        user32 = ctypes.windll.user32
        native_hwnd = wintypes.HWND(hwnd)
        if not user32.IsWindow(native_hwnd) or user32.IsIconic(native_hwnd):
            return None
        rect = wintypes.RECT()
        # DWMWA_EXTENDED_FRAME_BOUNDS = 9 (ウィンドウ影を除いた見た目通りの枠)
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            native_hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if res != 0 and not user32.GetWindowRect(native_hwnd, ctypes.byref(rect)):
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


def capture_foreground_window_bgr(
    target_hwnd: int | None = None, exclude_hwnd: int | None = None
) -> np.ndarray:
    """
    指定ウィンドウ (省略時は現在の前面ウィンドウ) のみを撮影して BGR 画像で返す。
    exclude_hwnd (自アプリ) が対象になった場合や矩形が取得できない場合は全画面にフォールバックする。
    """
    hwnd = target_hwnd
    if hwnd is None:
        fg = get_foreground_window_hwnd()
        hwnd = fg if fg else None
    if hwnd and exclude_hwnd and int(hwnd) == int(exclude_hwnd):
        hwnd = None

    rect = get_window_rect(int(hwnd)) if hwnd else None
    if rect is not None:
        shot = ImageGrab.grab(bbox=rect, all_screens=True)
    else:
        shot = ImageGrab.grab(all_screens=True)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def register_f4_hotkey(hwnd: int) -> bool:
    """指定 HWND に対して Win32 RegisterHotKey で F4 グローバルホットキーを登録する。"""
    if not hasattr(ctypes, "windll") or wintypes is None:
        print(
            "[Info] 非Windows環境のためグローバルホットキー登録をスキップしました (アプリ内F4キーやボタンで操作可能)。",
            flush=True,
        )
        return False
    if not hwnd:
        return False

    try:
        user32 = ctypes.windll.user32
        # 押しっぱなし連打を防ぐ MOD_NOREPEAT 付きを優先し、非対応環境では MOD_NONE にフォールバック
        ok = bool(
            user32.RegisterHotKey(
                wintypes.HWND(hwnd), HOTKEY_ID_F4, MOD_NONE | MOD_NOREPEAT, VK_F4
            )
        )
        if not ok:
            ok = bool(
                user32.RegisterHotKey(
                    wintypes.HWND(hwnd), HOTKEY_ID_F4, MOD_NONE, VK_F4
                )
            )
        if ok:
            print("[Info] グローバルホットキー (F4) を登録しました。", flush=True)
            return True
        print(
            "[Info] グローバルホットキー (F4) は他アプリで使用中のため登録をスキップしました (アプリ内F4キーやボタンで操作可能)。",
            flush=True,
        )
        return False
    except Exception as e:
        print(
            f"[Info] グローバルホットキーの登録をスキップしました (アプリ内F4キーやボタンで操作可能): {e}",
            flush=True,
        )
        return False


def unregister_f4_hotkey(hwnd: int) -> None:
    """指定 HWND に登録された Win32 F4 グローバルホットキーを解除する。"""
    if not hwnd or not hasattr(ctypes, "windll") or wintypes is None:
        return
    try:
        ctypes.windll.user32.UnregisterHotKey(wintypes.HWND(hwnd), HOTKEY_ID_F4)
    except Exception:
        pass


def is_f4_hotkey_message(event_type_bytes: bytes, message_addr: int) -> bool:
    """Qt nativeEvent のメッセージが F4 グローバルホットキー (WM_HOTKEY) かを判定する。"""
    if wintypes is None or not message_addr:
        return False
    if event_type_bytes not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
        return False
    try:
        msg = wintypes.MSG.from_address(message_addr)
        return bool(msg.message == WM_HOTKEY and int(msg.wParam) == HOTKEY_ID_F4)
    except Exception:
        return False


def activate_window_native(hwnd: int) -> None:
    """Windows API で指定ウィンドウを通常表示・最前面フォアグラウンド化する。"""
    if not hwnd or not hasattr(ctypes, "windll") or wintypes is None:
        return
    try:
        native_hwnd = wintypes.HWND(hwnd)
        ctypes.windll.user32.ShowWindow(native_hwnd, 1)  # 1 = SW_SHOWNORMAL
        ctypes.windll.user32.SetForegroundWindow(native_hwnd)
    except Exception as e:
        print(f"[Warn] フォアグラウンド設定警告: {e}", flush=True)
