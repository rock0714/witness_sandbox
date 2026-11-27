"""
幾何計算・画像解析・透視投影変換・ローカル画像入出力検証モジュール。
GUI フレームワーク (PyQt6) や OS 固有 API に依存しない純粋な画像処理層を提供する。
"""

import io
import math
import os
import warnings

import cv2
import numpy as np
from PIL import Image

# ローカル画像読込の上限設定 (OOM / Image Bomb 対策)
MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024  # ファイルサイズ上限: 50 MB
MAX_IMAGE_SIDE_PX = 10_000               # 1辺の最大ピクセル数: 10,000 px
MAX_IMAGE_PIXELS = 50_000_000            # 最大総画素数: 5,000万画素 (8K超まで許容)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4つの頂点座標を重心からの偏角に基づき時計回りに並べ、
    [左上, 右上, 右下, 左下] の順に幾何学的に整列する。
    pts: shape (4, 2)
    """
    arr = np.asarray(pts, dtype="float32")
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError(f"4点の有効な (x, y) 座標が必要です: shape={arr.shape}")

    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    # 画面座標系 (x右正, y下正) では偏角の昇順が時計回り周順になる
    cw_pts = arr[np.argsort(angles)]

    # x + y が最小の頂点（同値時は y が小さい上側頂点）を左上 [0] に合わせる
    sums = cw_pts[:, 0] + cw_pts[:, 1]
    min_sum = np.min(sums)
    candidates = np.where(np.isclose(sums, min_sum, atol=1e-4))[0]
    if len(candidates) > 1:
        tl_idx = int(candidates[np.argmin(cw_pts[candidates, 1])])
    else:
        tl_idx = int(candidates[0])

    return np.roll(cw_pts, -tl_idx, axis=0).astype("float32")


def validate_quad_points(
    pts: np.ndarray, min_edge_len: float = 5.0, min_area: float = 100.0
) -> np.ndarray:
    """
    4点が重複・共線・非凸・極小面積でない有効な凸四角形かを検証し、
    整列済みの [左上, 右上, 右下, 左下] 座標配列を返す。不正時は ValueError を送出する。
    """
    arr = np.asarray(pts, dtype="float32")
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError("4点の有効な座標が指定されていません。")

    # 任意の2点間の距離チェック（重複点・極端な近接点の検出）
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])) < min_edge_len:
                raise ValueError("指定された4点に重複または近すぎる点が含まれています。")

    rect = order_points(arr)

    # 時計回り順での連続辺ベクトルの外積による凸性・共線チェック
    edges = [rect[(i + 1) % 4] - rect[i] for i in range(4)]
    for i in range(4):
        e1 = edges[i]
        e2 = edges[(i + 1) % 4]
        cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        if cross <= 1e-3:
            raise ValueError("指定された4点が一直線上にあるか、凸四角形を形成していません。")

    # Shoelace 公式による面積チェック
    x = rect[:, 0]
    y = rect[:, 1]
    area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < min_area:
        raise ValueError("指定された4点の領域が小さすぎます。")

    return rect


def load_image_bgr(
    filepath: str,
    max_file_bytes: int = MAX_IMAGE_FILE_BYTES,
    max_side_px: int = MAX_IMAGE_SIDE_PX,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> np.ndarray:
    """
    日本語パス・ファイルサイズ上限・展開前解像度上限（Image Bomb 対策）を検証して
    ローカル画像を BGR 配列として安全に読み込む。
    ファイルアクセス失敗時は OSError、検証・デコード失敗時は ValueError を送出する。
    """
    file_size = os.path.getsize(filepath)
    if file_size <= 0:
        raise ValueError("画像ファイルが空です。")
    if file_size > max_file_bytes:
        limit_mb = max_file_bytes / (1024 * 1024)
        raise ValueError(f"ファイルサイズが上限 ({limit_mb:.0f} MB) を超えています。")

    with open(filepath, "rb") as stream:
        raw_bytes = stream.read()

    # ピクセル展開前にヘッダから画像寸法を検査して Image Bomb を防止
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                w, h = pil_img.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ValueError("画像解像度が安全上限を超えています。") from e
    except Exception as e:
        raise ValueError("有効な画像データとして認識できませんでした。") from e

    if w <= 0 or h <= 0:
        raise ValueError("画像の寸法が不正です。")
    if w > max_side_px or h > max_side_px or (w * h) > max_pixels:
        raise ValueError(
            f"画像解像度 ({w}×{h}) が上限 (1辺最大 {max_side_px}px / 最大 {max_pixels:,} 画素) を超えています。"
        )

    numpy_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("画像のデコードに失敗しました。")
    return img_bgr


def normalize_save_image_path(filepath: str, selected_filter: str = "") -> str:
    """保存ファイルパスに有効な画像拡張子がない場合、選択フィルタに応じて拡張子を補完する。"""
    _, ext = os.path.splitext(filepath)
    if ext.lower() in (".png", ".jpg", ".jpeg"):
        return filepath
    if "jpg" in selected_filter.lower() or "jpeg" in selected_filter.lower():
        return filepath + ".jpg"
    return filepath + ".png"


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
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


def detect_grid_dimensions(
    image_bgr: np.ndarray, pts: np.ndarray | None = None
) -> tuple[int, int, float]:
    """
    画像内のパズルグリッド線（列数 cols、行数 rows、マージン割合 margin）を画像解析で自動検知する。
    pts が指定された場合は、まず正方形で仮の透視投影変換を行ってからグリッドを解析する。
    戻り値: (cols, rows, margin_percent)
    """
    if pts is not None and len(pts) == 4:
        try:
            rect = validate_quad_points(pts)
        except ValueError:
            return 4, 4, 0.05
        sample_size = 600
        dst = np.array(
            [
                [0, 0],
                [sample_size - 1, 0],
                [sample_size - 1, sample_size - 1],
                [0, sample_size - 1],
            ],
            dtype="float32",
        )
        m = cv2.getPerspectiveTransform(rect, dst)
        analysis_img = cv2.warpPerspective(
            image_bgr, m, (sample_size, sample_size), flags=cv2.INTER_LINEAR
        )
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
    h_proj = np.mean(sobel_y[:, int(w * 0.15) : int(w * 0.85)], axis=1)
    v_proj = np.mean(sobel_x[int(h * 0.15) : int(h * 0.85), :], axis=0)

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
                    val = np.max(proj[max(0, pos - 3) : min(length, pos + 4)])
                    vals.append(val)

                vals_arr = np.array(vals)
                min_v = np.min(vals_arr)
                max_v = np.max(vals_arr)

                # 全てのグリッド線が一定の強度で存在しているかを判定
                if min_v > max_v * 0.35:
                    score = float(np.sum(vals_arr))
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_m = m_ratio

        return best_n, best_m

    detected_cols, m_cols = evaluate_projection(v_proj, w)
    detected_rows, m_rows = evaluate_projection(h_proj, h)
    avg_margin = round((m_cols + m_rows) / 2.0, 2)

    return detected_cols, detected_rows, avg_margin


def warp_perspective(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    aspect_mode: str = "square",
    cols: int = 4,
    rows: int = 4,
) -> np.ndarray:
    """
    指定された4隅をもとに透視投影変換を行い、真正面から見た画像に補正する。
    aspect_mode:
        - "square": 正方形 (1:1) の盤面として補正
        - "grid": マス目が正方形 (1:1) になるよう、列数:行数の比率で補正
        - "original": 元の斜め画面上の推定ユークリッド距離比で補正
    """
    rect = validate_quad_points(pts)
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

    dst = np.array(
        [
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ],
        dtype="float32",
    )

    # 変換行列の計算と実行
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image_bgr, m, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4
    )
    return warped


def create_sample_witness_image() -> np.ndarray:
    """
    The Witness風の斜めパズル画像を自動生成する (動作テスト・デモ用)。
    """
    # 真正面のベース画像 (600x600)
    base_size = 600
    panel = np.full((base_size, base_size, 3), (35, 30, 25), dtype=np.uint8)

    # 枠線と外枠
    cv2.rectangle(
        panel, (40, 40), (base_size - 40, base_size - 40), (200, 180, 150), 8
    )

    # 4x4 グリッド線
    rows, cols = 4, 4
    gx0, gy0 = 60, 60
    gw = (base_size - 120) / cols
    gh = (base_size - 120) / rows

    for r in range(rows + 1):
        y = int(gy0 + r * gh)
        cv2.line(
            panel, (int(gx0), y), (int(gx0 + cols * gw), y), (140, 130, 110), 4
        )
    for c in range(cols + 1):
        x = int(gx0 + c * gw)
        cv2.line(
            panel, (x, int(gy0)), (x, int(gy0 + rows * gh)), (140, 130, 110), 4
        )

    # スタート地点 (左下の大きな丸)
    start_x = int(gx0)
    start_y = int(gy0 + rows * gh)
    cv2.circle(panel, (start_x, start_y), 18, (230, 220, 200), -1)

    # ゴール地点 (右上の飛び出し線)
    end_x = int(gx0 + cols * gw)
    end_y = int(gy0)
    cv2.line(panel, (end_x, end_y), (end_x + 25, end_y - 25), (200, 180, 150), 6)

    # マス目の記号 (The Witnessでおなじみの白四角・黒四角)
    def draw_symbol(cell_r: int, cell_c: int, color_bgr: tuple[int, int, int]):
        cx = int(gx0 + cell_c * gw + gw / 2)
        cy = int(gy0 + cell_r * gh + gh / 2)
        s = 14
        cv2.rectangle(panel, (cx - s, cy - s), (cx + s, cy + s), color_bgr, -1)
        cv2.rectangle(panel, (cx - s, cy - s), (cx + s, cy + s), (30, 30, 30), 2)

    draw_symbol(0, 0, (240, 240, 240))  # 白
    draw_symbol(1, 1, (240, 240, 240))  # 白
    draw_symbol(2, 2, (30, 30, 30))     # 黒
    draw_symbol(3, 3, (30, 30, 30))     # 黒
    draw_symbol(1, 3, (0, 165, 255))    # オレンジ色の星

    # 背景のある大きなキャンバスに「斜めのパース」をかけて貼り付ける
    bg_w, bg_h = 1000, 700
    scene = np.full((bg_h, bg_w, 3), (60, 50, 45), dtype=np.uint8)

    src_pts = np.array(
        [
            [0, 0],
            [base_size, 0],
            [base_size, base_size],
            [0, base_size],
        ],
        dtype="float32",
    )

    skew_pts = np.array(
        [
            [220, 110],  # 左上
            [780, 70],   # 右上 (奥に向かってすぼまる)
            [890, 580],  # 右下
            [150, 610],  # 左下
        ],
        dtype="float32",
    )

    m_skew = cv2.getPerspectiveTransform(src_pts, skew_pts)
    warped_panel = cv2.warpPerspective(panel, m_skew, (bg_w, bg_h))

    gray = cv2.cvtColor(warped_panel, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    scene[mask > 0] = warped_panel[mask > 0]

    cv2.putText(
        scene,
        "Sample Witness Puzzle Screen (Skewed View)",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        scene,
        "Click 4 corners of the puzzle panel to correct perspective!",
        (30, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 200, 220),
        2,
        cv2.LINE_AA,
    )

    return scene
