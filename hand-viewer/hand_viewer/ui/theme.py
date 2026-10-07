"""Encord look for the Qt app: the light theme of the Encord Design System (claude.ai/artifact/YWugXU89YR9tbRSWbCXcTi,
mirroring @encord/ui's tokens.ts), as a QSS stylesheet plus the few widgets QSS can't express (switch, card, legend).

House rules applied here: flat warm surfaces separated by hairlines (no shadows); lime only as the fill of the single
primary action, always with ink text; olive for checked/on states; Inter for UI text, DM Sans 500 for headings;
sentence case; 4px spacing scale; 8px radius on controls, 12px on cards. From encord.com: the ink logo header with
hairline borders and small uppercase eyebrow labels.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QAbstractButton, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# ---------------------------------------------------------------------------------------------- tokens (light)

BACKGROUND = "#f6f5f3"        # page canvas (encord-neutral-25)
CARD = "#ffffff"              # in-flow surfaces
MUTED = "#f1f1f1"             # inert fills
VIEWPORT = "#f7f9fa"          # encord-neutral-50, quiet section fill: behind the video and the 3D scene
GRID = "#cdd3d6"              # encord-neutral-400
FOREGROUND = "#1b1916"        # encord-ink
MUTED_TEXT = "#5a5753"        # encord-neutral-800
PLACEHOLDER = "#c1c1c0"
BORDER = "#edf0f2"            # grouping hairline
RING = "#dfe5e9"              # resting control boundary
RING_HOVER = "#cdd3d6"
FOCUS_RING = "#5a5753"        # never lime
PRIMARY = "#a6e40e"           # lime: fill of the one primary action
PRIMARY_DARK = "#719d09"      # olive: primary hover/pressed, checked and on states
SECONDARY_BG = "#f6f5f3"      # default button
SECONDARY_BORDER = "#eaeaea"
SAND = "#e8e3dd"              # warm hover for default buttons / nav chrome
SAND_SOFT = "rgba(232, 227, 221, 0.5)"   # surface-selected
SAND_ACTIVE = "#dcd6ce"
HOVER = "#f3f9e7"             # row / menu-item hover
MUTED_BORDER = "#c1c1c0"      # unchecked checkbox, switch off track
STATUS_DANGER = "#c41430"     # error wording that meets the text floor
ERROR_SUBTLE = "#fdecef"

# Hands are data, so they take the ordered categorical chart series (chart-1 olive, chart-2 blue), not brand lime.
HAND_COLORS = {"left": "#719d09", "right": "#2563eb"}

RADIUS_SM, RADIUS_MD, RADIUS_LG = 4, 8, 12
CONTROL_MD = 32
SPACE = {1: 4, 2: 8, 3: 12, 4: 16, 6: 24}

SANS, DISPLAY = "Inter", "DM Sans"   # replaced by the registered family names in load_fonts()


def rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = QColor(hex_color)
    return c.redF(), c.greenF(), c.blueF(), alpha


def load_fonts() -> None:
    """Register the bundled Inter and DM Sans variable fonts (OFL, from @fontsource-variable, as encord-fe uses)."""
    global SANS, DISPLAY
    for name, attr in (("inter-latin-wght-normal.woff2", "SANS"), ("dm-sans-latin-wght-normal.woff2", "DISPLAY")):
        fid = QFontDatabase.addApplicationFont(str(ASSETS / "fonts" / name))
        if fid >= 0 and (families := QFontDatabase.applicationFontFamilies(fid)):
            globals()[attr] = families[0]


def app_font() -> QFont:
    f = QFont(SANS)
    f.setPixelSize(14)   # body: 14/22, weight 400
    return f


def display_font(px: int = 16) -> QFont:
    """Heading style (title-5 by default): DM Sans 500, -0.02em tracking."""
    f = QFont(DISPLAY)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight.Medium)
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 98)
    return f


def apply(app) -> None:
    """Fonts, stylesheet and icon for the whole QApplication. Call before building windows."""
    load_fonts()
    app.setFont(app_font())
    app.setWindowIcon(app_icon())
    app.setStyleSheet(stylesheet())


def app_icon() -> QIcon:
    return QIcon(str(ASSETS / "encord-app-icon.png"))


def stylesheet() -> str:
    return f"""
    QWidget {{ color: {FOREGROUND}; font-family: "{SANS}"; font-size: 14px; }}
    QMainWindow, #canvas {{ background: {BACKGROUND}; }}
    #header, #footer {{ background: {CARD}; }}
    #header {{ border-bottom: 1px solid {BORDER}; }}
    #footer {{ border-top: 1px solid {BORDER}; }}
    #footer QLabel {{ font-size: 12px; color: {MUTED_TEXT}; }}
    #footer QLabel#error {{ color: {STATUS_DANGER}; }}
    #card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: {RADIUS_LG}px; }}
    #cardHeader {{ border-bottom: 1px solid {BORDER}; }}
    QLabel#caption {{ font-size: 12px; color: {MUTED_TEXT}; }}
    QLabel#eyebrow {{ font-size: 10px; font-weight: 500; color: {MUTED_TEXT}; letter-spacing: 0.8px; }}
    QFrame#divider {{ background: {BORDER}; }}

    QPushButton {{
        min-height: {CONTROL_MD - 2}px; max-height: {CONTROL_MD - 2}px; padding: 0 15px;
        background: {SECONDARY_BG}; border: 1px solid {SECONDARY_BORDER}; border-radius: {RADIUS_MD}px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {SAND_SOFT}; border-color: {MUTED_BORDER}; }}
    QPushButton:pressed {{ background: {SAND_ACTIVE}; }}
    QPushButton:focus {{ border: 2px solid {FOCUS_RING}; }}
    QPushButton#primary {{ background: {PRIMARY}; border: 1px solid {PRIMARY}; color: {FOREGROUND}; }}
    QPushButton#primary:hover, QPushButton#primary:pressed {{
        background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; color: #ffffff;
    }}
    QPushButton#primary:focus {{ border: 2px solid {FOCUS_RING}; }}
    QPushButton#icon {{ padding: 0; min-width: {CONTROL_MD - 2}px; max-width: {CONTROL_MD - 2}px; }}
    QPushButton#icon:checked {{ background: {SAND_ACTIVE}; border-color: {MUTED_BORDER}; }}

    QComboBox {{
        min-height: {CONTROL_MD - 2}px; max-height: {CONTROL_MD - 2}px; padding: 0 8px 0 11px;
        background: {CARD}; border: 1px solid {RING}; border-radius: {RADIUS_MD}px;
    }}
    QComboBox:hover {{ border-color: {MUTED_BORDER}; }}
    QComboBox:focus, QComboBox:on {{ border-color: {FOCUS_RING}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ image: url({(ASSETS / "chevron-down.svg").as_posix()}); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: {RADIUS_MD}px; padding: 4px; outline: 0;
        selection-background-color: {SAND_SOFT}; selection-color: {FOREGROUND};
    }}
    QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 0 8px; border-radius: {RADIUS_SM}px; }}
    QComboBox QAbstractItemView::item:hover {{ background: {HOVER}; }}
    QComboBox QAbstractItemView::item:selected {{ background: {SAND_SOFT}; }}

    QToolTip {{
        background: {FOREGROUND}; color: {BACKGROUND}; border: none; border-radius: {RADIUS_MD}px; padding: 4px 8px;
        font-size: 12px;
    }}
    QSplitter::handle {{ background: transparent; }}
    QMessageBox {{ background: {CARD}; }}
    """


# ---------------------------------------------------------------------------------------------- widgets


class Switch(QAbstractButton):
    """Design-system Switch: an on/off that applies immediately. 40x22 full-radius track, olive when on, muted
    border grey when off, 18px white thumb; label on the left. The whole row is the target."""

    TRACK_W, TRACK_H, THUMB = 40, 22, 18

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def sizeHint(self) -> QSize:
        w = self.fontMetrics().horizontalAdvance(self.text())
        return QSize(w + SPACE[2] + self.TRACK_W + 4, CONTROL_MD)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(FOREGROUND))
        text_w = self.fontMetrics().horizontalAdvance(self.text())
        p.drawText(QRectF(0, 0, text_w, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        x = text_w + SPACE[2]
        track = QRectF(x, (self.height() - self.TRACK_H) / 2, self.TRACK_W, self.TRACK_H)
        if self.hasFocus():
            p.setPen(QPen(QColor(FOCUS_RING), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(track.adjusted(-3, -3, 3, 3), self.TRACK_H / 2 + 3, self.TRACK_H / 2 + 3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PRIMARY_DARK if self.isChecked() else MUTED_BORDER))
        p.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        m = (self.TRACK_H - self.THUMB) / 2
        cx = track.right() - m - self.THUMB / 2 if self.isChecked() else track.left() + m + self.THUMB / 2
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(cx, track.center().y()), self.THUMB / 2, self.THUMB / 2)
        p.end()


class Card(QFrame):
    """Flat bordered surface with a header (title-5 heading, caption meta, actions on the right) and a body."""

    def __init__(self, title: str, body: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        header = QWidget()
        header.setObjectName("cardHeader")
        self.header_layout = QHBoxLayout(header)
        self.header_layout.setContentsMargins(SPACE[4], SPACE[2], SPACE[3], SPACE[2])
        self.header_layout.setSpacing(SPACE[2])
        self.header = header
        heading = QLabel(title)
        heading.setFont(display_font(16))
        heading.setFixedHeight(CONTROL_MD)
        self.meta = QLabel()
        self.meta.setObjectName("caption")
        self.header_layout.addWidget(heading)
        self.header_layout.addWidget(self.meta)
        self.header_layout.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)   # inside the hairline
        lay.setSpacing(0)
        lay.addWidget(header)
        # The body sits inset on the card: a GL viewport can't be clipped to the card's rounded corners.
        inset = QWidget()
        il = QVBoxLayout(inset)
        il.setContentsMargins(SPACE[3], SPACE[3], SPACE[3], SPACE[3])
        il.addWidget(body)
        lay.addWidget(inset, 1)

    def add_action(self, w: QWidget) -> None:
        self.header_layout.addWidget(w)


class LegendDot(QWidget):
    """A data colour lives in a dot beside its label, never in the text."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {color}; border-radius: 5px;")
        text = QLabel(label)
        text.setObjectName("caption")
        lay.addWidget(dot)
        lay.addWidget(text)


def icon(name: str) -> QIcon:
    """An SVG from assets/, e.g. icon("settings")."""
    return QIcon(str(ASSETS / f"{name}.svg"))


def logo(height: int = 22) -> QSvgWidget:
    """Full lockup, black ink for a light ground (roughly 5:1; never below 90px wide)."""
    w = QSvgWidget(str(ASSETS / "encord-logo-light.svg"))
    w.setFixedSize(round(height * 243 / 48.32), height)
    return w


def eyebrow(text: str) -> QLabel:
    """Small uppercase label for dense chrome (10px, tracked out, three words or fewer), as on encord.com."""
    label = QLabel(text.upper())
    label.setObjectName("eyebrow")
    return label


def divider(vertical: bool = True) -> QFrame:
    d = QFrame()
    d.setObjectName("divider")
    d.setFixedWidth(1) if vertical else d.setFixedHeight(1)
    if vertical:
        d.setFixedHeight(20)
    return d
