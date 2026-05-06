"""
Interactive soft PneuNet / TDCR joint simulator for the NCKH_EE_2026 project.

Features:
- Tkinter-only GUI, no external dependencies.
- Manual geometry, servo, and physics parameters.
- Second-order tendon/backbone dynamics with damping, slack, and friction.
- Click and drag individual rigid plates away from the actuator centerline.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, List, Sequence, Tuple


# -----------------------------
# Model data
# -----------------------------


@dataclass
class ActuatorParams:
    length_mm: float = 100.0
    plate_count: int = 10
    tendon_offset_mm: float = 7.0
    plate_diameter_mm: float = 20.0
    plate_thickness_mm: float = 3.0
    spool_radius_mm: float = 10.0
    neutral_servo_deg: float = 90.0
    max_servo_deg: float = 180.0
    pw_min_us: float = 500.0
    pw_max_us: float = 2500.0

    @property
    def intervals(self) -> int:
        return max(1, self.plate_count - 1)

    @property
    def plate_spacing_mm(self) -> float:
        return self.length_mm / self.intervals


@dataclass
class PhysicsParams:
    tendon_gain: float = 0.72          # Equivalent tendon torque gain
    backbone_stiffness: float = 0.18   # Restoring torque per rad
    damping: float = 0.085             # Viscous damping
    inertia: float = 0.030             # Effective rotational inertia
    coulomb_friction: float = 0.010    # Static/kinetic dry friction threshold
    slack_mm: float = 0.35             # Cable slack before tendon force builds
    max_speed_deg_s: float = 620.0


@dataclass
class Kinematics:
    theta_deg: float
    theta_rad: float
    curvature_per_mm: float
    bend_radius_mm: float | None
    delta_l_mm: float
    tendon_1_mm: float
    tendon_2_mm: float
    servo_delta_deg: float
    servo_1_deg: float
    servo_2_deg: float
    pwm_1_us: float
    pwm_2_us: float
    per_interval_deg: float


@dataclass
class DynamicsSnapshot:
    target_deg: float
    actual_deg: float
    velocity_deg_s: float
    tendon_error_mm: float
    tendon_torque: float
    backbone_torque: float
    damping_torque: float
    friction_torque: float
    net_torque: float
    acceleration_deg_s2: float


class TDCRModel:
    def __init__(self, params: ActuatorParams):
        self.p = params

    def clamp_bend_by_servo_limit(self, theta_deg: float) -> float:
        max_delta = min(self.p.neutral_servo_deg, self.p.max_servo_deg - self.p.neutral_servo_deg)
        if self.p.tendon_offset_mm <= 0 or self.p.spool_radius_mm <= 0:
            return 0.0
        max_theta_rad = math.radians(max_delta) * self.p.spool_radius_mm / self.p.tendon_offset_mm
        max_theta_deg = math.degrees(max_theta_rad)
        return max(-max_theta_deg, min(max_theta_deg, theta_deg))

    def solve(self, theta_deg: float) -> Kinematics:
        theta_deg = self.clamp_bend_by_servo_limit(theta_deg)
        theta = math.radians(theta_deg)
        length = max(1e-6, self.p.length_mm)
        radius = max(1e-6, self.p.spool_radius_mm)
        offset = max(0.0, self.p.tendon_offset_mm)

        curvature = theta / length
        bend_radius = None if abs(theta) < 1e-9 else length / theta
        delta_l = 2.0 * offset * theta
        tendon_1 = length - offset * theta
        tendon_2 = length + offset * theta

        servo_delta_rad = abs(delta_l) / (2.0 * radius)
        servo_delta_deg = math.degrees(servo_delta_rad)
        sign = 1 if theta >= 0 else -1
        servo_1 = self.p.neutral_servo_deg - sign * servo_delta_deg
        servo_2 = self.p.neutral_servo_deg + sign * servo_delta_deg
        servo_1 = max(0.0, min(self.p.max_servo_deg, servo_1))
        servo_2 = max(0.0, min(self.p.max_servo_deg, servo_2))

        span = self.p.pw_max_us - self.p.pw_min_us
        pwm_1 = self.p.pw_min_us + servo_1 / self.p.max_servo_deg * span
        pwm_2 = self.p.pw_min_us + servo_2 / self.p.max_servo_deg * span

        return Kinematics(
            theta_deg=theta_deg,
            theta_rad=theta,
            curvature_per_mm=curvature,
            bend_radius_mm=bend_radius,
            delta_l_mm=delta_l,
            tendon_1_mm=tendon_1,
            tendon_2_mm=tendon_2,
            servo_delta_deg=servo_delta_deg,
            servo_1_deg=servo_1,
            servo_2_deg=servo_2,
            pwm_1_us=pwm_1,
            pwm_2_us=pwm_2,
            per_interval_deg=theta_deg / self.p.intervals,
        )

    def centerline_point(self, u_mm: float, theta_deg: float) -> Tuple[float, float]:
        length = max(1e-6, self.p.length_mm)
        theta = math.radians(theta_deg)
        k = theta / length
        if abs(k) < 1e-9:
            return 0.0, u_mm
        return (1.0 / k) * (1.0 - math.cos(k * u_mm)), (1.0 / k) * math.sin(k * u_mm)

    def tangent_normal(self, u_mm: float, theta_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        length = max(1e-6, self.p.length_mm)
        phi = math.radians(theta_deg) * (u_mm / length)
        tangent = (math.sin(phi), math.cos(phi))
        normal = (math.cos(phi), -math.sin(phi))
        return tangent, normal

    def offset_point(self, u_mm: float, theta_deg: float, offset_mm: float) -> Tuple[float, float]:
        x, z = self.centerline_point(u_mm, theta_deg)
        _, normal = self.tangent_normal(u_mm, theta_deg)
        return x + offset_mm * normal[0], z + offset_mm * normal[1]


# -----------------------------
# Tkinter application
# -----------------------------


class TDCRSimulatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Soft PneuNet / TDCR joint simulator")
        self.root.geometry("1360x820")
        self.root.minsize(1120, 700)

        self.params = ActuatorParams()
        self.physics = PhysicsParams()
        self.model = TDCRModel(self.params)

        self.target_bend = tk.DoubleVar(value=0.0)
        self.max_bend_var = tk.DoubleVar(value=120.0)
        self.auto_demo = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="READY - drag a plate to change spacing")

        self.actual_bend = 0.0
        self.angular_velocity = 0.0
        self.last_snapshot = DynamicsSnapshot(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.plate_positions: List[float] = self._uniform_plate_positions(self.params.plate_count)
        self._plate_hitboxes: List[Tuple[int, float, float, float]] = []
        self._tip_hitbox: Tuple[float, float, float] | None = None
        self._drag_mode: str | None = None
        self._drag_plate: int | None = None
        self._view = (0.0, 0.0, 1.0)
        self._last_time = time.perf_counter()
        self._demo_start = self._last_time

        self._style_ui()
        self._build_ui()
        self._tick()

    # ---------- UI ----------

    def _style_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#111827")
        style.configure(".", font=("Segoe UI", 9))
        style.configure("Main.TFrame", background="#111827")
        style.configure("Side.TFrame", background="#172033")
        style.configure("CanvasWrap.TFrame", background="#0b1120")
        style.configure("Panel.TLabelframe", background="#172033", foreground="#e5e7eb", bordercolor="#334155")
        style.configure("Panel.TLabelframe.Label", background="#172033", foreground="#f8fafc", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#172033", foreground="#dbeafe")
        style.configure("Muted.TLabel", background="#172033", foreground="#94a3b8")
        style.configure("Readout.TLabel", background="#172033", foreground="#fbbf24", font=("Consolas", 10, "bold"))
        style.configure("TCheckbutton", background="#172033", foreground="#e5e7eb")
        style.map("TCheckbutton", background=[("active", "#172033")], foreground=[("active", "#ffffff")])
        style.configure("TButton", padding=(8, 5), background="#263449", foreground="#f8fafc", borderwidth=1)
        style.map("TButton", background=[("active", "#334155")], foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background="#0f766e", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#115e59")])

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, style="Main.TFrame", padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        left_wrap = ttk.Frame(main, style="Side.TFrame", width=374)
        left_wrap.pack(side=tk.LEFT, fill=tk.Y)
        left_wrap.pack_propagate(False)

        right = ttk.Frame(main, style="CanvasWrap.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.controls_canvas = tk.Canvas(left_wrap, bg="#172033", highlightthickness=0, width=374)
        scrollbar = ttk.Scrollbar(left_wrap, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_frame = ttk.Frame(self.controls_canvas, style="Side.TFrame", padding=10)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_frame.bind("<Configure>", self._update_scroll_region)
        self.controls_canvas.bind("<Configure>", self._resize_controls_window)

        header = ttk.Label(
            self.controls_frame,
            text="Soft actuator lab",
            font=("Segoe UI Semibold", 17),
            foreground="#ffffff",
            background="#172033",
        )
        header.pack(anchor="w")
        ttk.Label(
            self.controls_frame,
            text="Servo tendons, rigid plates, flexible backbone, and adjustable plate spacing.",
            style="Muted.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(2, 10))

        self._build_command_panel()
        self._build_geometry_panel()
        self._build_physics_panel()
        self._build_output_panel()

        self.canvas = tk.Canvas(right, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        bottom = ttk.Frame(right, style="CanvasWrap.TFrame", padding=(6, 7))
        bottom.pack(fill=tk.X)
        ttk.Label(
            bottom,
            textvariable=self.status_text,
            background="#0b1120",
            foreground="#dbeafe",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            bottom,
            text="G: flex 90   R: neutral   E: extend -90   Space: step   Drag plate: spacing   Drag empty: angle",
            background="#0b1120",
            foreground="#94a3b8",
        ).pack(side=tk.RIGHT)

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_click)
        self.root.bind("g", lambda _e: self.set_command(90, "keyboard G -> flexion"))
        self.root.bind("G", lambda _e: self.set_command(90, "keyboard G -> flexion"))
        self.root.bind("r", lambda _e: self.set_command(0, "keyboard R -> neutral"))
        self.root.bind("R", lambda _e: self.set_command(0, "keyboard R -> neutral"))
        self.root.bind("e", lambda _e: self.set_command(-90, "keyboard E -> extension"))
        self.root.bind("E", lambda _e: self.set_command(-90, "keyboard E -> extension"))
        self.root.bind("<space>", lambda _e: self._space_step())

    def _build_command_panel(self) -> None:
        box = ttk.LabelFrame(self.controls_frame, text="Điều khiển", style="Panel.TLabelframe")
        box.pack(fill=tk.X, pady=7)

        row = ttk.Frame(box, style="Side.TFrame")
        row.pack(fill=tk.X, padx=10, pady=(10, 3))
        ttk.Label(row, text="Góc uốn mục tiêu").pack(side=tk.LEFT)
        self.bend_readout = ttk.Label(row, text="0.0 deg", style="Readout.TLabel")
        self.bend_readout.pack(side=tk.RIGHT)

        self.bend_scale = ttk.Scale(
            box,
            from_=-self.max_bend_var.get(),
            to=self.max_bend_var.get(),
            orient=tk.HORIZONTAL,
            variable=self.target_bend,
            command=self._on_slider,
        )
        self.bend_scale.pack(fill=tk.X, padx=10, pady=(0, 8))
        self._number_control(box, "Góc kéo tối đa (+/- deg)", self.max_bend_var, 30, 180, "{:.0f}", self._apply_angle_limit)

        quick = ttk.Frame(box, style="Side.TFrame")
        quick.pack(fill=tk.X, padx=8, pady=(0, 8))
        for label, value in (("-90", -90), ("0", 0), ("30", 30), ("60", 60), ("90", 90)):
            ttk.Button(quick, text=label, command=lambda v=value: self.set_command(v, f"target {v} deg")).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=2
            )

        row2 = ttk.Frame(box, style="Side.TFrame")
        row2.pack(fill=tk.X, padx=8, pady=(0, 10))
        ttk.Button(row2, text="Hold", command=self.hold_current).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row2, text="Reset dynamics", command=self.reset_dynamics).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row2, text="Reset spacing", style="Accent.TButton", command=self.reset_plate_spacing).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Checkbutton(box, text="Auto demo", variable=self.auto_demo).pack(anchor="w", padx=10, pady=(0, 10))

    def _build_geometry_panel(self) -> None:
        box = ttk.LabelFrame(self.controls_frame, text="Thông số hình học / servo", style="Panel.TLabelframe")
        box.pack(fill=tk.X, pady=7)

        self.length_var = tk.DoubleVar(value=self.params.length_mm)
        self.plate_count_var = tk.DoubleVar(value=self.params.plate_count)
        self.offset_var = tk.DoubleVar(value=self.params.tendon_offset_mm)
        self.diameter_var = tk.DoubleVar(value=self.params.plate_diameter_mm)
        self.thickness_var = tk.DoubleVar(value=self.params.plate_thickness_mm)
        self.spool_var = tk.DoubleVar(value=self.params.spool_radius_mm)

        self._number_control(box, "Chiều dài L (mm)", self.length_var, 40, 180, "{:.0f}", self._apply_params)
        self._number_control(box, "Số plate N", self.plate_count_var, 3, 24, "{:.0f}", self._apply_params, integer=True)
        self._number_control(box, "Offset tendon r_c (mm)", self.offset_var, 2, 14, "{:.1f}", self._apply_params)
        self._number_control(box, "Đường kính plate D (mm)", self.diameter_var, 10, 42, "{:.0f}", self._apply_params)
        self._number_control(box, "Độ dày plate (mm)", self.thickness_var, 1, 9, "{:.1f}", self._apply_params)
        self._number_control(box, "Bán kính spool R_s (mm)", self.spool_var, 4, 24, "{:.1f}", self._apply_params)

        row = ttk.Frame(box, style="Side.TFrame")
        row.pack(fill=tk.X, padx=8, pady=(4, 10))
        ttk.Button(row, text="+ Plate", command=self.add_plate).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row, text="- Tip plate", command=self.remove_tip_plate).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row, text="Reset N=10", command=self.reset_plate_count).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(box, text="Reset khoảng cách đều", command=self.reset_plate_spacing).pack(fill=tk.X, padx=10, pady=(0, 10))

    def _build_physics_panel(self) -> None:
        box = ttk.LabelFrame(self.controls_frame, text="Thông số vật lý", style="Panel.TLabelframe")
        box.pack(fill=tk.X, pady=7)

        self.tendon_gain_var = tk.DoubleVar(value=self.physics.tendon_gain)
        self.stiffness_var = tk.DoubleVar(value=self.physics.backbone_stiffness)
        self.damping_var = tk.DoubleVar(value=self.physics.damping)
        self.inertia_var = tk.DoubleVar(value=self.physics.inertia)
        self.friction_var = tk.DoubleVar(value=self.physics.coulomb_friction)
        self.slack_var = tk.DoubleVar(value=self.physics.slack_mm)
        self.speed_var = tk.DoubleVar(value=self.physics.max_speed_deg_s)

        self._number_control(box, "Lực tendon", self.tendon_gain_var, 0.05, 1.8, "{:.2f}", self._apply_params)
        self._number_control(box, "Độ cứng backbone", self.stiffness_var, 0.00, 0.80, "{:.2f}", self._apply_params)
        self._number_control(box, "Giảm chấn", self.damping_var, 0.00, 0.30, "{:.3f}", self._apply_params)
        self._number_control(box, "Quán tính", self.inertia_var, 0.005, 0.150, "{:.3f}", self._apply_params)
        self._number_control(box, "Ma sát khô", self.friction_var, 0.00, 0.080, "{:.3f}", self._apply_params)
        self._number_control(box, "Slack dây (mm)", self.slack_var, 0.00, 3.00, "{:.2f}", self._apply_params)
        self._number_control(box, "Tốc độ tối đa (deg/s)", self.speed_var, 120, 1200, "{:.0f}", self._apply_params)

    def _build_output_panel(self) -> None:
        box = ttk.LabelFrame(self.controls_frame, text="Kết quả tức thời", style="Panel.TLabelframe")
        box.pack(fill=tk.BOTH, expand=True, pady=7)
        self.calc_label = ttk.Label(box, text="", justify=tk.LEFT, wraplength=330, font=("Consolas", 9))
        self.calc_label.pack(anchor="nw", padx=10, pady=10, fill=tk.BOTH, expand=True)

    def _number_control(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        low: float,
        high: float,
        fmt: str,
        callback: Callable[[], None],
        integer: bool = False,
    ) -> None:
        frame = ttk.Frame(parent, style="Side.TFrame")
        frame.pack(fill=tk.X, padx=10, pady=5)
        top = ttk.Frame(frame, style="Side.TFrame")
        top.pack(fill=tk.X)
        ttk.Label(top, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(top, width=8)
        entry.pack(side=tk.RIGHT)
        entry.insert(0, fmt.format(variable.get()))

        def commit(_event=None) -> None:
            try:
                value = float(entry.get())
            except ValueError:
                value = variable.get()
            value = max(low, min(high, value))
            if integer:
                value = round(value)
            variable.set(value)
            entry.delete(0, tk.END)
            entry.insert(0, fmt.format(value))
            callback()

        def sync_entry(_name=None, _index=None, _mode=None) -> None:
            value = round(variable.get()) if integer else variable.get()
            entry.delete(0, tk.END)
            entry.insert(0, fmt.format(value))
            callback()

        ttk.Scale(frame, from_=low, to=high, orient=tk.HORIZONTAL, variable=variable, command=lambda _v: sync_entry()).pack(
            fill=tk.X, pady=(3, 0)
        )
        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)

    def _update_scroll_region(self, _event=None) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_window(self, event: tk.Event) -> None:
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    # ---------- commands and parameter plumbing ----------

    def _on_slider(self, _value: str | None = None) -> None:
        val = self._clamp_target(float(self.target_bend.get()))
        if abs(val - float(self.target_bend.get())) > 1e-6:
            self.target_bend.set(val)
        self.bend_readout.config(text=f"{val:.1f} deg")
        self.status_text.set("MANUAL target set")

    def _apply_angle_limit(self) -> None:
        limit = max(1.0, float(self.max_bend_var.get()))
        self.bend_scale.configure(from_=-limit, to=limit)
        self.target_bend.set(self._clamp_target(float(self.target_bend.get())))
        self.bend_readout.config(text=f"{self.target_bend.get():.1f} deg")

    def _clamp_target(self, theta_deg: float) -> float:
        limit = max(1.0, float(self.max_bend_var.get()))
        theta_deg = max(-limit, min(limit, theta_deg))
        return self.model.clamp_bend_by_servo_limit(theta_deg)

    def _apply_params(self) -> None:
        old_count = self.params.plate_count
        self.params.length_mm = max(1.0, float(self.length_var.get()))
        self.params.plate_count = max(3, int(round(self.plate_count_var.get())))
        self.params.tendon_offset_mm = max(0.1, float(self.offset_var.get()))
        self.params.plate_diameter_mm = max(1.0, float(self.diameter_var.get()))
        self.params.plate_thickness_mm = max(0.2, float(self.thickness_var.get()))
        self.params.spool_radius_mm = max(0.1, float(self.spool_var.get()))

        self.physics.tendon_gain = max(0.0, float(self.tendon_gain_var.get()))
        self.physics.backbone_stiffness = max(0.0, float(self.stiffness_var.get()))
        self.physics.damping = max(0.0, float(self.damping_var.get()))
        self.physics.inertia = max(0.001, float(self.inertia_var.get()))
        self.physics.coulomb_friction = max(0.0, float(self.friction_var.get()))
        self.physics.slack_mm = max(0.0, float(self.slack_var.get()))
        self.physics.max_speed_deg_s = max(1.0, float(self.speed_var.get()))

        if self.params.plate_count != old_count:
            self._resize_plate_positions(self.params.plate_count)

        clamped = self._clamp_target(float(self.target_bend.get()))
        self.target_bend.set(clamped)
        self._on_slider()

    @staticmethod
    def _uniform_plate_positions(count: int) -> List[float]:
        count = max(3, count)
        return [i / (count - 1) for i in range(count)]

    def _resize_plate_positions(self, new_count: int) -> None:
        if new_count <= 0:
            return
        if new_count < len(self.plate_positions):
            self.plate_positions = self.plate_positions[:new_count]
            self.plate_positions[-1] = 1.0
            self._normalize_plate_positions()
            return
        while len(self.plate_positions) < new_count:
            gaps = [
                (self.plate_positions[i + 1] - self.plate_positions[i], i)
                for i in range(len(self.plate_positions) - 1)
            ]
            if not gaps:
                self.plate_positions = self._uniform_plate_positions(new_count)
                return
            _gap, index = max(gaps)
            midpoint = (self.plate_positions[index] + self.plate_positions[index + 1]) * 0.5
            self.plate_positions.insert(index + 1, midpoint)
        self._normalize_plate_positions()

    def _normalize_plate_positions(self) -> None:
        if len(self.plate_positions) != self.params.plate_count:
            self.plate_positions = self._uniform_plate_positions(self.params.plate_count)
        self.plate_positions[0] = 0.0
        self.plate_positions[-1] = 1.0
        min_gap = self._min_plate_gap_fraction()
        for i in range(1, len(self.plate_positions) - 1):
            low = self.plate_positions[i - 1] + min_gap
            high = self.plate_positions[i + 1] - min_gap
            self.plate_positions[i] = max(low, min(high, self.plate_positions[i]))

    def _min_plate_gap_fraction(self) -> float:
        return min(0.08, max(0.012, self.params.plate_thickness_mm / max(1.0, self.params.length_mm) * 1.15))

    def add_plate(self) -> None:
        if self.params.plate_count >= 24:
            self.status_text.set("Plate count already at maximum")
            return
        self.plate_count_var.set(self.params.plate_count + 1)
        self._apply_params()
        self.status_text.set(f"Added plate, N = {self.params.plate_count}")

    def remove_tip_plate(self) -> None:
        self.remove_plate(self.params.plate_count - 1)

    def reset_plate_count(self) -> None:
        self.plate_count_var.set(10)
        self._apply_params()
        self.plate_positions = self._uniform_plate_positions(self.params.plate_count)
        self.status_text.set("Plate count reset to N = 10")

    def remove_plate(self, index: int) -> None:
        if self.params.plate_count <= 3:
            self.status_text.set("Need at least 3 plates")
            return
        if not 0 <= index < self.params.plate_count:
            return
        self.plate_positions.pop(index)
        self.params.plate_count -= 1
        self.plate_count_var.set(self.params.plate_count)
        self._normalize_plate_positions()
        self._drag_plate = None
        self._apply_params()
        self.status_text.set(f"Deleted plate {index + 1}, N = {self.params.plate_count}")

    def set_command(self, theta_deg: float, status: str) -> None:
        self.auto_demo.set(False)
        self.target_bend.set(self._clamp_target(theta_deg))
        self.bend_readout.config(text=f"{self.target_bend.get():.1f} deg")
        self.status_text.set(status)

    def hold_current(self) -> None:
        self.auto_demo.set(False)
        self.target_bend.set(self.actual_bend)
        self.angular_velocity = 0.0
        self.bend_readout.config(text=f"{self.actual_bend:.1f} deg")
        self.status_text.set("HOLD current position")

    def reset_dynamics(self) -> None:
        self.actual_bend = self._clamp_target(float(self.target_bend.get()))
        self.angular_velocity = 0.0
        self.status_text.set("Dynamics reset to target")

    def reset_plate_spacing(self) -> None:
        self.plate_positions = self._uniform_plate_positions(self.params.plate_count)
        self.status_text.set("Plate spacing reset to uniform")

    def _space_step(self) -> None:
        sequence = [0, 30, 60, 90, 60, 30, 0, -30, -60, -90]
        current = float(self.target_bend.get())
        idx = min(range(len(sequence)), key=lambda i: abs(sequence[i] - current))
        nxt = sequence[(idx + 1) % len(sequence)]
        self.set_command(nxt, f"step target {nxt} deg")

    # ---------- animation and physics ----------

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, min(0.045, now - self._last_time))
        self._last_time = now

        if self.auto_demo.get():
            elapsed = now - self._demo_start
            amplitude = min(0.80 * float(self.max_bend_var.get()), abs(self._clamp_target(180.0)))
            target = amplitude * math.sin(elapsed * 1.45)
            self.target_bend.set(self._clamp_target(target))
            self.bend_readout.config(text=f"{self.target_bend.get():.1f} deg")
            self.status_text.set("AUTO DEMO")
        else:
            self._demo_start = now

        self._step_dynamics(dt)
        self._update_calc_text()
        self._draw_scene()
        self.root.after(16, self._tick)

    def _step_dynamics(self, dt: float) -> None:
        if dt <= 0:
            return

        target_deg = self._clamp_target(float(self.target_bend.get()))
        actual_rad = math.radians(self.actual_bend)
        target_rad = math.radians(target_deg)
        velocity_rad_s = math.radians(self.angular_velocity)

        desired_delta_l = 2.0 * self.params.tendon_offset_mm * target_rad
        actual_delta_l = 2.0 * self.params.tendon_offset_mm * actual_rad
        tendon_error = desired_delta_l - actual_delta_l
        active_error = math.copysign(max(0.0, abs(tendon_error) - self.physics.slack_mm), tendon_error)

        tendon_torque = self.physics.tendon_gain * active_error
        backbone_torque = -self.physics.backbone_stiffness * actual_rad
        damping_torque = -self.physics.damping * velocity_rad_s
        raw_torque = tendon_torque + backbone_torque + damping_torque
        friction_torque = self._friction(raw_torque, velocity_rad_s)
        net_torque = raw_torque + friction_torque

        acceleration_rad_s2 = net_torque / self.physics.inertia
        velocity_rad_s += acceleration_rad_s2 * dt
        max_speed = math.radians(self.physics.max_speed_deg_s)
        velocity_rad_s = max(-max_speed, min(max_speed, velocity_rad_s))
        actual_rad += velocity_rad_s * dt

        actual_deg = self.model.clamp_bend_by_servo_limit(math.degrees(actual_rad))
        if abs(actual_deg - math.degrees(actual_rad)) > 1e-6:
            velocity_rad_s = 0.0

        self.actual_bend = actual_deg
        self.angular_velocity = math.degrees(velocity_rad_s)
        self.last_snapshot = DynamicsSnapshot(
            target_deg=target_deg,
            actual_deg=self.actual_bend,
            velocity_deg_s=self.angular_velocity,
            tendon_error_mm=tendon_error,
            tendon_torque=tendon_torque,
            backbone_torque=backbone_torque,
            damping_torque=damping_torque,
            friction_torque=friction_torque,
            net_torque=net_torque,
            acceleration_deg_s2=math.degrees(acceleration_rad_s2),
        )

    def _friction(self, raw_torque: float, velocity_rad_s: float) -> float:
        limit = self.physics.coulomb_friction
        if limit <= 0:
            return 0.0
        if abs(velocity_rad_s) < 1e-4 and abs(raw_torque) < limit:
            return -raw_torque
        direction = math.copysign(1.0, velocity_rad_s if abs(velocity_rad_s) >= 1e-4 else raw_torque)
        return -limit * direction

    # ---------- live text ----------

    def _update_calc_text(self) -> None:
        k = self.model.solve(self.actual_bend)
        d = self.last_snapshot
        radius_text = "inf" if k.bend_radius_mm is None else f"{k.bend_radius_mm:7.2f} mm"
        spacings = [
            (self.plate_positions[i + 1] - self.plate_positions[i]) * self.params.length_mm
            for i in range(max(0, self.params.plate_count - 1))
        ]
        spacing_text = "n/a" if not spacings else f"{min(spacings):4.1f}/{max(spacings):4.1f} mm"
        self.calc_label.config(
            text=(
                f"target        = {d.target_deg:8.2f} deg\n"
                f"actual        = {k.theta_deg:8.2f} deg\n"
                f"velocity      = {d.velocity_deg_s:8.2f} deg/s\n"
                f"curvature     = {k.curvature_per_mm:8.5f} 1/mm\n"
                f"radius        = {radius_text}\n"
                f"plate step    = {k.per_interval_deg:8.2f} deg/int\n"
                f"L1 / L2       = {k.tendon_1_mm:6.2f} / {k.tendon_2_mm:6.2f} mm\n"
                f"delta L       = {k.delta_l_mm:8.2f} mm\n"
                f"tendon error  = {d.tendon_error_mm:8.2f} mm\n"
                f"torque net    = {d.net_torque:8.3f}\n"
                f"torque tendon = {d.tendon_torque:8.3f}\n"
                f"torque spring = {d.backbone_torque:8.3f}\n"
                f"servo 1 / 2   = {k.servo_1_deg:6.1f} / {k.servo_2_deg:6.1f} deg\n"
                f"PWM 1 / 2     = {k.pwm_1_us:6.0f} / {k.pwm_2_us:6.0f} us\n"
                f"spacing min/max= {spacing_text}"
            )
        )

    # ---------- drawing ----------

    def _draw_scene(self) -> None:
        c = self.canvas
        c.delete("all")
        width = max(760, c.winfo_width())
        height = max(620, c.winfo_height())
        self._draw_background(c, width, height)
        self._draw_actuator(c, width, height)
        self._draw_formula_panel(c, width, height)

    def _draw_background(self, c: tk.Canvas, width: int, height: int) -> None:
        c.create_rectangle(0, 0, width, height, fill="#0f172a", outline="")
        c.create_rectangle(0, 0, width, 86, fill="#111827", outline="")
        c.create_text(28, 24, anchor="nw", text="Continuum-rigid joint", font=("Segoe UI Semibold", 20), fill="#f8fafc")
        c.create_text(
            31,
            57,
            anchor="nw",
            text="Click and drag plates; tune geometry and physics while the simulation runs.",
            font=("Segoe UI", 10),
            fill="#93c5fd",
        )

    def _draw_actuator(self, c: tk.Canvas, width: int, height: int) -> None:
        p = self.params
        theta = self.actual_bend
        scale = min(4.8, max(2.2, (height - 260) / max(90.0, p.length_mm + 20.0)))
        base_x = width * 0.48
        base_y = height - 124
        self._view = (base_x, base_y, scale)
        self._plate_hitboxes = []
        self._tip_hitbox = None

        def to_screen(pt: Tuple[float, float]) -> Tuple[float, float]:
            x_mm, z_mm = pt
            return base_x + x_mm * scale, base_y - z_mm * scale

        self._draw_grid(c, base_x, base_y, scale)

        samples = 110
        u_values = [p.length_mm * i / (samples - 1) for i in range(samples)]
        center_pts = [to_screen(self.model.centerline_point(u, theta)) for u in u_values]

        c.create_rectangle(base_x - 96, base_y + 10, base_x + 96, base_y + 47, fill="#1f2937", outline="#64748b", width=2)
        c.create_rectangle(base_x - 58, base_y - 9, base_x + 58, base_y + 12, fill="#334155", outline="#94a3b8")
        c.create_text(base_x, base_y + 66, text="base mount", font=("Segoe UI", 9, "bold"), fill="#cbd5e1")

        self._polyline(c, center_pts, fill="#020617", width=8)
        self._polyline(c, center_pts, fill="#f8fafc", width=3)
        self._draw_tendon_segments(c, to_screen, theta, -p.tendon_offset_mm, "#38bdf8")
        self._draw_tendon_segments(c, to_screen, theta, p.tendon_offset_mm, "#fb7185")
        self._draw_spacing_labels(c, to_screen, theta)

        for i in range(p.plate_count):
            self._draw_plate(c, to_screen, theta, i)

        tip = to_screen(self.model.centerline_point(p.length_mm, theta))
        self._tip_hitbox = (tip[0], tip[1], 22.0)
        c.create_oval(tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6, fill="#fbbf24", outline="#111827", width=2)
        c.create_text(tip[0] + 16, tip[1] - 16, text="tip", font=("Segoe UI", 9, "bold"), fill="#fde68a")

        self._draw_angle_arc(c, base_x, base_y, theta)
        self._draw_target_angle_guide(c, base_x, base_y, scale)
        self._draw_servos(c, base_x, base_y, scale)

    def _draw_grid(self, c: tk.Canvas, base_x: float, base_y: float, scale: float) -> None:
        for mm in range(-90, 151, 20):
            x = base_x + mm * scale
            c.create_line(x, base_y + 18, x, base_y - 150 * scale, fill="#1e293b")
        for mm in range(0, 161, 20):
            y = base_y - mm * scale
            c.create_line(base_x - 100 * scale, y, base_x + 150 * scale, y, fill="#1e293b")
        c.create_line(base_x, base_y + 12, base_x, base_y - 145 * scale, fill="#475569", dash=(5, 5))
        c.create_text(base_x + 10, base_y - 132 * scale, text="neutral axis", anchor="w", font=("Segoe UI", 8), fill="#64748b")

    @staticmethod
    def _polyline(c: tk.Canvas, pts: Sequence[Tuple[float, float]], **kwargs) -> None:
        flat: List[float] = []
        for x, y in pts:
            flat.extend([x, y])
        if len(flat) >= 4:
            c.create_line(*flat, smooth=True, splinesteps=28, **kwargs)

    def _plate_u(self, index: int) -> float:
        if not 0 <= index < len(self.plate_positions):
            return 0.0
        return self.params.length_mm * self.plate_positions[index]

    def _plate_center_local(self, index: int, theta: float) -> Tuple[float, float]:
        return self.model.centerline_point(self._plate_u(index), theta)

    def _plate_tendon_hole_local(self, index: int, theta: float, tendon_offset: float) -> Tuple[float, float]:
        center = self._plate_center_local(index, theta)
        _, normal = self.model.tangent_normal(self._plate_u(index), theta)
        return center[0] + tendon_offset * normal[0], center[1] + tendon_offset * normal[1]

    def _draw_tendon_segments(
        self,
        c: tk.Canvas,
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        theta: float,
        tendon_offset: float,
        color: str,
    ) -> None:
        points = [to_screen(self._plate_tendon_hole_local(i, theta, tendon_offset)) for i in range(self.params.plate_count)]
        if len(points) < 2:
            return
        for a, b in zip(points, points[1:]):
            c.create_line(a[0], a[1], b[0], b[1], fill="#0f172a", width=6)
            c.create_line(a[0], a[1], b[0], b[1], fill=color, width=3)

    def _draw_spacing_labels(
        self,
        c: tk.Canvas,
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        theta: float,
    ) -> None:
        if self.params.plate_count < 2:
            return
        for i in range(self.params.plate_count - 1):
            ua = self._plate_u(i)
            ub = self._plate_u(i + 1)
            mid = self.model.centerline_point((ua + ub) * 0.5, theta)
            _, normal = self.model.tangent_normal((ua + ub) * 0.5, theta)
            label_pt = (mid[0] - normal[0] * (self.params.plate_diameter_mm * 0.72), mid[1] - normal[1] * (self.params.plate_diameter_mm * 0.72))
            sx, sy = to_screen(label_pt)
            c.create_text(
                sx,
                sy,
                text=f"{ub - ua:.0f}",
                font=("Segoe UI", 7, "bold"),
                fill="#cbd5e1",
            )

    def _draw_plate(self, c: tk.Canvas, to_screen: Callable[[Tuple[float, float]], Tuple[float, float]], theta: float, index: int) -> None:
        p = self.params
        u = self._plate_u(index)
        center = self._plate_center_local(index, theta)
        tangent, normal = self.model.tangent_normal(u, theta)
        half_w = p.plate_diameter_mm / 2.0
        half_t = p.plate_thickness_mm / 2.0

        def add(sa: float, sb: float) -> Tuple[float, float]:
            return (
                center[0] + sa * normal[0] * half_w + sb * tangent[0] * half_t,
                center[1] + sa * normal[1] * half_w + sb * tangent[1] * half_t,
            )

        pts = [to_screen(pt) for pt in (add(-1, -1), add(1, -1), add(1, 1), add(-1, 1))]
        flat = [coord for point in pts for coord in point]
        fill = "#dbeafe" if index % 2 == 0 else "#bfdbfe"
        outline = "#f59e0b" if index == self._drag_plate else "#1e3a8a"
        c.create_polygon(*flat, fill=fill, outline=outline, width=2)

        face_a = to_screen(add(-0.58, -0.62))
        face_b = to_screen(add(0.58, 0.62))
        c.create_line(face_a[0], face_a[1], face_b[0], face_b[1], fill="#2563eb", width=2)

        for tendon_off, color in ((p.tendon_offset_mm, "#fb7185"), (-p.tendon_offset_mm, "#38bdf8")):
            hole = self._plate_tendon_hole_local(index, theta, tendon_off)
            hx, hy = to_screen(hole)
            c.create_oval(hx - 3.5, hy - 3.5, hx + 3.5, hy + 3.5, fill=color, outline="#0f172a", width=1)

        sx, sy = to_screen(center)
        c.create_text(sx, sy, text=str(index + 1), font=("Segoe UI", 7, "bold"), fill="#0f172a")
        self._plate_hitboxes.append((index, sx, sy, max(14.0, half_w * self._view[2])))

    def _draw_angle_arc(self, c: tk.Canvas, base_x: float, base_y: float, theta_deg: float) -> None:
        radius = 40
        c.create_arc(
            base_x - radius,
            base_y - radius,
            base_x + radius,
            base_y + radius,
            start=-90,
            extent=-theta_deg,
            style=tk.ARC,
            width=3,
            outline="#fbbf24",
        )
        c.create_text(base_x + 58, base_y - 46, text=f"Theta = {theta_deg:.1f} deg", font=("Segoe UI", 11, "bold"), fill="#fde68a")

    def _draw_target_angle_guide(self, c: tk.Canvas, base_x: float, base_y: float, scale: float) -> None:
        target_deg = float(self.target_bend.get())
        angle = math.radians(target_deg)
        length_px = min(110.0, max(65.0, self.params.length_mm * scale * 0.42))
        x = base_x + math.sin(angle) * length_px
        y = base_y - math.cos(angle) * length_px
        c.create_line(base_x, base_y, x, y, fill="#fbbf24", width=2, dash=(5, 5))
        c.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#fbbf24", outline="#0f172a", width=2)
        c.create_text(x + 12, y, anchor="w", text=f"target {target_deg:.0f}", font=("Segoe UI", 8, "bold"), fill="#fef3c7")

    def _draw_servos(self, c: tk.Canvas, base_x: float, base_y: float, scale: float) -> None:
        k = self.model.solve(self.actual_bend)
        y = base_y + 104
        left_x = base_x - 150
        right_x = base_x + 150
        radius = 34
        self._draw_servo(c, left_x, y, radius, k.servo_1_deg, "Servo 1", "#38bdf8")
        self._draw_servo(c, right_x, y, radius, k.servo_2_deg, "Servo 2", "#fb7185")
        bneg = self._screen_from_local(base_x, base_y, (0, 0), scale, -self.params.tendon_offset_mm)
        bpos = self._screen_from_local(base_x, base_y, (0, 0), scale, self.params.tendon_offset_mm)
        c.create_line(left_x, y - radius, bneg[0], bneg[1], fill="#38bdf8", width=2, dash=(6, 4))
        c.create_line(right_x, y - radius, bpos[0], bpos[1], fill="#fb7185", width=2, dash=(6, 4))

    @staticmethod
    def _screen_from_local(
        base_x: float,
        base_y: float,
        center: Tuple[float, float],
        scale: float,
        offset_mm: float = 0.0,
    ) -> Tuple[float, float]:
        return base_x + (center[0] + offset_mm) * scale, base_y - center[1] * scale

    @staticmethod
    def _draw_servo(c: tk.Canvas, x: float, y: float, radius: float, angle_deg: float, label: str, color: str) -> None:
        c.create_rectangle(x - 50, y + 38, x + 50, y + 68, fill="#1f2937", outline="#64748b", width=2)
        c.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#e2e8f0", outline="#0f172a", width=2)
        c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#0f172a", outline="")
        a = math.radians(angle_deg - 90)
        px = x + math.sin(a) * (radius - 4)
        py = y - math.cos(a) * (radius - 4)
        c.create_line(x, y, px, py, fill=color, width=5)
        c.create_text(x, y + 86, text=f"{label}: {angle_deg:.1f} deg", font=("Segoe UI", 9, "bold"), fill="#dbeafe")

    def _draw_formula_panel(self, c: tk.Canvas, width: int, height: int) -> None:
        k = self.model.solve(self.actual_bend)
        x0 = max(24, width - 350)
        y0 = 112
        x1 = width - 24
        y1 = 354
        c.create_rectangle(x0, y0, x1, y1, fill="#111827", outline="#334155", width=2)
        c.create_text(x0 + 14, y0 + 12, anchor="nw", text="Physics core", font=("Segoe UI Semibold", 12), fill="#f8fafc")
        text = (
            "Kinematics:\n"
            "  kappa = theta / L\n"
            "  L1 = L - r_c theta\n"
            "  L2 = L + r_c theta\n\n"
            "Dynamics:\n"
            "  eL = commanded_deltaL - actual_deltaL\n"
            "  torque = tendon(eL - slack)\n"
            "           - backbone(theta)\n"
            "           - damping(theta_dot)\n"
            "           - friction\n\n"
            f"delta L = {k.delta_l_mm:.2f} mm\n"
            f"servo delta = {k.servo_delta_deg:.2f} deg"
        )
        c.create_text(x0 + 14, y0 + 42, anchor="nw", text=text, font=("Consolas", 9), fill="#cbd5e1")

    # ---------- plate dragging ----------

    def _on_canvas_press(self, event: tk.Event) -> None:
        hit = self._hit_plate(event.x, event.y)
        if hit is not None:
            self._drag_mode = "spacing"
            self._drag_plate = hit
            self._set_plate_spacing_from_canvas(hit, event.x, event.y)
            return
        self._drag_mode = "angle"
        self._drag_plate = None
        self._set_target_from_canvas(event.x, event.y)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_mode == "angle":
            self._set_target_from_canvas(event.x, event.y)
            return
        if self._drag_mode == "spacing" and self._drag_plate is not None:
            self._set_plate_spacing_from_canvas(self._drag_plate, event.x, event.y)
            return

    def _on_canvas_release(self, _event: tk.Event) -> None:
        if self._drag_mode == "spacing" and self._drag_plate is not None:
            self.status_text.set(f"Plate {self._drag_plate + 1} spacing set at {self._plate_u(self._drag_plate):.1f} mm")
        elif self._drag_mode == "angle":
            self.status_text.set(f"Angle target released at {self.target_bend.get():.1f} deg")
        self._drag_mode = None
        self._drag_plate = None

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        hit = self._hit_plate(event.x, event.y)
        if hit is not None:
            self.remove_plate(hit)

    def _set_target_from_canvas(self, x: float, y: float) -> None:
        base_x, base_y, _scale = self._view
        dx = x - base_x
        up = base_y - y
        if abs(dx) < 1e-6 and abs(up) < 1e-6:
            return
        theta = math.degrees(math.atan2(dx, up))
        theta = self._clamp_target(theta)
        self.auto_demo.set(False)
        self.target_bend.set(theta)
        self.bend_readout.config(text=f"{theta:.1f} deg")
        self.status_text.set(f"Angle drag target {theta:.1f} deg")

    def _set_plate_spacing_from_canvas(self, index: int, x: float, y: float) -> None:
        if index <= 0 or index >= self.params.plate_count - 1:
            self.status_text.set("Base and tip plates stay fixed; drag middle plates to change spacing")
            return

        target_fraction = self._closest_fraction_on_centerline(x, y)
        min_gap = self._min_plate_gap_fraction()
        low = self.plate_positions[index - 1] + min_gap
        high = self.plate_positions[index + 1] - min_gap
        self.plate_positions[index] = max(low, min(high, target_fraction))

        before = (self.plate_positions[index] - self.plate_positions[index - 1]) * self.params.length_mm
        after = (self.plate_positions[index + 1] - self.plate_positions[index]) * self.params.length_mm
        self.status_text.set(f"Plate {index + 1} spacing: {before:.1f} mm / {after:.1f} mm")

    def _closest_fraction_on_centerline(self, x: float, y: float) -> float:
        base_x, base_y, scale = self._view
        best_fraction = 0.0
        best_dist2 = float("inf")
        sample_count = 140
        for i in range(sample_count + 1):
            fraction = i / sample_count
            local = self.model.centerline_point(self.params.length_mm * fraction, self.actual_bend)
            sx = base_x + local[0] * scale
            sy = base_y - local[1] * scale
            dist2 = (x - sx) ** 2 + (y - sy) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_fraction = fraction
        return best_fraction

    def _hit_plate(self, x: float, y: float) -> int | None:
        best: Tuple[float, int] | None = None
        for index, sx, sy, radius in self._plate_hitboxes:
            dist2 = (x - sx) ** 2 + (y - sy) ** 2
            if dist2 <= radius**2 and (best is None or dist2 < best[0]):
                best = (dist2, index)
        return None if best is None else best[1]


# -----------------------------
# Entry point
# -----------------------------


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.12)
    except tk.TclError:
        pass
    TDCRSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
