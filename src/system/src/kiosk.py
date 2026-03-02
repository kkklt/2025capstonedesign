#!/usr/bin/env python3
"""
Touch-friendly ROS1 kiosk UI.

This script provides a PyQt5-based kiosk that can publish JSON-formatted
orders (or commands) to other ROS nodes while showing live status feedback.

- Version History
- v0.11.1 (2025-11-24): 조리/컵/트레이/경로 노드를 담당자 사양으로 복원, 자동 시퀀스 로직을 간결화하고 메뉴 이미지 캐시로 렌더링 속도를 향상.
- v0.11.0 (2025-01-XX): 고객/개발자 UI 테마 전면 개편, 카드형 메뉴와 이미지 지원, 공통 스타일 정비.
- v0.10.7 (2025-11-20): Added Doosan Gazebo test motion nodes, 고객 메뉴 항목, 및 순차 실행 테스트 지원.
- v0.10.6 (2025-11-20): Added 실행 노드 목록 카탈로그와 시퀀스 리스트 번호 표시.
- v0.10.5 (2025-11-20): Added 단일 워커 가드, 주문 시퀀스 커스터마이저, 사용자 순서 적용.
- v0.10.4 (2025-11-20): 고객 화면 진행/로그 패널과 이미지 영역 제거, 워크플로우 표시 로직 가드.
- v0.10.3 (2025-11-20): 창 고정을 복원하고 고객 패널을 스플리터 기반 비율 레이아웃으로 재구성.
- v0.10.2 (2025-11-20): 고객 화면 패널을 비율 기반으로 재배치하고 창 고정 제약을 해제.
- v0.10.1 (2025-11-20): Added 고객 화면 실시간 로그 패널과 로그 공유 메서드.
- v0.10.0 (2025-11-20): Added 비동기 워크플로우 스레드, 순차 실행 보장, 고객 화면 진행 표시, 시나리오 실행 안정화.
- v0.9.0 (2025-11-18): Added 트레이 복귀 노드와 자동 복귀 시퀀스 및 개발자 카드.
- v0.8.0 (2025-11-18): Added tray pick/move/drop nodes plus 기본 목적지 설정 UI와 유연한 트레이 라우팅.
- v0.7.0 (2025-11-18): Added five dedicated 테스트 노드 스크립트 and wired each scenario to its own executable.
- v0.6.1 (2025-11-18): Removed automatic path planner execution from customer workflow and added flexible 음료 2잔 로직 with tray-per-drink routing.
- v0.6.0 (2025-11-18): Fixed indentation regressions, restored global constants, and added tray selectors for developer nodes plus OrderLine summary fix.
- v0.5.0 (2025-11-17): Introduced node runner workflow, tray/destination automation, and UI color refresh.
- v0.4.0 (2025-11-15): Added seamless 음료 선택 화면 전환, refreshed kiosk/developer UI styling, and improved scenario grouping.
- v0.3.0 (2025-11-15): Added selectable 기본 구성, automatic drink prompt flow, Doosan M1013 OctoMap path module toggle, and portrait sizing tweaks.

Typical usage inside a catkin workspace:
$ python3 keyhosk.py --order-topic /kiosk/order --status-topic /robot/status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpacerItem,
        QStackedWidget,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyQt5가 설치되어 있지 않습니다. `sudo apt install python3-pyqt5`로 설치하세요."
    ) from exc

try:
    import rospy
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "ROS Python 환경에서 실행해야 합니다. `source /opt/ros/<distro>/setup.bash` 이후 다시 실행하세요."
    ) from exc


@dataclass
class MenuItem:
    """주문의 기본 단위를 정의."""

    name: str
    price: int
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    image: Optional[str] = None


@dataclass
class OrderLine:
    """선택된 항목과 수량 관리."""

    item: MenuItem
    quantity: int = 1

    @property
    def total(self) -> int:
        return self.item.price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.item.name,
            "price": self.item.price,
            "quantity": self.quantity,
            "total": self.total,
            "payload": self.item.payload,
        }
        if self.item.description:
            data["description"] = self.item.description
        return data


class OrderModel:
    """주문 내역을 관리하고 직렬화한다."""

    def __init__(self) -> None:
        self._lines: List[OrderLine] = []

    def add_item(self, item: MenuItem) -> None:
        for line in self._lines:
            if line.item.name == item.name:
                line.quantity += 1
                return
        self._lines.append(OrderLine(item=item))

    def remove_index(self, index: int) -> None:
        if 0 <= index < len(self._lines):
            del self._lines[index]

    def remove_by_name(self, name: str) -> None:
        self._lines = [line for line in self._lines if line.item.name != name]

    def clear(self) -> None:
        self._lines.clear()

    def contains(self, name: str) -> bool:
        return any(line.item.name == name for line in self._lines)

    def total(self) -> int:
        return sum(line.total for line in self._lines)

    def serialize(self) -> Dict[str, Any]:
        return {
            "items": [line.to_dict() for line in self._lines],
            "total": self.total(),
        }

    @property
    def lines(self) -> List[OrderLine]:
        return self._lines

    def __bool__(self) -> bool:
        return bool(self._lines)


DEFAULT_MENU: List[Dict[str, Any]] = [
    {
        "category": "기본 메뉴",
        "items": [
            {
                "name": "치킨 튀김",
                "price": 15000,
                "payload": {"menu": "chicken_fry"},
                "description": "프라이어 기반 치킨 튀김 시나리오.",
                "image": "images/chicken.png",
            },
            {
                "name": "닭꼬치 구이",
                "price": 8000,
                "payload": {"menu": "skewer_roast"},
                "description": "석쇠 구이 기반 닭꼬치 조리 시나리오.",
                "image": "images/skewer.png",
            },
        ],
    },
    {
        "category": "음료 옵션",
        "items": [
            {
                "name": "탄산 음료",
                "price": 2500,
                "payload": {"menu": "drink"},
                "description": "기본 음료 1잔 (선택 사항).",
                "image": "images/drink.png",
            }
        ],
    },
]

DEFAULT_DESTINATIONS: Tuple[str, str] = ("A존", "B존")
HOME_DESTINATION = "대기존"
TRAY_IDS: Tuple[str, str] = ("tray_a", "tray_b")
TRAY_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("tray_a", "트레이 A"),
    ("tray_b", "트레이 B"),
)
DEFAULT_TRAY_DEST_MAP: Dict[str, str] = {
    "tray_a": DEFAULT_DESTINATIONS[0],
    "tray_b": DEFAULT_DESTINATIONS[1],
}
NODE_COMMANDS: Dict[str, Dict[str, Any]] = {
    "chicken_cook_to_tray1": {
        "label": "치킨 조리 노드 (트레이 A)",
        "description": "치킨 튀김 공정을 실행해 트레이 A에 적재합니다.",
        "command": ["rosrun", "system", "frying1.py"],
        "default_tray": "tray_a",
    },
    "chicken_cook_to_tray2": {
        "label": "치킨 조리 노드 (트레이 B)",
        "description": "치킨 튀김 공정을 실행해 트레이 B에 적재합니다.",
        "command": ["rosrun", "system", "frying2.py"],
        "default_tray": "tray_b",
    },
    "skewer_cook_to_tray1": {
        "label": "닭꼬치 조리 노드 (트레이 A)",
        "description": "닭꼬치 구이 공정을 실행해 트레이 A에 적재합니다.",
        "command": ["rosrun", "system", "grilling1.py"],
        "default_tray": "tray_a",
    },
    "skewer_cook_to_tray2": {
        "label": "닭꼬치 조리 노드 (트레이 B)",
        "description": "닭꼬치 구이 공정을 실행해 트레이 B에 적재합니다.",
        "command": ["rosrun", "system", "grilling2.py"],
        "default_tray": "tray_b",
    },
    "cup_transfer_to_tray1": {
        "label": "컵 적재 노드 (트레이 A)",
        "description": "음료 컵을 감지해 트레이 A에 적재합니다.",
        "command": ["rosrun", "system", "cup1.py"],
        "default_tray": "tray_a",
    },
    "cup_transfer_to_tray2": {
        "label": "컵 적재 노드 (트레이 B)",
        "description": "음료 컵을 감지해 트레이 B에 적재합니다.",
        "command": ["rosrun", "system", "cup2.py"],
        "default_tray": "tray_b",
    },
    "tray_transfer_to_destination1": {
        "label": "트레이 이송 노드 (목적지 1)",
        "description": "트레이를 파지하고 목적지 1로 이동/전달합니다.",
        "command": ["rosrun", "system", "tray1.py"],
        "default_tray": "tray_a",
    },
    "tray_transfer_to_destination2": {
        "label": "트레이 이송 노드 (목적지 2)",
        "description": "트레이를 파지하고 목적지 2로 이동/전달합니다.",
        "command": ["rosrun", "system", "tray2.py"],
        "default_tray": "tray_b",
    },
    "path_planner": {
        "label": "옥토맵 경로 생성 노드",
        "description": "사전 환경 인식 데이터로 경로를 생성합니다.",
        "command": ["rosrun", "system", "cup1.py"],
    },
}


def load_menu(source: Optional[Path]) -> List[Dict[str, Any]]:
    """JSON 파일이 존재하면 로드하고, 없으면 기본 메뉴를 반환."""
    if source and source.exists():
        with source.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
            if not isinstance(data, list):
                raise ValueError("메뉴 JSON은 List[Dict] 형태여야 합니다.")
            return data
    return DEFAULT_MENU


def build_menu_map(raw_sections: List[Dict[str, Any]]) -> Dict[str, List[MenuItem]]:
    """카테고리별 MenuItem 리스트를 구성."""
    menu: Dict[str, List[MenuItem]] = {}
    for section in raw_sections:
        category = section.get("category", "Misc")
        items = []
        for raw in section.get("items", []):
            items.append(
                MenuItem(
                    name=raw["name"],
                    price=int(raw["price"]),
                    description=raw.get("description", ""),
                    payload=raw.get("payload", {}),
                    image=raw.get("image"),
                )
            )
        if items:
            menu[category] = items
    if not menu:
        raise ValueError("메뉴에 표시할 항목이 없습니다.")
    return menu


class RosBridge(QObject):
    """ROS Pub/Sub을 캡슐화."""

    status_received = pyqtSignal(str)

    def __init__(
        self,
        node_name: str,
        order_topic: str,
        status_topic: Optional[str],
        namespace: Optional[str],
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.node_name = node_name
        self.order_topic = order_topic
        self.status_topic = status_topic
        self.namespace = namespace.strip("/") if namespace else None
        self.dry_run = dry_run
        self._publisher: Optional[Any] = None
        self._status_sub = None

    def _resolve(self, name: str) -> str:
        if name.startswith("/"):
            return name
        if self.namespace:
            return f"/{self.namespace}/{name}"
        return name

    def start(self) -> None:
        if self.dry_run:
            return
        if not rospy.core.is_initialized():
            rospy.init_node(
                self.node_name,
                anonymous=True,
                disable_signals=True,
            )
        self._publisher = rospy.Publisher(
            self._resolve(self.order_topic),
            String,
            queue_size=10,
        )
        if self.status_topic:
            self._status_sub = rospy.Subscriber(
                self._resolve(self.status_topic),
                String,
                self._on_status,
            )

    def _on_status(self, msg: String) -> None:
        self.status_received.emit(msg.data)

    def publish(self, payload: Dict[str, Any]) -> bool:
        message = json.dumps(payload, ensure_ascii=False)
        if self.dry_run:
            print(f"[DRY RUN] publish -> {message}")
            return True
        if not self._publisher:
            rospy.logwarn("ROS publisher가 아직 준비되지 않았습니다.")
            return False
        self._publisher.publish(String(data=message))
        return True

    def publish_command(
        self,
        scenario_id: str,
        action: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """시나리오 제어용 공통 명령 헬퍼."""

        payload: Dict[str, Any] = {
            "type": "scenario_command",
            "scenario": scenario_id,
            "action": action,
        }
        if extra:
            payload["metadata"] = extra
        return self.publish(payload)


class NodeRunner:
    """External node executor with easy-to-edit command table."""

    def __init__(self, *, dry_run: bool, node_commands: Dict[str, Dict[str, Any]]) -> None:
        self.dry_run = dry_run
        self.node_commands = node_commands

    def run_node(
        self,
        node_id: str,
        *,
        tray: Optional[str] = None,
        destination: Optional[str] = None,
    ) -> int:
        spec = self.node_commands.get(node_id)
        if not spec:
            print(f"[NODE] Unknown node_id: {node_id}")
            return -1
        command = list(spec.get("command", []))
        if tray:
            command += ["--tray", tray]
        if destination:
            command += ["--destination", destination]

        if self.dry_run:
            print(f"[NODE-DRY] {' '.join(command)}")
            return 0

        try:
            proc = subprocess.Popen(command)
            proc.wait()
            return proc.returncode
        except FileNotFoundError as exc:
            print(f"[NODE-ERR] {command} 실행 실패: {exc}")
            return -1


class WorkflowThread(QThread):
    """순차적으로 노드를 실행하고 진행 상황을 신호로 알리는 워커."""

    step_started = pyqtSignal(int, dict)
    step_finished = pyqtSignal(int, dict)
    sequence_finished = pyqtSignal(bool)

    def __init__(
        self,
        *,
        steps: List[Tuple[str, Optional[str], Optional[str]]],
        runner: NodeRunner,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.runner = runner
        self._stop_requested = False

    def run(self) -> None:
        success = True
        for index, (node_id, tray, destination) in enumerate(self.steps):
            if self._stop_requested:
                success = False
                break
            info = {
                "node_id": node_id,
                "tray": tray,
                "destination": destination,
            }
            self.step_started.emit(index, info)
            result = self.runner.run_node(node_id, tray=tray, destination=destination)
            info["result"] = result
            self.step_finished.emit(index, info)
            if result != 0:
                success = False
                break
        self.sequence_finished.emit(success)

    def request_stop(self) -> None:
        self._stop_requested = True
@dataclass(frozen=True)
class Scenario:
    """개발자 화면에서 제어할 핵심 시나리오 정의."""

    id: str
    label: str
    description: str
    action_hint: str = ""
    needs_destination: bool = False
    tray_selectable: bool = False


SCENARIOS: List[Scenario] = [
    Scenario(
        id="chicken_cook_to_tray1",
        label=NODE_COMMANDS["chicken_cook_to_tray1"]["label"],
        description=NODE_COMMANDS["chicken_cook_to_tray1"]["description"],
        action_hint="트레이 A 적재 전에 오일 온도 등을 확인하세요.",
    ),
    Scenario(
        id="chicken_cook_to_tray2",
        label=NODE_COMMANDS["chicken_cook_to_tray2"]["label"],
        description=NODE_COMMANDS["chicken_cook_to_tray2"]["description"],
    ),
    Scenario(
        id="skewer_cook_to_tray1",
        label=NODE_COMMANDS["skewer_cook_to_tray1"]["label"],
        description=NODE_COMMANDS["skewer_cook_to_tray1"]["description"],
    ),
    Scenario(
        id="skewer_cook_to_tray2",
        label=NODE_COMMANDS["skewer_cook_to_tray2"]["label"],
        description=NODE_COMMANDS["skewer_cook_to_tray2"]["description"],
    ),
    Scenario(
        id="cup_transfer_to_tray1",
        label=NODE_COMMANDS["cup_transfer_to_tray1"]["label"],
        description=NODE_COMMANDS["cup_transfer_to_tray1"]["description"],
    ),
    Scenario(
        id="cup_transfer_to_tray2",
        label=NODE_COMMANDS["cup_transfer_to_tray2"]["label"],
        description=NODE_COMMANDS["cup_transfer_to_tray2"]["description"],
    ),
    Scenario(
        id="tray_transfer_to_destination1",
        label=NODE_COMMANDS["tray_transfer_to_destination1"]["label"],
        description=NODE_COMMANDS["tray_transfer_to_destination1"]["description"],
    ),
    Scenario(
        id="tray_transfer_to_destination2",
        label=NODE_COMMANDS["tray_transfer_to_destination2"]["label"],
        description=NODE_COMMANDS["tray_transfer_to_destination2"]["description"],
    ),
]


ORDERABLE_NODE_IDS: Tuple[str, ...] = (
    "chicken_cook_to_tray1",
    "chicken_cook_to_tray2",
    "skewer_cook_to_tray1",
    "skewer_cook_to_tray2",
    "cup_transfer_to_tray1",
    "cup_transfer_to_tray2",
    "tray_transfer_to_destination1",
    "tray_transfer_to_destination2",
)
DEFAULT_WORKFLOW_ORDER: List[str] = list(ORDERABLE_NODE_IDS)


class KioskWindow(QMainWindow):
    """고객/개발자 화면을 모두 제공하는 통합 키오스크 UI."""

    def __init__(
        self,
        ros_bridge: RosBridge,
        menu_map: Dict[str, List[MenuItem]],
        *,
        app_args: Optional[argparse.Namespace] = None,
        title: str,
        lock_window_size: bool = True,
    ) -> None:
        super().__init__()
        self.ros_bridge = ros_bridge
        self.menu_map = menu_map
        self.item_categories: Dict[str, str] = {
            item.name: category for category, items in menu_map.items() for item in items
        }
        self.order = OrderModel()
        self.current_category = next(iter(menu_map.keys()))
        self.scenario_lookup = {scenario.id: scenario for scenario in SCENARIOS}
        self.scenario_states: Dict[str, bool] = {scenario.id: True for scenario in SCENARIOS}
        self.scenario_widgets: Dict[str, Dict[str, Any]] = {}
        self.dev_status_summary: Dict[str, QLabel] = {}
        self.base_category_name = "기본 메뉴" if "기본 메뉴" in menu_map else next(iter(menu_map.keys()))
        self.drink_category_name: Optional[str] = "음료 옵션" if "음료 옵션" in menu_map else None
        self.drink_item: Optional[MenuItem] = (
            menu_map[self.drink_category_name][0] if self.drink_category_name else None
        )
        self.awaiting_drink = False
        self.pending_drink_tray: Optional[str] = None
        self.drink_tray_overrides: List[str] = []
        self.tray_destination_defaults: Dict[str, str] = dict(DEFAULT_TRAY_DEST_MAP)
        self._aspect_ratio = (9 * 1.6) / 16  # width / height for portrait-friendly window
        self.app_args = app_args
        self.in_fullscreen_mode = False
        self.node_runner = NodeRunner(dry_run=ros_bridge.dry_run, node_commands=NODE_COMMANDS)
        self.workflow_thread: Optional[WorkflowThread] = None
        self.workflow_steps_meta: List[Dict[str, Any]] = []
        self.scenario_threads: Dict[str, WorkflowThread] = {}
        self.lock_window_size = lock_window_size
        self.workflow_progress_list: Optional[QListWidget] = None
        self.workflow_progress_bar: Optional[QProgressBar] = None
        self.workflow_status_label: Optional[QLabel] = None
        self.menu_image_placeholder: Optional[QLabel] = None
        self.workflow_custom_order: List[str] = list(DEFAULT_WORKFLOW_ORDER)
        self.sequence_list: Optional[QListWidget] = None
        self.workflow_catalog_list: Optional[QListWidget] = None
        self._menu_image_size: Tuple[int, int] = (180, 140)
        self._pixmap_cache: Dict[str, QPixmap] = {}

        self.setWindowTitle(title)
        self._build_main_layout()
        self._apply_theme()
        self._apply_initial_size()
        if self.lock_window_size:
            self._lock_initial_size()
        self._connect_signals()
        self._render_categories()
        self._render_items(self.current_category)
        self._refresh_summary()

    # ------------------------------------------------------------------ #
    # UI 빌드
    # ------------------------------------------------------------------ #
    def _build_main_layout(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout()
        root.setLayout(root_layout)
        self.setCentralWidget(root)

        self.nav_buttons: Dict[str, QPushButton] = {}
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(12)
        root_layout.addLayout(nav_layout)

        self.nav_buttons["customer"] = self._create_nav_button("고객 화면")
        self.nav_buttons["developer"] = self._create_nav_button("개발자 화면")
        nav_layout.addWidget(self.nav_buttons["customer"])
        nav_layout.addWidget(self.nav_buttons["developer"])
        nav_layout.addStretch(1)

        self.view_stack = QStackedWidget()
        root_layout.addWidget(self.view_stack)

        self.customer_page = self._build_customer_page()
        self.developer_page = self._build_developer_page()
        self.view_stack.addWidget(self.customer_page)
        self.view_stack.addWidget(self.developer_page)

        self._set_active_view("customer")

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #fcf8f2;
                font-family: 'Pretendard', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
                color: #333333;
            }
            QLabel {
                background-color: transparent;
            }
            /* 상단 헤더 영역 */
            QFrame#customerHero {
                background-color: #ffe0b2;
                border-radius: 0px;
                border-bottom: 2px solid #ffdbb0;
            }
            QLabel#heroTitle {
                color: #1f1b16;
                font-weight: 800;
                font-size: 42px;
                background-color: transparent;
            }
            QLabel#heroSubtitle {
                color: #5d4037;
                font-size: 20px;
                background-color: transparent;
            }

            /* 좌측 카테고리 사이드바 */
            QFrame#categoryCard {
                background-color: #fff3e0;
                border-radius: 16px;
                border: 1px solid #ffe0b2;
            }
            QPushButton#categoryButton {
                text-align: left;
                padding: 16px 24px;
                border-radius: 12px;
                font-size: 18px;
                font-weight: 600;
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                color: #555;
                margin-bottom: 8px;
            }
            QPushButton#categoryButton:checked, QPushButton#categoryButton:hover {
                background-color: #ffb74d;
                border: none;
                color: white;
            }

            /* 중앙 메뉴 아이템 */
            QFrame#itemsCard {
                background-color: transparent;
                border: none;
            }
            QPushButton#menuButton {
                background-color: #ffffff;
                border-radius: 16px;
                border: 2px solid #eeeeee;
                font-size: 20px;
                padding: 16px;
                color: #333;
            }
            QPushButton#menuButton:hover {
                border-color: #ffb74d;
                background-color: #fff8f0;
            }
            QPushButton#menuButton:pressed {
                background-color: #ffe0b2;
            }

            /* 우측 주문 요약 & 총액 */
            QFrame#summaryCard {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #e0e0e0;
            }
            QListWidget {
                background-color: #f9f9f9;
                border: 1px solid #eeeeee;
                border-radius: 12px;
                padding: 12px;
                font-size: 16px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QLabel#totalLabel {
                background-color: #ffe0b2;
                border-radius: 12px;
                padding: 16px;
                color: #3e2723;
                font-weight: bold;
                font-size: 24px;
            }

            /* 하단 액션 버튼 */
            QPushButton#subActionButton {
                background-color: #e0e0e0;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
                color: #333;
            }
            QPushButton#subActionButton:hover {
                background-color: #d5d5d5;
            }
            QPushButton#sendButton {
                background-color: #ff9800;
                border: none;
                border-radius: 16px;
                padding: 20px;
                font-size: 24px;
                font-weight: bold;
                color: white;
            }
            QPushButton#sendButton:hover {
                background-color: #f57c00;
            }
            QPushButton#sendButton:disabled {
                background-color: #cccccc;
            }
            
            /* 로봇 상태 바 */
            QLabel#statusLabel {
                color: #666;
                background-color: #eeeeee;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 14px;
            }
            
            /* 기타 공통 */
            QProgressBar {
                background: #eeeeee;
                border-radius: 8px;
                height: 12px;
            }
            QProgressBar::chunk {
                background: #ff9800;
                border-radius: 8px;
            }
            
            /* 입력 컨트롤 스타일 (콤보박스, 체크박스 등) */
            QComboBox {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #333333;
                selection-background-color: #ffe0b2;
                selection-color: #333333;
                border: 1px solid #d0d0d0;
            }
            QCheckBox {
                color: #333333;
                font-size: 14px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #aaa;
                background: #fff;
            }
            QCheckBox::indicator:checked {
                background-color: #ff9800;
                border-color: #ff9800;
            }

            /* --- 개발자 화면 전용 스타일 --- */
            QFrame#developerHero {
                background-color: #ffe0b2;
                border-radius: 0px;
                border-bottom: 2px solid #ffdbb0;
            }
            /* 기능 블록 카드 스타일 */
            QFrame#scenarioCard, QFrame#devInfoCard {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 16px;
                margin-bottom: 12px;
            }
            /* 섹션 제목 스타일 */
            QLabel#devSectionTitle {
                color: #e65100;
                font-size: 18px;
                font-weight: 800;
                margin-bottom: 4px;
                padding-left: 4px;
            }
            /* 그룹박스 (로그/상태창) */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 12px;
                margin-top: 1.5em;
                font-weight: bold;
                color: #333;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 12px;
                color: #ef6c00;
                font-size: 16px;
            }
            QPlainTextEdit {
                background-color: #fffbf5;
                border: 1px solid #ffe0b2;
                border-radius: 8px;
                font-family: 'Consolas', 'Monospace';
            }
            """
        )

    def _apply_initial_size(self) -> None:
        if self.in_fullscreen_mode:
            screen = QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                self.setGeometry(geom)
            return

        if self.app_args:
            target_width = getattr(self.app_args, "window_width", None)
            target_height = getattr(self.app_args, "window_height", None)
            if target_width and target_height:
                self.resize(target_width, target_height)
                return

        screen = QApplication.primaryScreen()
        default_height = 1280
        if not screen:
            self.resize(int(default_height * self._aspect_ratio), default_height)
            return

        avail = screen.availableGeometry()
        target_height = max(900, int(avail.height() * 0.9))
        target_height = min(target_height, avail.height())
        target_width = int(target_height * self._aspect_ratio)

        if target_width > avail.width():
            target_width = int(avail.width() * 0.9)
            target_height = int(target_width / self._aspect_ratio)

        self.resize(target_width, target_height)

    def _lock_initial_size(self) -> None:
        if self.in_fullscreen_mode:
            return
        initial_size = self.size()
        self.setMinimumSize(initial_size)
        self.setMaximumSize(initial_size)

    def _create_nav_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setMinimumHeight(60)
        btn.setObjectName("navButton")
        return btn

    def _build_customer_page(self) -> QWidget:
        container = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        container.setLayout(page_layout)

        # 1. 상단 헤더 (Hero Section)
        hero = QFrame()
        hero.setObjectName("customerHero")
        hero_layout = QVBoxLayout()
        hero_layout.setContentsMargins(40, 30, 40, 30)
        hero.setLayout(hero_layout)
        
        title = QLabel("비대면 식음료 키오스크")
        title.setObjectName("heroTitle")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("치킨 · 닭꼬치 · 음료를 간편하게 주문하세요")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        page_layout.addWidget(hero)

        # 2. 메인 콘텐츠 (3단 컬럼)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)
        page_layout.addLayout(content_layout)

        # [Left Column] 카테고리 사이드바
        category_frame = QFrame()
        category_frame.setObjectName("categoryCard")
        category_frame.setFixedWidth(220)
        category_layout = QVBoxLayout()
        category_layout.setContentsMargins(16, 24, 16, 24)
        category_layout.setSpacing(10)
        category_frame.setLayout(category_layout)
        
        cat_title = QLabel("메뉴 카테고리")
        cat_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #ef6c00; margin-bottom: 10px;")
        category_layout.addWidget(cat_title)
        
        self.category_layout = QVBoxLayout()
        self.category_layout.setSpacing(12)
        category_layout.addLayout(self.category_layout)
        category_layout.addStretch(1)

        content_layout.addWidget(category_frame)

        # [Middle Column] 메뉴 그리드
        items_frame = QFrame()
        items_frame.setObjectName("itemsCard")
        items_layout_container = QVBoxLayout()
        items_layout_container.setContentsMargins(10, 0, 10, 0)
        items_frame.setLayout(items_layout_container)
        
        menu_header = QLabel("메뉴 선택")
        menu_header.setStyleSheet("font-size: 22px; font-weight: 700; margin-bottom: 10px;")
        items_layout_container.addWidget(menu_header)
        
        self.items_layout = QGridLayout()
        self.items_layout.setSpacing(20)
        items_layout_container.addLayout(self.items_layout)
        items_layout_container.addStretch(1)

        content_layout.addWidget(items_frame, 1) # Stretch factor 1 to take available space

        # [Right Column] 주문 요약 및 액션
        summary_frame = QFrame()
        summary_frame.setObjectName("summaryCard")
        summary_frame.setFixedWidth(350)
        summary_layout = QVBoxLayout()
        summary_layout.setContentsMargins(24, 24, 24, 24)
        summary_layout.setSpacing(16)
        summary_frame.setLayout(summary_layout)
        
        summary_header = QLabel("주문 요약")
        summary_header.setStyleSheet("font-size: 20px; font-weight: 700;")
        summary_layout.addWidget(summary_header)
        
        # 장바구니 리스트
        list_header = QLabel("항목  |  수량  |  가격")
        list_header.setStyleSheet("color: #888; font-size: 14px;")
        summary_layout.addWidget(list_header)

        self.order_list = QListWidget()
        self.order_list.setFocusPolicy(Qt.NoFocus)
        summary_layout.addWidget(self.order_list, 1)

        # 음료 안내 메시지 (가변)
        self.drink_prompt_label = QLabel("")
        self.drink_prompt_label.setWordWrap(True)
        self.drink_prompt_label.setStyleSheet("font-size: 15px; color: #d84315; font-weight: 600;")
        self.drink_prompt_label.hide()
        summary_layout.addWidget(self.drink_prompt_label)

        self.skip_drink_button = QPushButton("음료 건너뛰기")
        self.skip_drink_button.setObjectName("subActionButton")
        self.skip_drink_button.hide()
        self.skip_drink_button.clicked.connect(self._skip_drink_flow)
        summary_layout.addWidget(self.skip_drink_button)

        # 총액 표시 영역
        self.total_label = QLabel("총액: ₩0")
        self.total_label.setObjectName("totalLabel")
        self.total_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.total_label)

        # 보조 버튼 (삭제 / 초기화)
        sub_btn_layout = QHBoxLayout()
        sub_btn_layout.setSpacing(10)
        
        self.remove_button = QPushButton("선택 삭제")
        self.remove_button.setObjectName("subActionButton")
        
        self.clear_button = QPushButton("초기화")
        self.clear_button.setObjectName("subActionButton")
        
        sub_btn_layout.addWidget(self.remove_button, 1)
        sub_btn_layout.addWidget(self.clear_button, 1)
        summary_layout.addLayout(sub_btn_layout)

        # 주문 전송 버튼 (강조)
        self.send_button = QPushButton("주문 전송")
        self.send_button.setObjectName("sendButton")
        self.send_button.setCursor(Qt.PointingHandCursor)
        summary_layout.addWidget(self.send_button)
        
        # 로봇 상태 표시줄
        self.status_label = QLabel("로봇 상태: 대기 중...")
        self.status_label.setObjectName("statusLabel")
        summary_layout.addWidget(self.status_label)

        content_layout.addWidget(summary_frame)

        self._reset_workflow_display()
        return container

    def _build_developer_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        container.setLayout(layout)

        # 상단 헤더
        header = QFrame()
        header.setObjectName("developerHero")
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(30, 24, 30, 24)
        header.setLayout(header_layout)
        
        title = QLabel("두산 M1013 제어 패널")
        title.setStyleSheet("font-size: 32px; font-weight: 800; color: #1f1b16;")
        subtitle = QLabel("시나리오 실행 · OctoMap 경로 생성 · 상태 모니터링을 한 화면에서 확인하세요.")
        subtitle.setStyleSheet("font-size: 16px; color: #5d4037; margin-top: 4px;")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        # 메인 바디 (좌: 제어 / 우: 정보)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)
        body_layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(body_layout)

        # [Left Panel] 스크롤 가능한 시나리오 제어 영역
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        left_panel = QWidget()
        left_panel.setStyleSheet("background: transparent;")
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(0, 0, 10, 0) # 스크롤바 여백
        left_panel_layout.setSpacing(16)
        left_panel.setLayout(left_panel_layout)

        # 섹션 1: 시나리오 실행 타이틀
        section_label = QLabel("시나리오 및 설정")
        section_label.setObjectName("devSectionTitle")
        left_panel_layout.addWidget(section_label)

        # 카드 1: 기본 트레이 목적지 설정
        routing_card = QFrame()
        routing_card.setObjectName("scenarioCard")
        routing_layout = QVBoxLayout()
        routing_layout.setContentsMargins(20, 20, 20, 20)
        routing_card.setLayout(routing_layout)
        
        routing_title = QLabel("기본 트레이 목적지")
        routing_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #333;")
        routing_desc = QLabel("고객 주문 자동 실행 시 적용될 트레이별 목적지입니다.")
        routing_desc.setWordWrap(True)
        routing_desc.setStyleSheet("font-size: 13px; color: #666; margin-bottom: 10px;")
        routing_layout.addWidget(routing_title)
        routing_layout.addWidget(routing_desc)

        for tray_id, tray_label in TRAY_CHOICES:
            row = QHBoxLayout()
            lbl = QLabel(tray_label)
            lbl.setStyleSheet("font-size: 15px; font-weight: 500;")
            combo = QComboBox()
            combo.addItems(DEFAULT_DESTINATIONS)
            combo.setCurrentText(self.tray_destination_defaults.get(tray_id, DEFAULT_DESTINATIONS[0]))
            combo.currentTextChanged.connect(
                lambda value, t=tray_id: self._update_tray_default_destination(t, value)
            )
            combo.setStyleSheet("font-size: 14px; padding: 4px 8px;")
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(combo, 0)
            routing_layout.addLayout(row)
        left_panel_layout.addWidget(routing_card)

        # 카드 2: 자동 시퀀스 순서
        sequence_card = QFrame()
        sequence_card.setObjectName("scenarioCard")
        sequence_layout = QVBoxLayout()
        sequence_layout.setContentsMargins(20, 20, 20, 20)
        sequence_card.setLayout(sequence_layout)
        
        seq_title = QLabel("자동 시퀀스 순서")
        seq_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #333;")
        seq_desc = QLabel("주문 처리 시 실행될 노드 순서를 조정합니다.")
        seq_desc.setStyleSheet("font-size: 13px; color: #666; margin-bottom: 10px;")
        sequence_layout.addWidget(seq_title)
        sequence_layout.addWidget(seq_desc)

        catalog_label = QLabel("▼ 전체 노드 목록 (참고용)")
        catalog_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #888; margin-top: 6px;")
        sequence_layout.addWidget(catalog_label)
        
        self.workflow_catalog_list = QListWidget()
        self.workflow_catalog_list.setStyleSheet("font-size: 13px; background: #f9f9f9; border: 1px solid #eee; border-radius: 8px; color: #777;")
        self.workflow_catalog_list.setFixedHeight(100)
        self.workflow_catalog_list.setSelectionMode(QListWidget.NoSelection)
        self.workflow_catalog_list.setFocusPolicy(Qt.NoFocus)
        sequence_layout.addWidget(self.workflow_catalog_list)
        
        seq_list_label = QLabel("▼ 현재 실행 순서 (드래그 앤 드롭 불가, 버튼 사용)")
        seq_list_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #333; margin-top: 12px;")
        sequence_layout.addWidget(seq_list_label)

        self.sequence_list = QListWidget()
        self.sequence_list.setSelectionMode(QListWidget.SingleSelection)
        self.sequence_list.setStyleSheet("font-size: 14px; border: 1px solid #d0d0d0;")
        self.sequence_list.setFixedHeight(180)
        sequence_layout.addWidget(self.sequence_list)
        
        btn_row = QHBoxLayout()
        up_btn = QPushButton("▲ 위로")
        down_btn = QPushButton("▼ 아래로")
        reset_btn = QPushButton("↺ 기본값 복원")
        for btn in (up_btn, down_btn, reset_btn):
            btn.setMinimumHeight(36)
            btn.setStyleSheet("""
                QPushButton { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 8px; font-size: 13px; }
                QPushButton:hover { background-color: #e0e0e0; }
            """)
            btn_row.addWidget(btn)
        sequence_layout.addLayout(btn_row)
        left_panel_layout.addWidget(sequence_card)
        
        self._populate_catalog_list()
        self._populate_sequence_list(self.workflow_custom_order, log=False)
        up_btn.clicked.connect(lambda: self._adjust_sequence_order(-1))
        down_btn.clicked.connect(lambda: self._adjust_sequence_order(1))
        reset_btn.clicked.connect(self._reset_sequence_order)

        # 개별 시나리오 카드 생성
        scenarios_label = QLabel("개별 노드 수동 제어")
        scenarios_label.setObjectName("devSectionTitle")
        scenarios_label.setStyleSheet("margin-top: 20px;")
        left_panel_layout.addWidget(scenarios_label)

        for scenario in SCENARIOS:
            card = QFrame()
            card.setObjectName("scenarioCard")
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(20, 16, 20, 16)
            card.setLayout(card_layout)
            
            # 헤더: 타이틀 + 활성화 체크박스
            title_row = QHBoxLayout()
            title_lbl = QLabel(scenario.label)
            title_lbl.setStyleSheet("font-size: 17px; font-weight: 700; color: #1f1b16;")
            
            enable_box = QCheckBox("활성화")
            enable_box.setChecked(True)
            enable_box.setStyleSheet("font-size: 14px;")
            
            title_row.addWidget(title_lbl)
            title_row.addStretch(1)
            title_row.addWidget(enable_box)
            card_layout.addLayout(title_row)

            # 설명 및 힌트
            desc = QLabel(scenario.description or "")
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #555; font-size: 14px; margin-bottom: 6px;")
            card_layout.addWidget(desc)

            if scenario.action_hint:
                hint = QLabel(f"💡 {scenario.action_hint}")
                hint.setWordWrap(True)
                hint.setStyleSheet("font-size: 13px; color: #2e7d32; background: #e8f5e9; border-radius: 6px; padding: 4px 8px;")
                card_layout.addWidget(hint)

            # 옵션 선택기 (목적지/트레이)
            opts_layout = QHBoxLayout()
            dest_selector = None
            if scenario.needs_destination:
                opts_layout.addWidget(QLabel("목적지:"))
                dest_selector = QComboBox()
                dest_selector.addItems(DEFAULT_DESTINATIONS)
                opts_layout.addWidget(dest_selector)
            
            tray_selector = None
            if scenario.tray_selectable:
                opts_layout.addWidget(QLabel("트레이:"))
                tray_selector = QComboBox()
                for value, label in TRAY_CHOICES:
                    tray_selector.addItem(label, value)
                default_tray = NODE_COMMANDS.get(scenario.id, {}).get("default_tray")
                if default_tray is not None:
                    idx = tray_selector.findData(default_tray)
                    if idx >= 0:
                        tray_selector.setCurrentIndex(idx)
                opts_layout.addWidget(tray_selector)

            if scenario.needs_destination or scenario.tray_selectable:
                opts_layout.addStretch(1)
                card_layout.addLayout(opts_layout)

            # 실행 버튼 및 상태바
            control_row = QHBoxLayout()
            run_btn = QPushButton("실행")
            run_btn.setCursor(Qt.PointingHandCursor)
            run_btn.setStyleSheet("""
                QPushButton { background-color: #ff9800; color: white; border-radius: 8px; font-weight: bold; padding: 8px 16px; }
                QPushButton:hover { background-color: #f57c00; }
                QPushButton:disabled { background-color: #ccc; }
            """)
            
            stop_btn = QPushButton("중지")
            stop_btn.setEnabled(False)
            stop_btn.setStyleSheet("""
                QPushButton { background-color: #e0e0e0; color: #333; border-radius: 8px; padding: 8px 16px; }
            """)

            control_row.addWidget(run_btn)
            control_row.addWidget(stop_btn)

            status_label = QLabel("대기 중")
            status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_label.setStyleSheet("font-size: 14px; color: #666; margin-right: 8px;")
            control_row.addStretch(1)
            control_row.addWidget(status_label)
            card_layout.addLayout(control_row)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setFixedHeight(6)
            card_layout.addWidget(progress)

            left_panel_layout.addWidget(card)

            # 위젯 등록
            self.scenario_widgets[scenario.id] = {
                "enable": enable_box,
                "run": run_btn,
                "stop": stop_btn,
                "status": status_label,
                "progress": progress,
            }
            if dest_selector:
                self.scenario_widgets[scenario.id]["destination"] = dest_selector
            if tray_selector:
                self.scenario_widgets[scenario.id]["tray"] = tray_selector

            enable_box.toggled.connect(
                lambda state, scenario_id=scenario.id: self._handle_scenario_toggle(scenario_id, state)
            )
            run_btn.clicked.connect(
                lambda _, scenario_id=scenario.id: self._handle_scenario_action(scenario_id, "start")
            )
            stop_btn.clicked.connect(
                lambda _, scenario_id=scenario.id: self._handle_scenario_action(scenario_id, "stop")
            )

        left_panel_layout.addStretch(1)
        scroll_area.setWidget(left_panel)
        body_layout.addWidget(scroll_area, 2) # Left Panel 비율 2

        # [Right Panel] 상태 및 로그 정보
        right_panel = QFrame()
        right_panel.setObjectName("devInfoCard")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(20)
        right_panel.setLayout(right_layout)

        info_title = QLabel("시스템 모니터링")
        info_title.setObjectName("devSectionTitle")
        right_layout.addWidget(info_title)

        # 로봇 상태 그룹
        status_box = QGroupBox("두산 M1013 상태")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(8)
        status_box.setLayout(status_layout)

        self.dev_status_summary = {
            "m1013": QLabel("● 두산 M1013: 대기 중"),
            "latest": QLabel("● 최근 메시지: 수신 없음"),
        }
        for lbl in self.dev_status_summary.values():
            lbl.setStyleSheet("font-size: 15px; color: #333;")
            status_layout.addWidget(lbl)
        right_layout.addWidget(status_box)

        # OctoMap 그룹
        octomap_box = QGroupBox("경로 생성 (OctoMap)")
        octomap_layout = QVBoxLayout()
        octomap_msg = QLabel(
            "Realsense D435i 센서 데이터를 기반으로 환경을 스캔하고 경로를 생성합니다."
        )
        octomap_msg.setWordWrap(True)
        octomap_msg.setStyleSheet("font-size: 14px; color: #555; line-height: 1.4;")
        octomap_layout.addWidget(octomap_msg)
        octomap_box.setLayout(octomap_layout)
        right_layout.addWidget(octomap_box)

        # 로그 그룹
        log_box = QGroupBox("실시간 시스템 로그")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(10, 25, 10, 10) # 타이틀 공간 확보
        self.dev_log = QPlainTextEdit()
        self.dev_log.setReadOnly(True)
        self.dev_log.setPlaceholderText("여기에 시스템 및 노드 실행 로그가 표시됩니다...")
        log_layout.addWidget(self.dev_log)
        log_box.setLayout(log_layout)
        
        right_layout.addWidget(log_box, 1) # 로그창이 남은 공간 차지
        body_layout.addWidget(right_panel, 1) # Right Panel 비율 1

        return container

    # ------------------------------------------------------------------ #
    # 신호 연결 및 상태 업데이트
    # ------------------------------------------------------------------ #
    def _connect_signals(self) -> None:
        self.nav_buttons["customer"].clicked.connect(lambda: self._set_active_view("customer"))
        self.nav_buttons["developer"].clicked.connect(lambda: self._set_active_view("developer"))
        self.send_button.clicked.connect(self._handle_send)
        self.clear_button.clicked.connect(self._handle_clear)
        self.remove_button.clicked.connect(self._handle_remove_selected)
        self.ros_bridge.status_received.connect(self._update_status)

    def _set_active_view(self, key: str) -> None:
        if key == "customer":
            self.view_stack.setCurrentIndex(0)
        else:
            self.view_stack.setCurrentIndex(1)
        for name, btn in self.nav_buttons.items():
            self._apply_nav_style(btn, name == key)

    def _apply_nav_style(self, btn: QPushButton, active: bool) -> None:
        btn.setChecked(active)

    def _render_categories(self) -> None:
        while self.category_layout.count():
            child = self.category_layout.takeAt(0)
            if widget := child.widget():
                widget.deleteLater()

        for category in self.menu_map:
            btn = QPushButton(category)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setObjectName("categoryButton")
            if category == self.current_category:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, c=category: self._render_items(c))
            self.category_layout.addWidget(btn)

    def _render_items(self, category: str) -> None:
        self.current_category = category
        while self.items_layout.count():
            child = self.items_layout.takeAt(0)
            if widget := child.widget():
                widget.deleteLater()

        items = self.menu_map.get(category, [])
        columns = 1 if category == self.base_category_name else 3
        for idx, item in enumerate(items):
            # 카드형 버튼 컨테이너
            btn = QPushButton()
            btn.setMinimumSize(200, 240)
            btn.setObjectName("menuButton")
            btn.setCursor(Qt.PointingHandCursor)

            # 내부 레이아웃 (이미지 위, 텍스트 아래)
            layout = QVBoxLayout(btn)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)
            
            # 이미지 영역
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background-color: transparent;")
            img_label.setFixedSize(*self._menu_image_size)
            if pixmap := self._load_menu_pixmap(item.image):
                img_label.setPixmap(pixmap)
            else:
                # 이미지가 없을 때 플레이스홀더
                img_label.setText(item.name[:2])
                img_label.setStyleSheet(
                    "background-color: #f0f0f0; color: #ccc; font-size: 30px; font-weight: bold; border-radius: 10px;"
                )
            
            layout.addWidget(img_label)

            # 텍스트 영역
            name_label = QLabel(item.name)
            name_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; border: none; background: transparent;")
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name_label.setWordWrap(True)
            
            price_label = QLabel(f"₩{item.price:,}")
            price_label.setStyleSheet("font-size: 16px; font-weight: 500; color: #d84315; border: none; background: transparent;")
            price_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            layout.addWidget(name_label)
            layout.addWidget(price_label)
            layout.addStretch(1)

            btn.clicked.connect(lambda _, m=item: self._add_item(m))
            
            row, col = divmod(idx, columns)
            self.items_layout.addWidget(btn, row, col)

    def _load_menu_pixmap(self, image_name: Optional[str]) -> Optional[QPixmap]:
        if not image_name:
            return None
        base_dir = Path(__file__).resolve().parent
        image_path = base_dir / image_name
        if not image_path.exists():
            return None
        width, height = self._menu_image_size
        cache_key = f"{image_path.resolve()}::{width}x{height}"
        cached = self._pixmap_cache.get(cache_key)
        if cached:
            return cached
        pixmap = QPixmap(str(image_path)).scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._pixmap_cache[cache_key] = pixmap
        return pixmap

    def _add_item(self, item: MenuItem) -> None:
        if item.payload.get("menu") == "drink":
            current = self._count_drinks()
            limit = self._drink_limit()
            if current >= limit:
                QMessageBox.information(
                    self,
                    "안내",
                    f"현재 구성에서는 음료를 최대 {limit}잔까지 선택할 수 있습니다.",
                )
                return
        self.order.add_item(item)
        payload_menu = item.payload.get("menu")
        if payload_menu == "drink":
            target_tray = self.pending_drink_tray or self._default_drink_tray()
            self.drink_tray_overrides = [target_tray]
            self.pending_drink_tray = None
        else:
            tray_for_item = self._tray_for_item(item)
            if tray_for_item:
                self.pending_drink_tray = tray_for_item
        self._refresh_summary()
        category = self.item_categories.get(item.name)
        if category == self.base_category_name:
            self._update_menu_image(item)
            if self.drink_item and not self.order.contains(self.drink_item.name):
                self._start_drink_flow()
        elif self.drink_item and category == self.drink_category_name:
            self._append_dev_log("[ORDER] 고객이 음료를 추가했습니다.")
            self._end_drink_flow()

    def _refresh_summary(self) -> None:
        self.order_list.clear()
        if not self.order.lines:
            placeholder = QListWidgetItem("선택된 항목이 없습니다.")
            placeholder.setFlags(Qt.NoItemFlags)
            self.order_list.addItem(placeholder)
        else:
            for line in self.order.lines:
                qty = f" x{line.quantity}" if line.quantity > 1 else ""
                text = f"{line.item.name}{qty} - ₩{line.total:,}"
                self.order_list.addItem(QListWidgetItem(text))
        self.total_label.setText(f"총액: ₩{self.order.total():,}")

    # ------------------------------------------------------------------ #
    # 고객 화면 이벤트
    # ------------------------------------------------------------------ #
    def _handle_send(self) -> None:
        if not self.order:
            QMessageBox.information(self, "안내", "선택된 항목이 없습니다.")
            return
        if self._workflow_active():
            QMessageBox.information(self, "안내", "이전 주문 실행이 아직 진행 중입니다.")
            return
        payload = self.order.serialize()
        success = self.ros_bridge.publish(payload)
        if success:
            QMessageBox.information(self, "전송 완료", "ROS 토픽으로 전송했습니다.")
            self._append_dev_log("[ORDER] 고객 주문을 전송했습니다.")
            self._execute_order_workflow(payload)
            self._reset_order()
        else:
            QMessageBox.warning(self, "전송 실패", "ROS 토픽 발행에 실패했습니다.")

    def _handle_clear(self) -> None:
        self._reset_order()

    def _handle_remove_selected(self) -> None:
        if not self.order.lines:
            return
        row = self.order_list.currentRow()
        if row < 0:
            return
        line = self.order.lines[row]
        self.order.remove_index(row)
        self._refresh_summary()
        self._enforce_drink_limit()

    def _reset_order(self) -> None:
        self.order.clear()
        self._end_drink_flow()
        self._reset_menu_image()
        self.pending_drink_tray = None
        self.drink_tray_overrides = []
        if self.base_category_name in self.menu_map:
            self._render_items(self.base_category_name)
        self._refresh_summary()

    def _start_drink_flow(self) -> None:
        if not self.drink_item or self.awaiting_drink:
            return
        self.awaiting_drink = True
        limit = self._drink_limit()
        self.drink_prompt_label.setText(f"음료를 함께 선택해 주세요 (최대 {limit}잔).")
        self.drink_prompt_label.show()
        self.skip_drink_button.show()
        self._append_dev_log("[ORDER] 음료 선택 화면으로 전환했습니다.")
        if self.drink_category_name and self.drink_category_name in self.menu_map:
            self._render_items(self.drink_category_name)

    def _end_drink_flow(self) -> None:
        if not self.awaiting_drink:
            self.drink_prompt_label.hide()
            self.skip_drink_button.hide()
            return
        self.awaiting_drink = False
        self.drink_prompt_label.hide()
        self.skip_drink_button.hide()
        self._append_dev_log("[ORDER] 음료 선택 단계가 종료되었습니다.")
        if self.base_category_name in self.menu_map:
            self._render_items(self.base_category_name)

    def _skip_drink_flow(self) -> None:
        if not self.awaiting_drink:
            return
        self._append_dev_log("[ORDER] 고객이 음료를 건너뛰었습니다.")
        self._end_drink_flow()
        self._enforce_drink_limit()

    def _count_drinks(self) -> int:
        return sum(
            line.quantity
            for line in self.order.lines
            if line.item.payload.get("menu") == "drink"
        )

    def _drink_limit(self) -> int:
        # 태블릿/현장 운영 요구에 따라 음료는 항상 최대 1잔만 허용.
        return 1

    def _tray_for_item(self, item: MenuItem) -> Optional[str]:
        payload_menu = (item.payload or {}).get("menu")
        if payload_menu == "chicken_fry":
            return NODE_COMMANDS["chicken_cook_to_tray1"]["default_tray"]
        if payload_menu == "skewer_roast":
            return NODE_COMMANDS["skewer_cook_to_tray2"]["default_tray"]
        return None

    def _default_drink_tray(self) -> str:
        if self.order.contains("닭꼬치 구이"):
            return NODE_COMMANDS["skewer_cook_to_tray2"]["default_tray"]
        return NODE_COMMANDS["chicken_cook_to_tray1"]["default_tray"]

    def _node_suffix_for_tray(self, tray: str) -> str:
        return "1" if tray == "tray_a" else "2"

    def _cook_node_id(self, food: str, tray: str) -> str:
        suffix = self._node_suffix_for_tray(tray)
        base = "chicken_cook" if food == "chicken" else "skewer_cook"
        return f"{base}_to_tray{suffix}"

    def _cup_node_id(self, tray: str) -> str:
        suffix = self._node_suffix_for_tray(tray)
        return f"cup_transfer_to_tray{suffix}"

    def _tray_transfer_node_id(self, destination: str) -> str:
        suffix = self._destination_suffix(destination)
        return f"tray_transfer_to_destination{suffix}"

    def _destination_suffix(self, destination: Optional[str]) -> str:
        dest = destination or DEFAULT_DESTINATIONS[0]
        try:
            idx = DEFAULT_DESTINATIONS.index(dest)
        except ValueError:
            idx = 0
        return "1" if idx == 0 else "2"

    def _remove_one_drink(self) -> bool:
        for idx, line in enumerate(self.order.lines):
            if line.item.payload.get("menu") == "drink":
                if line.quantity > 1:
                    line.quantity -= 1
                else:
                    self.order.remove_index(idx)
                return True
        return False

    def _enforce_drink_limit(self) -> None:
        limit = self._drink_limit()
        current = self._count_drinks()
        changed = False
        while current > limit and self._remove_one_drink():
            current -= 1
            changed = True
        if changed:
            QMessageBox.information(self, "안내", "음료 수량이 제한에 맞게 조정되었습니다.")
            self._refresh_summary()

    # ------------------------------------------------------------------ #
    # 워크플로우 진행 관리
    # ------------------------------------------------------------------ #
    def _workflow_active(self) -> bool:
        return bool(self.workflow_thread and self.workflow_thread.isRunning())

    def _format_step_label(
        self,
        node_id: str,
        tray: Optional[str],
        destination: Optional[str],
    ) -> str:
        label = NODE_COMMANDS.get(node_id, {}).get("label", node_id)
        suffix: List[str] = []
        if tray:
            suffix.append(tray.upper())
        if destination:
            suffix.append(destination)
        if suffix:
            return f"{label} ({', '.join(suffix)})"
        return label

    def _reset_workflow_display(self) -> None:
        if getattr(self, "workflow_progress_list", None):
            self.workflow_progress_list.clear()
            self.workflow_progress_list.addItem("대기 중입니다.")
        if getattr(self, "workflow_status_label", None):
            self.workflow_status_label.setText("대기 중")
        if getattr(self, "workflow_progress_bar", None):
            self.workflow_progress_bar.setValue(0)

    def _init_workflow_progress(self) -> None:
        if not self.workflow_progress_list or not self.workflow_progress_bar or not self.workflow_status_label:
            return
        self.workflow_progress_list.clear()
        if not self.workflow_steps_meta:
            self.workflow_progress_list.addItem("실행할 단계가 없습니다.")
            self.workflow_progress_bar.setValue(0)
            self.workflow_status_label.setText("대기 중")
            return
        for meta in self.workflow_steps_meta:
            self.workflow_progress_list.addItem(QListWidgetItem(f"[대기] {meta['label']}"))
        self.workflow_progress_bar.setValue(0)
        self.workflow_status_label.setText("실행 준비 중...")

    def _start_workflow_thread(self, steps: List[Tuple[str, Optional[str], Optional[str]]]) -> None:
        if not steps:
            return
        if self._any_thread_running():
            QMessageBox.information(self, "안내", "다른 노드 실행이 진행 중입니다. 잠시 후 다시 시도하세요.")
            return
        thread = WorkflowThread(steps=steps, runner=self.node_runner)
        thread.step_started.connect(self._on_workflow_step_started)
        thread.step_finished.connect(self._on_workflow_step_finished)
        thread.sequence_finished.connect(self._on_workflow_finished)
        thread.finished.connect(thread.deleteLater)
        self.workflow_thread = thread
        self.send_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self._append_dev_log(f"[FLOW] 총 {len(steps)}단계 실행을 시작합니다.")
        thread.start()

    def _on_workflow_step_started(self, index: int, info: Dict[str, Any]) -> None:
        if not (0 <= index < len(self.workflow_steps_meta)):
            return
        meta = self.workflow_steps_meta[index]
        if self.workflow_progress_list and (item := self.workflow_progress_list.item(index)):
            item.setText(f"[실행 중] {meta['label']}")
        if self.workflow_progress_bar:
            percent = int((index / max(1, len(self.workflow_steps_meta))) * 100)
            self.workflow_progress_bar.setValue(percent)
        if self.workflow_status_label:
            self.workflow_status_label.setText(f"{meta['label']} 실행 중...")
        self._append_dev_log(f"[FLOW] {meta['label']} 시작")

    def _on_workflow_step_finished(self, index: int, info: Dict[str, Any]) -> None:
        if not (0 <= index < len(self.workflow_steps_meta)):
            return
        meta = self.workflow_steps_meta[index]
        success = info.get("result") == 0
        status = "완료" if success else "실패"
        if self.workflow_progress_list and (item := self.workflow_progress_list.item(index)):
            item.setText(f"[{status}] {meta['label']}")
        if self.workflow_progress_bar:
            percent = int(((index + 1) / max(1, len(self.workflow_steps_meta))) * 100)
            self.workflow_progress_bar.setValue(percent)
        if not success and self.workflow_status_label:
            self.workflow_status_label.setText(f"{meta['label']} 실패")
            self._append_dev_log(f"[FLOW] {meta['label']} 단계 실패")

    def _on_workflow_finished(self, success: bool) -> None:
        if success:
            if self.workflow_progress_bar:
                self.workflow_progress_bar.setValue(100)
            if self.workflow_status_label:
                self.workflow_status_label.setText("전체 시퀀스 완료")
            self._append_dev_log("[FLOW] 고객 주문 시퀀스 완료")
        else:
            if self.workflow_status_label:
                self.workflow_status_label.setText("시퀀스가 중단되었습니다.")
            self._append_dev_log("[FLOW] 고객 주문 시퀀스가 중단/실패했습니다.")
        self.send_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        if self.workflow_thread:
            self.workflow_thread = None
        self.workflow_steps_meta = []

    def _is_scenario_running(self, scenario_id: str) -> bool:
        thread = self.scenario_threads.get(scenario_id)
        return bool(thread and thread.isRunning())

    def _start_scenario_thread(
        self,
        scenario_id: str,
        *,
        tray: Optional[str],
        destination: Optional[str],
    ) -> None:
        if self._any_thread_running():
            QMessageBox.information(self, "안내", "다른 노드 실행이 진행 중입니다. 잠시 후 다시 시도하세요.")
            return
        thread = WorkflowThread(steps=[(scenario_id, tray, destination)], runner=self.node_runner)
        thread.step_started.connect(lambda idx, info, sid=scenario_id: self._on_scenario_step_started(sid))
        thread.step_finished.connect(
            lambda idx, info, sid=scenario_id: self._on_scenario_step_finished(sid, info)
        )
        thread.sequence_finished.connect(
            lambda success, sid=scenario_id: self._on_scenario_sequence_finished(sid, success)
        )
        thread.finished.connect(thread.deleteLater)
        self.scenario_threads[scenario_id] = thread
        widgets = self.scenario_widgets.get(scenario_id, {})
        if run_btn := widgets.get("run"):
            run_btn.setEnabled(False)
        thread.start()

    def _on_scenario_step_started(self, scenario_id: str) -> None:
        widgets = self.scenario_widgets.get(scenario_id, {})
        if status_lbl := widgets.get("status"):
            status_lbl.setText("실행 중...")
        if progress := widgets.get("progress"):
            progress.setRange(0, 0)
        label = self.scenario_lookup[scenario_id].label
        self._append_dev_log(f"[NODE] {label} 실행 시작")

    def _on_scenario_step_finished(self, scenario_id: str, info: Dict[str, Any]) -> None:
        widgets = self.scenario_widgets.get(scenario_id, {})
        success = info.get("result") == 0
        label = self.scenario_lookup[scenario_id].label
        if progress := widgets.get("progress"):
            progress.setRange(0, 100)
            progress.setValue(100 if success else 0)
        if status_lbl := widgets.get("status"):
            status_lbl.setText("실행 완료" if success else "실패")
        if success:
            if robot_label := self.dev_status_summary.get("m1013"):
                robot_label.setText(f"두산 M1013: {label} 실행 완료")
            self._append_dev_log(f"[NODE] {label} 실행 성공")
        else:
            QMessageBox.warning(self, "실행 실패", f"{label} 노드 실행에 실패했습니다.")
            self._append_dev_log(f"[NODE] {label} 실행 실패")

    def _on_scenario_sequence_finished(self, scenario_id: str, success: bool) -> None:
        self.scenario_threads.pop(scenario_id, None)
        widgets = self.scenario_widgets.get(scenario_id, {})
        if run_btn := widgets.get("run"):
            run_btn.setEnabled(True)

    def _update_tray_default_destination(self, tray: str, destination: str) -> None:
        self.tray_destination_defaults[tray] = destination
        self._append_dev_log(f"[CONFIG] {tray.upper()} 기본 목적지를 {destination}로 설정했습니다.")

    def _populate_catalog_list(self) -> None:
        if not self.workflow_catalog_list:
            return
        self.workflow_catalog_list.clear()
        for node_id in ORDERABLE_NODE_IDS:
            scenario = self.scenario_lookup.get(node_id)
            label = (
                scenario.label
                if scenario
                else NODE_COMMANDS.get(node_id, {}).get("label", node_id)
            )
            desc = scenario.description or ""
            text = f"{label} — {desc}"
            item = QListWidgetItem(text)
            item.setFlags(Qt.NoItemFlags)
            self.workflow_catalog_list.addItem(item)

    def _populate_sequence_list(self, order: List[str], *, log: bool = True) -> None:
        if not self.sequence_list:
            return
        self.sequence_list.clear()
        for idx, node_id in enumerate(order, start=1):
            scenario = self.scenario_lookup.get(node_id)
            label = (
                scenario.label
                if scenario
                else NODE_COMMANDS.get(node_id, {}).get("label", node_id)
            )
            display = f"{idx}. {label}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, node_id)
            self.sequence_list.addItem(item)
        if self.sequence_list.count():
            self.sequence_list.setCurrentRow(0)
        if log:
            labels = ", ".join(
                self.scenario_lookup.get(node_id).label if self.scenario_lookup.get(node_id) else node_id
                for node_id in order
            )
            self._append_dev_log(f"[CONFIG] 자동 시퀀스 순서를 적용했습니다: {labels}")

    def _adjust_sequence_order(self, delta: int) -> None:
        if not self.sequence_list or self.sequence_list.count() == 0:
            return
        row = self.sequence_list.currentRow()
        if row < 0:
            row = 0
        target = row + delta
        if target < 0 or target >= self.sequence_list.count():
            return
        item = self.sequence_list.takeItem(row)
        self.sequence_list.insertItem(target, item)
        self.sequence_list.setCurrentRow(target)
        self._sync_sequence_order()

    def _reset_sequence_order(self) -> None:
        self.workflow_custom_order = list(DEFAULT_WORKFLOW_ORDER)
        self._populate_sequence_list(self.workflow_custom_order, log=False)
        self._append_dev_log("[CONFIG] 자동 시퀀스를 기본 순서로 복원했습니다.")

    def _sync_sequence_order(self) -> None:
        if not self.sequence_list:
            return
        order: List[str] = []
        for idx in range(self.sequence_list.count()):
            node_id = self.sequence_list.item(idx).data(Qt.UserRole)
            if node_id:
                order.append(node_id)
        if not order:
            return
        self.workflow_custom_order = order
        self._populate_sequence_list(order, log=False)
        labels = ", ".join(
            (
                self.scenario_lookup.get(node_id).label
                if self.scenario_lookup.get(node_id)
                else NODE_COMMANDS.get(node_id, {}).get("label", node_id)
            )
            for node_id in order
        )
        self._append_dev_log(f"[CONFIG] 자동 시퀀스 순서를 변경했습니다: {labels}")

    def _compute_tray_destinations(self, tray_sequence: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        fallback_cycle = list(DEFAULT_DESTINATIONS)
        for tray in tray_sequence:
            preferred = self.tray_destination_defaults.get(tray)
            if preferred:
                mapping[tray] = preferred
                continue
            if not fallback_cycle:
                fallback_cycle = list(DEFAULT_DESTINATIONS)
            mapping[tray] = fallback_cycle.pop(0)
        return mapping

    def _apply_custom_order(
        self, steps: List[Tuple[str, Optional[str], Optional[str]]]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        if not self.workflow_custom_order:
            return steps
        order_index = {node_id: idx for idx, node_id in enumerate(self.workflow_custom_order)}
        max_idx = len(order_index)
        enumerated = list(enumerate(steps))

        def sort_key(pair: Tuple[int, Tuple[str, Optional[str], Optional[str]]]) -> Tuple[int, int]:
            node_id = pair[1][0]
            return (order_index.get(node_id, max_idx), pair[0])

        enumerated.sort(key=sort_key)
        return [step for _, step in enumerated]

    def _any_thread_running(self) -> bool:
        if self.workflow_thread and self.workflow_thread.isRunning():
            return True
        return any(thread.isRunning() for thread in self.scenario_threads.values())

    def _execute_order_workflow(self, payload: Dict[str, Any]) -> None:
        items = payload.get("items", [])
        names = {item.get("name") for item in items}
        has_chicken = "치킨 튀김" in names
        has_skewer = "닭꼬치 구이" in names
        drink_count = self._count_drinks()
        drink_override_queue: List[str] = list(self.drink_tray_overrides)
        self.drink_tray_overrides = []
        self.pending_drink_tray = None
        steps: List[Tuple[str, Optional[str], Optional[str]]] = []

        drink_trays: List[str] = []
        if drink_count:
            if drink_override_queue:
                while len(drink_override_queue) < drink_count:
                    drink_override_queue.append(drink_override_queue[-1])
                drink_trays = drink_override_queue[:drink_count]
            else:
                if has_chicken and has_skewer:
                    drink_trays = list(TRAY_IDS)
                elif has_chicken:
                    drink_trays = ["tray_a"]
                elif has_skewer:
                    drink_trays = ["tray_b"]
                else:
                    drink_trays = ["tray_a"]

        tray_sequence: List[str] = []

        def mark_tray(tray: str) -> None:
            if tray not in tray_sequence:
                tray_sequence.append(tray)

        if has_chicken:
            steps.append((self._cook_node_id("chicken", "tray_a"), "tray_a", None))
            mark_tray("tray_a")
        if has_skewer:
            steps.append((self._cook_node_id("skewer", "tray_b"), "tray_b", None))
            mark_tray("tray_b")
        for tray in drink_trays[:drink_count]:
            mark_tray(tray)
        if not tray_sequence:
            mark_tray("tray_a")

        tray_dest_map = self._compute_tray_destinations(tray_sequence)

        for idx in range(min(drink_count, len(drink_trays))):
            tray = drink_trays[idx]
            destination = tray_dest_map.get(tray, DEFAULT_DESTINATIONS[0])
            steps.append((self._cup_node_id(tray), tray, destination))

        for tray in tray_sequence:
            destination = tray_dest_map.get(tray, DEFAULT_DESTINATIONS[0])
            node_id = self._tray_transfer_node_id(destination)
            steps.append((node_id, tray, destination))

        if not steps:
            self._append_dev_log("[FLOW] 실행할 노드가 없어 시퀀스를 생략했습니다.")
            return

        steps = self._apply_custom_order(steps)
        self.workflow_steps_meta = [
            {
                "node_id": node_id,
                "tray": tray,
                "destination": destination,
                "label": self._format_step_label(node_id, tray, destination),
            }
            for node_id, tray, destination in steps
        ]
        self._init_workflow_progress()
        self._start_workflow_thread(steps)

    def _update_menu_image(self, item: MenuItem) -> None:
        if hasattr(self, "menu_image_placeholder") and self.menu_image_placeholder:
            self.menu_image_placeholder.setText(f"{item.name}\n₩{item.price:,}")

    def _reset_menu_image(self) -> None:
        if hasattr(self, "menu_image_placeholder") and self.menu_image_placeholder:
            self.menu_image_placeholder.setText("이미지 영역")

    # ------------------------------------------------------------------ #
    # 개발자 화면 이벤트
    # ------------------------------------------------------------------ #
    def _handle_scenario_toggle(self, scenario_id: str, enabled: bool) -> None:
        self.scenario_states[scenario_id] = enabled
        widgets = self.scenario_widgets.get(scenario_id, {})
        for key in ("run", "stop"):
            if key in widgets:
                widgets[key].setEnabled(enabled)
        status_lbl = widgets.get("status")
        if status_lbl:
            status_lbl.setText("대기 중" if enabled else "비활성화됨")
        self._append_dev_log(
            f"[TOGGLE] {self.scenario_lookup[scenario_id].label} -> {'ON' if enabled else 'OFF'}"
        )

    def _handle_scenario_action(self, scenario_id: str, action: str) -> None:
        if action == "stop":
            QMessageBox.information(
                self,
                "안내",
                "UI에서의 강제 중지는 추후 제공 예정입니다.\n현재는 ROS 터미널에서 노드를 종료해 주세요.",
            )
            return
        if not self.scenario_states.get(scenario_id, False):
            QMessageBox.information(self, "안내", "토글을 활성화한 뒤 실행할 수 있습니다.")
            return
        if self._is_scenario_running(scenario_id):
            QMessageBox.information(self, "안내", "이미 실행 중인 노드가 있습니다.")
            return
        scenario = self.scenario_lookup[scenario_id]
        widgets = self.scenario_widgets.get(scenario_id, {})
        destination = None
        if scenario.needs_destination:
            dest_widget = widgets.get("destination")
            if isinstance(dest_widget, QComboBox):
                destination = dest_widget.currentText()
        tray = NODE_COMMANDS.get(scenario_id, {}).get("default_tray")
        if scenario.tray_selectable:
            tray_widget = widgets.get("tray")
            if isinstance(tray_widget, QComboBox) and tray_widget.currentData():
                tray = tray_widget.currentData()
        self._start_scenario_thread(scenario_id, tray=tray, destination=destination)

    # ------------------------------------------------------------------ #
    # 상태 업데이트/로그
    # ------------------------------------------------------------------ #
    def _update_status(self, text: str) -> None:
        message = text or "수신된 상태가 없습니다."
        self.status_label.setText(message)
        latest_label = self.dev_status_summary.get("latest")
        if latest_label:
            latest_label.setText(f"최근 메시지: {message}")
        robot_label = self.dev_status_summary.get("m1013")
        if robot_label:
            robot_label.setText(f"두산 M1013: {message}")
        self._append_dev_log(f"[STATUS] {message}")

    def _append_dev_log(self, message: str) -> None:
        if not hasattr(self, "dev_log"):
            return
        self.dev_log.appendPlainText(message)
        self.dev_log.verticalScrollBar().setValue(self.dev_log.verticalScrollBar().maximum())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 키오스크 UI")
    parser.add_argument("--menu", type=Path, help="메뉴 JSON 파일 경로")
    parser.add_argument("--order-topic", default="/kiosk/order")
    parser.add_argument("--status-topic", default="/kiosk/status")
    parser.add_argument("--node-name", default="kiosk_ui")
    parser.add_argument("--namespace", default=None, help="ROS 네임스페이스")
    parser.add_argument("--window-title", default="Kiosk UI")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--windowed", action="store_true", help="강제로 창 모드 실행")
    parser.add_argument("--window-width", type=int, help="창 모드일 때 고정 폭(px)")
    parser.add_argument("--window-height", type=int, help="창 모드일 때 고정 높이(px)")
    parser.add_argument("--dry-run", action="store_true", help="ROS 없이 UI만 테스트")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    menu_data = load_menu(args.menu)
    menu_map = build_menu_map(menu_data)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv or ["keyhosk"])

    ros_bridge = RosBridge(
        node_name=args.node_name,
        order_topic=args.order_topic,
        status_topic=args.status_topic,
        namespace=args.namespace,
        dry_run=args.dry_run,
    )
    ros_bridge.start()

    start_fullscreen = False
    if args.windowed:
        start_fullscreen = False
    if args.fullscreen:
        start_fullscreen = True

    window = KioskWindow(
        ros_bridge=ros_bridge,
        menu_map=menu_map,
        app_args=args,
        title=args.window_title,
        lock_window_size=not start_fullscreen,
    )
    window.in_fullscreen_mode = start_fullscreen

    if start_fullscreen:
        window.showFullScreen()
    else:
        window.show()

    return app.exec_()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())