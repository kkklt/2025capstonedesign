import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if not devices:
    print("연결된 RealSense 카메라가 없습니다.")
else:
    print(f"{len(devices)}개의 카메라가 발견되었습니다:")
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  - 이름: {name}, 시리얼 번호: {serial}")
