# coding=utf-8
"""
Tango 스테이지 컨트롤러 테스트 스크립트
- rays-on.exe와의 연결 충돌 테스트용
- 스테이지 직접 제어 테스트용
"""

import ctypes
import sys
import re
from ctypes import *


class TangoController:
    """Tango 스테이지 컨트롤러 래퍼 클래스"""
    
    def __init__(self, dll_path: str = r"./Tango_DLL.dll"):
        self.dll_path = dll_path
        self.dll = None
        self.LSID = c_int()
        self.connected = False
        
    def load_dll(self) -> bool:
        """DLL 로드"""
        try:
            self.dll = cdll.LoadLibrary(self.dll_path)
            print(f"[OK] DLL 로드 성공: {self.dll_path}")
            return True
        except Exception as e:
            print(f"[ERROR] DLL 로드 실패: {e}")
            print("  - Tango_DLL.dll이 현재 디렉토리에 있는지 확인하세요")
            print("  - Python 비트(32/64)와 DLL 비트가 일치하는지 확인하세요")
            return False
    
    def create_session(self) -> bool:
        """세션 ID 생성"""
        if not self.dll:
            print("[ERROR] DLL이 로드되지 않았습니다")
            return False
            
        error = self.dll.LSX_CreateLSID(byref(self.LSID))
        if error > 0:
            print(f"[ERROR] 세션 생성 실패, 에러코드: {error}")
            return False
        
        print(f"[OK] 세션 생성 성공, LSID: {self.LSID.value}")
        return True
    
    def connect(self, interface: int = -1, port: str = "", baudrate: int = 57600) -> bool:
        """
        Tango 컨트롤러에 연결
        
        Args:
            interface: -1=자동, 1=RS232, 2=USB, 3=PCIe
            port: COM 포트 (RS232일 때만 사용, 예: "COM5")
            baudrate: 통신 속도 (RS232일 때만 의미 있음)
        """
        if not self.dll:
            print("[ERROR] DLL이 로드되지 않았습니다")
            return False
        
        port_bytes = port.encode("utf-8") if port else b""
        port_param = c_char_p(port_bytes) if port_bytes else b""
        
        print(f"[INFO] 연결 시도 중... (interface={interface}, port='{port}')")
        error = self.dll.LSX_ConnectSimple(self.LSID, interface, port_param, baudrate, 0)
        
        if error > 0:
            print(f"[ERROR] 연결 실패, 에러코드: {error}")
            self._print_connection_error_help(error)
            return False
        
        self.connected = True
        print("[OK] Tango 컨트롤러 연결 성공!")
        return True
    
    def _print_connection_error_help(self, error_code: int):
        """연결 에러 도움말"""
        print("  확인사항:")
        print("  1. Tango 컨트롤러 전원이 켜져 있는지")
        print("  2. USB/PCIe 케이블이 제대로 연결되어 있는지")
        print("  3. 다른 프로그램(rays-on.exe 등)이 이미 연결 중인지")
        print("  4. 장치 드라이버가 설치되어 있는지")
    
    def disconnect(self) -> bool:
        """연결 해제"""
        if not self.connected:
            return True
            
        error = self.dll.LSX_Disconnect(self.LSID)
        if error > 0:
            print(f"[ERROR] 연결 해제 실패, 에러코드: {error}")
            return False
        
        self.connected = False
        print("[OK] 연결 해제 완료")
        return True
    
    def free_session(self):
        """세션 ID 해제"""
        if self.dll and self.LSID.value > 0:
            self.dll.LSX_FreeLSID(self.LSID)
            print("[OK] 세션 해제 완료")
    
    def get_position(self) -> tuple:
        """현재 위치 조회"""
        if not self.connected:
            print("[ERROR] 연결되지 않았습니다")
            return None
        
        dx, dy, dz, da = c_double(), c_double(), c_double(), c_double()
        error = self.dll.LSX_GetPos(self.LSID, byref(dx), byref(dy), byref(dz), byref(da))
        
        if error > 0:
            print(f"[ERROR] 위치 조회 실패, 에러코드: {error}")
            return None
        
        return (dx.value, dy.value, dz.value, da.value)
    
    def get_version(self) -> str:
        """DLL 버전 조회"""
        if not self.dll:
            return None
            
        resp = create_string_buffer(256)
        error = self.dll.LSX_GetDLLVersionString(self.LSID, resp, 256)
        
        if error > 0:
            return None
        return resp.value.decode("ascii")
    
    def get_firmware_version(self) -> str:
        """펌웨어 버전 조회"""
        if not self.connected:
            return None
            
        inp = c_char_p("?version\r".encode("utf-8"))
        resp = create_string_buffer(256)
        error = self.dll.LSX_SendString(self.LSID, inp, resp, 256, True, 5000)
        
        if error > 0:
            return None
        return resp.value.decode("ascii").strip()
    
    def move_absolute(self, x: float, y: float, z: float, a: float = 0, wait: bool = True) -> bool:
        """절대 위치로 이동"""
        if not self.connected:
            print("[ERROR] 연결되지 않았습니다")
            return False
        
        dx, dy, dz, da = c_double(x), c_double(y), c_double(z), c_double(a)
        error = self.dll.LSX_MoveAbs(self.LSID, dx, dy, dz, da, wait)
        
        if error > 0:
            print(f"[ERROR] 이동 실패, 에러코드: {error}")
            return False
        
        return True
    
    def move_relative(self, dx: float, dy: float, dz: float, da: float = 0, wait: bool = True) -> bool:
        """상대 위치로 이동"""
        if not self.connected:
            print("[ERROR] 연결되지 않았습니다")
            return False
        
        ddx, ddy, ddz, dda = c_double(dx), c_double(dy), c_double(dz), c_double(da)
        error = self.dll.LSX_MoveRel(self.LSID, ddx, ddy, ddz, dda, wait)
        
        if error > 0:
            print(f"[ERROR] 이동 실패, 에러코드: {error}")
            return False
        
        return True
    
    def set_velocity(self, vx: float, vy: float, vz: float, va: float) -> bool:
        """속도 설정"""
        if not self.connected:
            print("[ERROR] 연결되지 않았습니다")
            return False
        
        dx, dy, dz, da = c_double(vx), c_double(vy), c_double(vz), c_double(va)
        error = self.dll.LSX_SetVel(self.LSID, dx, dy, dz, da)
        
        if error > 0:
            print(f"[ERROR] 속도 설정 실패, 에러코드: {error}")
            return False
        
        return True
    
    def send_command(self, command: str) -> str:
        """직접 명령 전송"""
        if not self.connected:
            print("[ERROR] 연결되지 않았습니다")
            return None
        
        if not command.endswith('\r'):
            command += '\r'
        
        inp = c_char_p(command.encode("utf-8"))
        resp = create_string_buffer(256)
        error = self.dll.LSX_SendString(self.LSID, inp, resp, 256, True, 5000)
        
        if error > 0:
            print(f"[ERROR] 명령 전송 실패, 에러코드: {error}")
            return None
        
        return resp.value.decode("ascii").strip()


def print_menu():
    """메뉴 출력"""
    print("\n" + "=" * 50)
    print("Tango 스테이지 테스트 메뉴")
    print("=" * 50)
    print("1. 현재 위치 조회")
    print("2. 절대 위치로 이동")
    print("3. 상대 위치로 이동")
    print("4. 속도 설정")
    print("5. 직접 명령 전송")
    print("6. 버전 정보 조회")
    print("-" * 50)
    print("7. [테스트] 연결 유지하고 대기 (rays-on.exe 테스트용)")
    print("-" * 50)
    print("0. 종료")
    print("=" * 50)


def test_rays_on_conflict(tango: TangoController):
    """rays-on.exe 충돌 테스트"""
    print("\n" + "=" * 50)
    print("rays-on.exe 충돌 테스트 모드")
    print("=" * 50)
    print()
    print("현재 상태: Python이 Tango 스테이지 연결을 점유 중")
    print()
    print("테스트 방법:")
    print("  1. 이 상태에서 rays-on.exe를 실행해보세요")
    print("  2. rays-on.exe가 어떻게 반응하는지 확인:")
    print("     - 스테이지 연결 에러만 뜨고 분광 측정은 되는지?")
    print("     - 프로그램 자체가 실행 안 되는지?")
    print("     - 정상적으로 둘 다 되는지? (드문 경우)")
    print()
    print("Enter 누르면 이 테스트를 종료하고 메뉴로 돌아갑니다.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n[대기 중] rays-on.exe 테스트 후 Enter: ")
            if user_input.lower() == 'q':
                break
            
            # 주기적으로 위치 조회해서 연결 상태 확인
            pos = tango.get_position()
            if pos:
                print(f"[연결 유지 중] 현재 위치: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
            else:
                print("[경고] 연결이 끊어졌을 수 있습니다!")
            
            break
            
        except KeyboardInterrupt:
            print("\n테스트 종료")
            break


def interactive_mode(tango: TangoController):
    """대화형 제어 모드"""
    while True:
        print_menu()
        choice = input("선택: ").strip()
        
        if choice == '0':
            print("종료합니다...")
            break
            
        elif choice == '1':
            pos = tango.get_position()
            if pos:
                print(f"\n현재 위치:")
                print(f"  X = {pos[0]:.3f}")
                print(f"  Y = {pos[1]:.3f}")
                print(f"  Z = {pos[2]:.3f}")
                print(f"  A = {pos[3]:.3f}")
        
        elif choice == '2':
            try:
                print("\n절대 위치 입력 (단위: mm)")
                x = float(input("  X: "))
                y = float(input("  Y: "))
                z = float(input("  Z: "))
                
                print(f"\n({x}, {y}, {z})으로 이동 중...")
                if tango.move_absolute(x, y, z):
                    print("[OK] 이동 완료")
            except ValueError:
                print("[ERROR] 숫자를 입력하세요")
        
        elif choice == '3':
            try:
                print("\n상대 이동량 입력 (단위: mm)")
                dx = float(input("  dX: "))
                dy = float(input("  dY: "))
                dz = float(input("  dZ: "))
                
                print(f"\n({dx}, {dy}, {dz}) 만큼 이동 중...")
                if tango.move_relative(dx, dy, dz):
                    print("[OK] 이동 완료")
            except ValueError:
                print("[ERROR] 숫자를 입력하세요")
        
        elif choice == '4':
            try:
                print("\n속도 설정 (단위: mm/s)")
                vx = float(input("  X 속도: "))
                vy = float(input("  Y 속도: "))
                vz = float(input("  Z 속도: "))
                
                if tango.set_velocity(vx, vy, vz, 5.0):
                    print(f"[OK] 속도 설정 완료: ({vx}, {vy}, {vz})")
            except ValueError:
                print("[ERROR] 숫자를 입력하세요")
        
        elif choice == '5':
            print("\nTango 명령어 직접 입력 (예: ?pos, ?vel, !cal)")
            print("도움말: ?=조회, !=실행, 값=설정")
            cmd = input("명령: ").strip()
            if cmd:
                result = tango.send_command(cmd)
                if result is not None:
                    print(f"응답: {result}")
        
        elif choice == '6':
            dll_ver = tango.get_version()
            fw_ver = tango.get_firmware_version()
            print(f"\nDLL 버전: {dll_ver}")
            print(f"펌웨어 버전: {fw_ver}")
        
        elif choice == '7':
            test_rays_on_conflict(tango)
        
        else:
            print("[ERROR] 잘못된 선택입니다")


def main():
    """메인 함수"""
    print("=" * 50)
    print("Tango 스테이지 컨트롤러 테스트")
    print("=" * 50)
    
    # DLL 경로 설정 (필요시 수정)
    dll_path = r"./Tango_DLL.dll"
    
    # 컨트롤러 초기화
    tango = TangoController(dll_path)
    
    # 1. DLL 로드
    if not tango.load_dll():
        input("Enter 누르면 종료...")
        return 1
    
    # 2. 세션 생성
    if not tango.create_session():
        input("Enter 누르면 종료...")
        return 1
    
    # 3. 연결
    print("\n연결 방식 선택:")
    print("  1. 자동 감지 (USB/PCIe)")
    print("  2. COM 포트 지정")
    conn_choice = input("선택 [1]: ").strip() or "1"
    
    if conn_choice == "2":
        port = input("COM 포트 (예: COM5): ").strip()
        if not tango.connect(interface=1, port=port):
            input("Enter 누르면 종료...")
            return 1
    else:
        if not tango.connect():
            input("Enter 누르면 종료...")
            return 1
    
    # 4. 초기 정보 출력
    print("\n" + "-" * 50)
    pos = tango.get_position()
    if pos:
        print(f"현재 위치: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}, A={pos[3]:.3f}")
    
    # 5. 대화형 모드 시작
    try:
        interactive_mode(tango)
    except KeyboardInterrupt:
        print("\n\nCtrl+C 감지, 종료합니다...")
    finally:
        # 정리
        tango.disconnect()
        tango.free_session()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())