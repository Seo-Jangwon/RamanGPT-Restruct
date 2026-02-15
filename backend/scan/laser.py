import serial
import time

class LaserController:
    def __init__(self, port='COM4', baud=115200):
        """
        클래스 초기화: 시리얼 포트 연결
        """
        self.port = port
        self.baud = baud
        self.ser = None
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[LaserController] {self.port} 연결 성공")
        except Exception as e:
            print(f"[LaserController] 연결 실패: {e}")
            self.ser = None

    def _make_packet(self, target_id, cmd, arg):
        """
        내부 함수: 프로토콜 규칙에 맞춰 체크섬 계산 후 패킷 생성
        규칙: @ + 대상 + 명령 + 인자 + 체크섬(hex) + $
        """
        # 1. 체크섬 계산 바디
        body = f"{target_id}{cmd}{arg}"
        
        # 2. 체크섬 계산 (ASCII 합 % 256 -> 2자리 hex 소문자)
        ascii_sum = sum(ord(c) for c in body)
        checksum = f"{ascii_sum % 256:02x}"
        
        # 3. 패킷 조립 및 인코딩
        packet = f"@{body}{checksum}$"
        return packet.encode('utf-8')

    def _send_raw(self, cmd_packet):
        """
        내부 함수: 실제 시리얼 전송 수행
        """
        if self.ser and self.ser.is_open:
            self.ser.write(cmd_packet)
            # 장비가 명령을 먹을 시간을 줌 (안정성 확보)
            time.sleep(0.05) 
        else:
            print("[Error] 시리얼 포트가 열려있지 않습니다.")

    def laser_on(self, duration):
        """
        [메인 함수] 레이저를 지정된 시간(초) 동안 발사
        :param duration: 레이저 유지 시간 (float, 초 단위)
        """
        # 1. 레이저 켜기 (Flag: 1)
        # 패킷: @00SSPW1de$ (예상)
        packet_on = self._make_packet("00", "SSPW", "1")
        print(f"⚡ 레이저 발사 (ON) -> {duration}초 유지")
        self._send_raw(packet_on)

        # 2. 지정된 시간만큼 대기
        time.sleep(duration)

        # 3. 레이저 끄기 (Flag: 0)
        # 패킷: @00SSPW0dd$
        packet_off = self._make_packet("00", "SSPW", "0")
        print("🛑 레이저 정지 (OFF)")
        self._send_raw(packet_off)

    def close(self):
        """
        장비 연결 해제
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[LaserController] {self.port} 연결 종료")

# ==========================================
# 사용 예시 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    # 1. 컨트롤러 생성
    laser = LaserController(port='COM4')
    
    # 2. 레이저 3초 발사 테스트
    # (내부적으로: 켜기 -> 3초 대기 -> 끄기 수행)
    if laser.ser:
        laser.laser_on(3.0)
        
        # 연속 테스트 (1초 쉬고 0.5초 짧게 쏘기)
        time.sleep(1)
        laser.laser_on(0.5)
        
        # 3. 종료
        laser.close()