"""
이메일 발송 모듈 (Gmail SMTP)

설정:
  1. email_config.json 을 열고 Gmail 정보 입력
  2. Gmail → 계정 관리 → 보안 → 앱 비밀번호 생성 (2단계 인증 필요)
  3. 16자리 앱 비밀번호를 'app_password'에 입력

email_config.json 은 .gitignore에 포함되어 있다 — git에 올리지 말 것.
"""

import json
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / 'email_config.json'


def _create_default_config():
    default = {
        "sender_email": "your_gmail@gmail.com",
        "app_password": "xxxx xxxx xxxx xxxx",
        "recipient_email": "your_gmail@gmail.com",
        "enabled": False,
        "_note": "Gmail 앱 비밀번호: myaccount.google.com/apppasswords (2단계 인증 필요)"
    }
    with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(default, f, ensure_ascii=False, indent=2)


def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        _create_default_config()
        raise FileNotFoundError(
            f'email_config.json 이 없어 기본 파일을 생성했습니다. '
            f'Gmail 정보를 입력하세요: {_CONFIG_PATH}')
    with open(_CONFIG_PATH, encoding='utf-8') as f:
        return json.load(f)


def send_report(subject: str, body: str, attachment_path: str | None = None) -> bool:
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        print(f'  [이메일] 설정 없음: {e}')
        return False
    if not cfg.get('enabled', False):
        print('  [이메일] 비활성화 (email_config.json → "enabled": true)')
        return False

    sender    = cfg['sender_email']
    password  = cfg['app_password'].replace(' ', '')
    recipient = cfg['recipient_email']

    msg = MIMEMultipart()
    msg['From'], msg['To'], msg['Subject'] = sender, recipient, subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        print(f'  [이메일] 발송 완료 → {recipient}')
        return True
    except smtplib.SMTPAuthenticationError:
        print('  [이메일] 인증 실패: 앱 비밀번호 확인 필요')
        return False
    except Exception as e:
        print(f'  [이메일] 발송 실패: {e}')
        return False
