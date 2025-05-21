# app/db_manager.py
import pymysql
from typing import List, Dict, Union
import os
from dotenv import load_dotenv

# ✅ .env 불러오기 (루트 디렉토리에 .env 있어야 함)
load_dotenv()

# ✅ 환경 변수 가져오기
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def fetch_recent_dialogue(member_id: Union[int, str], limit: int = 20) -> List[Dict]:
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        query = """
            SELECT message, sender, type, send_time
            FROM chat
            WHERE member_id = %s
              AND (
                type = 'SEND' OR
                (sender = 'BOT' AND type != 'ENTER')
              )
            ORDER BY send_time DESC
            LIMIT %s;
        """

        with conn.cursor() as cursor:
            cursor.execute(query, (member_id, limit))
            rows = cursor.fetchall()

        conn.close()
        rows.reverse()

        return [
            {
                "message": row["message"],
                "type": "SEND" if row["sender"] == "USER" else "RECEIVE",
                "sender": row["sender"],  # ✅ 추가됨
                "send_time": str(row["send_time"]),  # ✅ merge에서 사용됨
                "member_id": int(member_id)
            }
            for row in rows
        ]

    except Exception as e:
        print(f"❌ DB fetch_recent_dialogue 오류: {e}")
        return []