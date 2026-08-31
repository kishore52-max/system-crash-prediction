import sqlite3
from datetime import datetime


# ==========================================
# DATABASE CONNECTION
# ==========================================

DB_NAME = "system_monitor.db"


def get_connection():

    return sqlite3.connect(DB_NAME)


# ==========================================
# CREATE TABLE
# ==========================================

def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            cpu_usage REAL,

            ram_usage REAL,

            disk_usage REAL,

            network_usage REAL,

            active_processes INTEGER,

            risk TEXT,

            confidence REAL,

            stress_score REAL

        )
    """)

    connection.commit()

    connection.close()


# ==========================================
# SAVE PREDICTION
# ==========================================

def save_prediction(
    cpu,
    ram,
    disk,
    network,
    processes,
    risk,
    confidence,
    stress_score
):

    connection = get_connection()

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO predictions
        (
            timestamp,
            cpu_usage,
            ram_usage,
            disk_usage,
            network_usage,
            active_processes,
            risk,
            confidence,
            stress_score
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        timestamp,
        cpu,
        ram,
        disk,
        network,
        processes,
        risk,
        confidence,
        stress_score

    ))

    connection.commit()

    connection.close()


# ==========================================
# GET PREDICTION HISTORY
# ==========================================

def get_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            timestamp,
            cpu_usage,
            ram_usage,
            disk_usage,
            network_usage,
            active_processes,
            risk,
            confidence,
            stress_score

        FROM predictions

        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


# ==========================================
# DELETE ALL HISTORY
# ==========================================

def delete_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    connection.commit()

    connection.close()