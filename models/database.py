"""
Database models and operations for Face Authentication Attendance System
Uses SQLite for lightweight, portable storage
"""

import sqlite3
import pickle
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import config


class Database:
    """Database handler for user and attendance management"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or str(config.DATABASE_PATH)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.close()


def init_database(db_path: str = None):
    """
    Initialize database with required tables
    
    Args:
        db_path: Path to database file (optional)
    """
    db_path = db_path or str(config.DATABASE_PATH)
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            department TEXT,
            face_encoding BLOB NOT NULL,
            registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # Create attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            punch_in_time TIMESTAMP NOT NULL,
            punch_out_time TIMESTAMP,
            date DATE NOT NULL,
            duration REAL,
            status TEXT DEFAULT 'present',
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date, punch_in_time)
        )
    """)
    
    # Create face_captures table (optional audit trail)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT,
            capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            capture_type TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_employee_id 
        ON users(employee_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_user_date 
        ON attendance(user_id, date)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized successfully at {db_path}")


def create_user(employee_id: str, name: str, face_encoding: Any, 
                email: str = None, department: str = None) -> int:
    """
    Create a new user in the database
    
    Args:
        employee_id: Unique employee identifier
        name: Full name of the user
        face_encoding: 128D face embedding (numpy array)
        email: Email address (optional)
        department: Department name (optional)
    
    Returns:
        int: User ID of the created user
    
    Raises:
        sqlite3.IntegrityError: If employee_id or email already exists
    """
    # Serialize face encoding to binary
    encoding_blob = pickle.dumps(face_encoding)
    
    with Database() as db:
        db.cursor.execute("""
            INSERT INTO users (employee_id, name, email, department, face_encoding)
            VALUES (?, ?, ?, ?, ?)
        """, (employee_id, name, email, department, encoding_blob))
        
        user_id = db.cursor.lastrowid
    
    return user_id


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve user by user ID
    
    Args:
        user_id: User ID
    
    Returns:
        Dict containing user data or None if not found
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, employee_id, name, email, department, 
                   face_encoding, registered_date, is_active
            FROM users WHERE id = ?
        """, (user_id,))
        
        row = db.cursor.fetchone()
    
    if row:
        user = dict(row)
        user['face_encoding'] = pickle.loads(user['face_encoding'])
        return user
    return None


def get_user_by_employee_id(employee_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve user by employee ID
    
    Args:
        employee_id: Employee ID
    
    Returns:
        Dict containing user data or None if not found
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, employee_id, name, email, department, 
                   face_encoding, registered_date, is_active
            FROM users WHERE employee_id = ?
        """, (employee_id,))
        
        row = db.cursor.fetchone()
    
    if row:
        user = dict(row)
        user['face_encoding'] = pickle.loads(user['face_encoding'])
        return user
    return None


def get_all_active_users() -> List[Dict[str, Any]]:
    """
    Retrieve all active users
    
    Returns:
        List of user dictionaries
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, employee_id, name, email, department, 
                   face_encoding, registered_date, is_active
            FROM users WHERE is_active = 1
            ORDER BY name
        """)
        
        rows = db.cursor.fetchall()
    
    users = []
    for row in rows:
        user = dict(row)
        user['face_encoding'] = pickle.loads(user['face_encoding'])
        users.append(user)
    
    return users


def get_all_user_encodings() -> Dict[int, Any]:
    """
    Retrieve all face encodings for active users
    
    Returns:
        Dict mapping user_id to face_encoding
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, face_encoding
            FROM users WHERE is_active = 1
        """)
        
        rows = db.cursor.fetchall()
    
    encodings = {}
    for row in rows:
        encodings[row['id']] = pickle.loads(row['face_encoding'])
    
    return encodings


def update_user(user_id: int, **kwargs) -> bool:
    """
    Update user information
    
    Args:
        user_id: User ID
        **kwargs: Fields to update (name, email, department, is_active)
    
    Returns:
        bool: True if update successful
    """
    allowed_fields = ['name', 'email', 'department', 'is_active', 'face_encoding']
    
    # Filter only allowed fields
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not updates:
        return False
    
    # Serialize face_encoding if present
    if 'face_encoding' in updates:
        updates['face_encoding'] = pickle.dumps(updates['face_encoding'])
    
    # Build UPDATE query dynamically
    set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [user_id]
    
    with Database() as db:
        db.cursor.execute(f"""
            UPDATE users SET {set_clause}
            WHERE id = ?
        """, values)
        
        success = db.cursor.rowcount > 0
    
    return success


def deactivate_user(user_id: int) -> bool:
    """
    Deactivate a user (soft delete)
    
    Args:
        user_id: User ID
    
    Returns:
        bool: True if deactivation successful
    """
    return update_user(user_id, is_active=False)


def create_attendance_record(user_id: int, punch_in_time: datetime, 
                             attendance_date: date = None) -> int:
    """
    Create a new attendance record (punch-in)
    
    Args:
        user_id: User ID
        punch_in_time: Timestamp of punch-in
        attendance_date: Date of attendance (defaults to today)
    
    Returns:
        int: Attendance record ID
    """
    if attendance_date is None:
        attendance_date = punch_in_time.date()
    
    with Database() as db:
        db.cursor.execute("""
            INSERT INTO attendance (user_id, punch_in_time, date)
            VALUES (?, ?, ?)
        """, (user_id, punch_in_time, attendance_date))
        
        record_id = db.cursor.lastrowid
    
    return record_id


def update_punch_out(attendance_id: int, punch_out_time: datetime) -> bool:
    """
    Update attendance record with punch-out time and calculate duration
    
    Args:
        attendance_id: Attendance record ID
        punch_out_time: Timestamp of punch-out
    
    Returns:
        bool: True if update successful
    """
    with Database() as db:
        # Get punch_in_time first
        db.cursor.execute("""
            SELECT punch_in_time FROM attendance WHERE id = ?
        """, (attendance_id,))
        
        row = db.cursor.fetchone()
        if not row:
            return False
        
        punch_in_time = datetime.fromisoformat(row['punch_in_time'])
        duration = (punch_out_time - punch_in_time).total_seconds() / 3600  # Hours
        
        # Determine status based on duration
        if duration >= config.MIN_WORK_HOURS:
            status = 'present'
        elif duration >= config.HALF_DAY_HOURS:
            status = 'half-day'
        else:
            status = 'early-exit'
        
        # Update record
        db.cursor.execute("""
            UPDATE attendance 
            SET punch_out_time = ?, duration = ?, status = ?
            WHERE id = ?
        """, (punch_out_time, duration, status, attendance_id))
        
        success = db.cursor.rowcount > 0
    
    return success


def get_attendance_by_user_date(user_id: int, attendance_date: date) -> List[Dict[str, Any]]:
    """
    Get attendance records for a user on a specific date
    
    Args:
        user_id: User ID
        attendance_date: Date to query
    
    Returns:
        List of attendance records
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, user_id, punch_in_time, punch_out_time, 
                   date, duration, status
            FROM attendance 
            WHERE user_id = ? AND date = ?
            ORDER BY punch_in_time DESC
        """, (user_id, attendance_date))
        
        rows = db.cursor.fetchall()
    
    return [dict(row) for row in rows]


def get_latest_attendance(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get the most recent attendance record for a user
    
    Args:
        user_id: User ID
    
    Returns:
        Latest attendance record or None
    """
    with Database() as db:
        db.cursor.execute("""
            SELECT id, user_id, punch_in_time, punch_out_time, 
                   date, duration, status
            FROM attendance 
            WHERE user_id = ?
            ORDER BY punch_in_time DESC
            LIMIT 1
        """, (user_id,))
        
        row = db.cursor.fetchone()
    
    return dict(row) if row else None


def get_attendance_history(user_id: int, start_date: date = None, 
                           end_date: date = None) -> List[Dict[str, Any]]:
    """
    Get attendance history for a user within date range
    
    Args:
        user_id: User ID
        start_date: Start date (optional)
        end_date: End date (optional)
    
    Returns:
        List of attendance records
    """
    query = """
        SELECT id, user_id, punch_in_time, punch_out_time, 
               date, duration, status
        FROM attendance 
        WHERE user_id = ?
    """
    
    params = [user_id]
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY date DESC, punch_in_time DESC"
    
    with Database() as db:
        db.cursor.execute(query, params)
        rows = db.cursor.fetchall()
    
    return [dict(row) for row in rows]


def save_face_capture(user_id: int, image_path: str, capture_type: str):
    """
    Save face capture record for audit trail
    
    Args:
        user_id: User ID
        image_path: Path to captured image
        capture_type: Type of capture (registration/authentication)
    
    Returns:
        int: Capture record ID
    """
    with Database() as db:
        db.cursor.execute("""
            INSERT INTO face_captures (user_id, image_path, capture_type)
            VALUES (?, ?, ?)
        """, (user_id, image_path, capture_type))
        
        capture_id = db.cursor.lastrowid
    
    return capture_id


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
