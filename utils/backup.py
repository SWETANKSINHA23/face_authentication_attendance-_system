"""
Database Backup and Maintenance Utilities
"""

import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import config


def create_backup(backup_dir: str = None) -> dict:
    """
    Create backup of database and important files
    
    Args:
        backup_dir: Directory to store backup (default: data/backups)
    
    Returns:
        Dictionary with backup status and file paths
    """
    try:
        # Setup backup directory
        if backup_dir is None:
            backup_dir = config.DATA_DIR / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup database
        db_backup_path = backup_dir / f"database_backup_{timestamp}.db"
        shutil.copy2(config.DATABASE_PATH, db_backup_path)
        
        # Backup config
        config_backup_path = backup_dir / f"config_backup_{timestamp}.py"
        shutil.copy2(config.BASE_DIR / "config.py", config_backup_path)
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'database': str(db_backup_path),
            'config': str(config_backup_path),
            'database_size': db_backup_path.stat().st_size,
        }
        
        manifest_path = backup_dir / f"manifest_{timestamp}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return {
            'success': True,
            'message': 'Backup created successfully',
            'backup_dir': str(backup_dir),
            'files': manifest
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Backup failed: {str(e)}',
            'error': str(e)
        }


def restore_backup(backup_file: str) -> dict:
    """
    Restore database from backup
    
    Args:
        backup_file: Path to backup database file
    
    Returns:
        Dictionary with restore status
    """
    try:
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            return {
                'success': False,
                'message': f'Backup file not found: {backup_file}'
            }
        
        # Create backup of current database first
        current_backup = create_backup()
        
        # Restore from backup
        shutil.copy2(backup_path, config.DATABASE_PATH)
        
        return {
            'success': True,
            'message': 'Database restored successfully',
            'restored_from': str(backup_path),
            'current_backed_up': current_backup.get('backup_dir')
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Restore failed: {str(e)}',
            'error': str(e)
        }


def cleanup_old_backups(days: int = 30, backup_dir: str = None) -> dict:
    """
    Delete backups older than specified days
    
    Args:
        days: Keep backups from last N days
        backup_dir: Directory containing backups
    
    Returns:
        Dictionary with cleanup status
    """
    try:
        if backup_dir is None:
            backup_dir = config.DATA_DIR / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        if not backup_dir.exists():
            return {
                'success': True,
                'message': 'No backup directory found',
                'deleted_count': 0
            }
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for file in backup_dir.glob('*'):
            if file.stat().st_mtime < cutoff_date.timestamp():
                file.unlink()
                deleted_count += 1
        
        return {
            'success': True,
            'message': f'Deleted {deleted_count} old backup(s)',
            'deleted_count': deleted_count,
            'kept_days': days
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Cleanup failed: {str(e)}',
            'error': str(e)
        }


def verify_database() -> dict:
    """
    Verify database integrity
    
    Returns:
        Dictionary with verification status
    """
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        
        # Get database statistics
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = cursor.fetchone()[0]
        
        conn.close()
        
        is_ok = result == "ok"
        
        return {
            'success': is_ok,
            'message': 'Database is healthy' if is_ok else f'Database issues: {result}',
            'integrity': result,
            'statistics': {
                'users': user_count,
                'attendance_records': attendance_count,
                'database_size': config.DATABASE_PATH.stat().st_size
            }
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Verification failed: {str(e)}',
            'error': str(e)
        }


def export_data_to_json(output_file: str = None) -> dict:
    """
    Export all data to JSON format
    
    Args:
        output_file: Output JSON file path
    
    Returns:
        Dictionary with export status
    """
    try:
        import pandas as pd
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = config.DATA_DIR / f"export_{timestamp}.json"
        
        conn = sqlite3.connect(config.DATABASE_PATH)
        
        # Export users
        users_df = pd.read_sql_query("SELECT * FROM users", conn)
        users_df['face_encoding'] = users_df['face_encoding'].apply(
            lambda x: 'BINARY_DATA' if x else None
        )
        
        # Export attendance
        attendance_df = pd.read_sql_query("SELECT * FROM attendance", conn)
        
        conn.close()
        
        # Combine data
        export_data = {
            'export_date': datetime.now().isoformat(),
            'users': users_df.to_dict('records'),
            'attendance': attendance_df.to_dict('records')
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return {
            'success': True,
            'message': 'Data exported successfully',
            'output_file': str(output_file),
            'records': {
                'users': len(users_df),
                'attendance': len(attendance_df)
            }
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Export failed: {str(e)}',
            'error': str(e)
        }


if __name__ == "__main__":
    print("Database Backup & Maintenance Utilities\n")
    
    # Test backup
    print("1. Creating backup...")
    result = create_backup()
    print(f"   Status: {result['message']}")
    
    # Test verification
    print("\n2. Verifying database...")
    result = verify_database()
    print(f"   Status: {result['message']}")
    if result['success']:
        print(f"   Users: {result['statistics']['users']}")
        print(f"   Records: {result['statistics']['attendance_records']}")
    
    print("\nUtilities loaded successfully!")
