"""
Advanced Analytics Module
Provides detailed attendance analytics, late arrival detection, and comprehensive reports
"""

from datetime import datetime, date, timedelta, time
from typing import Dict, List, Tuple
import pandas as pd

from models.database import (
    get_attendance_history,
    get_all_active_users,
    get_attendance_by_user_date
)
import config


def detect_late_arrival(punch_in_time: datetime, work_start_time: str = None) -> Dict:
    """
    Detect if punch-in is late
    
    Args:
        punch_in_time: Actual punch-in timestamp
        work_start_time: Expected start time (HH:MM format)
    
    Returns:
        Dictionary with late status and details
    """
    work_start_time = work_start_time or config.WORK_START_TIME
    
    # Parse expected start time
    expected_time = datetime.strptime(work_start_time, '%H:%M').time()
    
    # Get actual punch-in time
    actual_time = punch_in_time.time()
    
    # Calculate if late (with 15-minute grace period)
    grace_period = timedelta(minutes=15)
    expected_datetime = datetime.combine(punch_in_time.date(), expected_time)
    grace_end = expected_datetime + grace_period
    
    is_late = punch_in_time > grace_end
    
    if is_late:
        late_by = punch_in_time - grace_end
        late_minutes = int(late_by.total_seconds() / 60)
    else:
        late_minutes = 0
    
    return {
        'is_late': is_late,
        'expected_time': expected_time.strftime('%H:%M'),
        'actual_time': actual_time.strftime('%H:%M'),
        'late_by_minutes': late_minutes,
        'status': 'Late' if is_late else 'On Time'
    }


def get_weekly_report(user_id: int = None, week_start: date = None) -> Dict:
    """
    Generate comprehensive weekly attendance report
    
    Args:
        user_id: Optional user ID (None for all users)
        week_start: Start of week (default: current week Monday)
    
    Returns:
        Dictionary with weekly statistics
    """
    # Calculate week start and end
    if week_start is None:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
    
    week_end = week_start + timedelta(days=6)
    
    # Get users
    if user_id:
        from models.database import get_user_by_id
        users = [get_user_by_id(user_id)]
    else:
        users = get_all_active_users()
    
    # Collect statistics
    total_days_worked = 0
    total_hours = 0
    late_days = 0
    early_exits = 0
    perfect_days = 0
    
    daily_breakdown = []
    
    for user in users:
        records = get_attendance_history(user['id'], week_start, week_end)
        
        for record in records:
            total_days_worked += 1
            
            if record['duration']:
                total_hours += record['duration']
            
            # Check if late
            punch_in = datetime.fromisoformat(record['punch_in_time'])
            late_info = detect_late_arrival(punch_in)
            
            if late_info['is_late']:
                late_days += 1
            
            # Check status
            if record['status'] == 'early-exit':
                early_exits += 1
            elif record['status'] == 'present' and not late_info['is_late']:
                perfect_days += 1
            
            daily_breakdown.append({
                'user_id': user['id'],
                'user_name': user['name'],
                'date': record['date'],
                'punch_in': record['punch_in_time'],
                'punch_out': record['punch_out_time'],
                'duration': record['duration'],
                'status': record['status'],
                'late': late_info['is_late'],
                'late_by': late_info['late_by_minutes']
            })
    
    avg_hours = total_hours / total_days_worked if total_days_worked > 0 else 0
    
    return {
        'week_start': week_start.isoformat(),
        'week_end': week_end.isoformat(),
        'summary': {
            'total_days_worked': total_days_worked,
            'total_hours': round(total_hours, 2),
            'average_hours_per_day': round(avg_hours, 2),
            'late_arrivals': late_days,
            'early_exits': early_exits,
            'perfect_attendance_days': perfect_days,
            'punctuality_rate': round((1 - late_days / total_days_worked) * 100, 1) if total_days_worked > 0 else 0
        },
        'daily_breakdown': daily_breakdown
    }


def get_monthly_report(user_id: int = None, year: int = None, month: int = None) -> Dict:
    """
    Generate comprehensive monthly attendance report
    
    Args:
        user_id: Optional user ID (None for all users)
        year: Year (default: current year)
        month: Month (default: current month)
    
    Returns:
        Dictionary with monthly statistics
    """
    # Default to current month
    if year is None or month is None:
        today = date.today()
        year = today.year
        month = today.month
    
    # Calculate month start and end
    month_start = date(year, month, 1)
    
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)
    
    # Get users
    if user_id:
        from models.database import get_user_by_id
        users = [get_user_by_id(user_id)]
    else:
        users = get_all_active_users()
    
    # Collect statistics
    user_stats = []
    
    for user in users:
        records = get_attendance_history(user['id'], month_start, month_end)
        
        days_present = len([r for r in records if r['status'] in ['present', 'half-day']])
        total_hours = sum([r['duration'] for r in records if r['duration']])
        late_count = 0
        
        for record in records:
            punch_in = datetime.fromisoformat(record['punch_in_time'])
            if detect_late_arrival(punch_in)['is_late']:
                late_count += 1
        
        avg_hours = total_hours / days_present if days_present > 0 else 0
        
        user_stats.append({
            'user_id': user['id'],
            'employee_id': user['employee_id'],
            'name': user['name'],
            'department': user['department'],
            'days_present': days_present,
            'total_hours': round(total_hours, 2),
            'average_hours': round(avg_hours, 2),
            'late_arrivals': late_count,
            'punctuality_rate': round((1 - late_count / days_present) * 100, 1) if days_present > 0 else 0
        })
    
    # Calculate totals
    total_days = sum([u['days_present'] for u in user_stats])
    total_hours = sum([u['total_hours'] for u in user_stats])
    total_late = sum([u['late_arrivals'] for u in user_stats])
    
    return {
        'year': year,
        'month': month,
        'month_name': month_start.strftime('%B %Y'),
        'working_days': (month_end - month_start).days + 1,
        'summary': {
            'total_attendance_days': total_days,
            'total_working_hours': round(total_hours, 2),
            'total_late_arrivals': total_late,
            'average_attendance_rate': round(total_days / len(users) / ((month_end - month_start).days + 1) * 100, 1) if users else 0
        },
        'user_stats': user_stats
    }


def get_user_performance_score(user_id: int, days: int = 30) -> Dict:
    """
    Calculate user performance score based on multiple factors
    
    Args:
        user_id: User ID
        days: Number of days to analyze
    
    Returns:
        Dictionary with performance score and breakdown
    """
    from models.database import get_user_by_id
    from services.attendance_service import get_attendance_summary
    
    user = get_user_by_id(user_id)
    if not user:
        return {'error': 'User not found'}
    
    # Get attendance summary
    summary = get_attendance_summary(user_id, days)
    
    # Get records for analysis
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    records = get_attendance_history(user_id, start_date, end_date)
    
    # Calculate metrics
    attendance_rate = (summary['total_days'] / days) * 100
    
    # Late arrivals
    late_count = 0
    for record in records:
        punch_in = datetime.fromisoformat(record['punch_in_time'])
        if detect_late_arrival(punch_in)['is_late']:
            late_count += 1
    
    punctuality_rate = (1 - late_count / summary['total_days']) * 100 if summary['total_days'] > 0 else 0
    
    # Hours consistency
    hours_consistency = min(100, (summary['average_hours'] / config.MIN_WORK_HOURS) * 100)
    
    # Calculate weighted score
    score = (
        attendance_rate * 0.4 +  # 40% weight
        punctuality_rate * 0.3 + # 30% weight
        hours_consistency * 0.3  # 30% weight
    )
    
    # Grade
    if score >= 90:
        grade = 'A (Excellent)'
    elif score >= 80:
        grade = 'B (Good)'
    elif score >= 70:
        grade = 'C (Average)'
    elif score >= 60:
        grade = 'D (Below Average)'
    else:
        grade = 'F (Poor)'
    
    return {
        'user_id': user_id,
        'employee_id': user['employee_id'],
        'name': user['name'],
        'period_days': days,
        'performance_score': round(score, 1),
        'grade': grade,
        'metrics': {
            'attendance_rate': round(attendance_rate, 1),
            'punctuality_rate': round(punctuality_rate, 1),
            'hours_consistency': round(hours_consistency, 1),
            'days_present': summary['total_days'],
            'late_arrivals': late_count,
            'total_hours': summary['total_hours'],
            'average_hours': summary['average_hours']
        }
    }


def get_department_analytics(department: str = None, days: int = 30) -> Dict:
    """
    Get department-wise analytics
    
    Args:
        department: Department name (None for all)
        days: Number of days to analyze
    
    Returns:
        Dictionary with department analytics
    """
    users = get_all_active_users()
    
    if department:
        users = [u for u in users if u['department'] == department]
    
    if not users:
        return {'error': 'No users found'}
    
    # Get analytics for each user
    dept_stats = []
    
    for user in users:
        perf = get_user_performance_score(user['id'], days)
        if 'error' not in perf:
            dept_stats.append(perf)
    
    # Calculate department averages
    if dept_stats:
        avg_score = sum([s['performance_score'] for s in dept_stats]) / len(dept_stats)
        avg_attendance = sum([s['metrics']['attendance_rate'] for s in dept_stats]) / len(dept_stats)
        avg_punctuality = sum([s['metrics']['punctuality_rate'] for s in dept_stats]) / len(dept_stats)
        total_hours = sum([s['metrics']['total_hours'] for s in dept_stats])
    else:
        avg_score = avg_attendance = avg_punctuality = total_hours = 0
    
    return {
        'department': department or 'All Departments',
        'user_count': len(users),
        'period_days': days,
        'department_average': {
            'performance_score': round(avg_score, 1),
            'attendance_rate': round(avg_attendance, 1),
            'punctuality_rate': round(avg_punctuality, 1),
            'total_hours': round(total_hours, 2)
        },
        'user_stats': dept_stats
    }


if __name__ == "__main__":
    print("Advanced Analytics Module Loaded")
    print("\nFeatures:")
    print("  - Late arrival detection")
    print("  - Weekly reports")
    print("  - Monthly reports")
    print("  - Performance scoring")
    print("  - Department analytics")
