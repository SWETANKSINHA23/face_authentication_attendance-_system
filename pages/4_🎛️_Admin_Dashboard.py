"""
Admin Dashboard Page
System statistics, monitoring, and analytics
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.database import (
    get_all_active_users,
    get_attendance_history,
    get_attendance_by_user_date
)
from services.attendance_service import get_attendance_summary
import config

st.title("🎛️ Admin Dashboard")

st.markdown("""
Monitor system performance and view analytics across all users.
""")

# Get all users
users = get_all_active_users()
today = date.today()

if not users:
    st.warning("⚠️ No users registered yet.")
    st.info("Navigate to 'Register User' page to get started.")
    st.stop()

# Top metrics
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Total users
total_users = len(users)

# Today's attendance
today_attendance = []
for user in users:
    records = get_attendance_by_user_date(user['id'], today)
    if records:
        today_attendance.append({
            'user_id': user['id'],
            'name': user['name'],
            'record': records[0]
        })

today_count = len(today_attendance)
attendance_rate = (today_count / total_users * 100) if total_users > 0 else 0

# Currently active (punched in but not out)
active_count = len([a for a in today_attendance if a['record']['punch_out_time'] is None])

# Total hours today
today_hours = sum([a['record']['duration'] for a in today_attendance if a['record']['duration']])

with col1:
    st.metric("👥 Total Users", total_users)

with col2:
    st.metric("✅ Present Today", today_count, 
             delta=f"{attendance_rate:.0f}%" if attendance_rate > 0 else None)

with col3:
    st.metric("🔄 Currently Active", active_count)

with col4:
    st.metric("⏱️ Total Hours Today", f"{today_hours:.1f}h")

st.markdown("---")

# Weekly statistics
st.subheader("📅 This Week")

week_start = today - timedelta(days=today.weekday())
week_end = today

week_stats = []
for user in users:
    records = get_attendance_history(user['id'], week_start, week_end)
    days_present = len([r for r in records if r['status'] in ['present', 'half-day']])
    total_hours = sum([r['duration'] for r in records if r['duration']])
    
    week_stats.append({
        'days': days_present,
        'hours': total_hours
    })

col1, col2, col3 = st.columns(3)

with col1:
    avg_days = sum([s['days'] for s in week_stats]) / len(week_stats) if week_stats else 0
    st.metric("Avg Days/User", f"{avg_days:.1f}")

with col2:
    total_week_hours = sum([s['hours'] for s in week_stats])
    st.metric("Total Week Hours", f"{total_week_hours:.1f}h")

with col3:
    avg_hours = sum([s['hours'] for s in week_stats]) / len(week_stats) if week_stats else 0
    st.metric("Avg Hours/User", f"{avg_hours:.1f}h")

st.markdown("---")

# Today's attendance details
st.subheader("📋 Today's Attendance Details")

if today_attendance:
    attendance_table = []
    
    for att in today_attendance:
        record = att['record']
        
        # Calculate status emoji
        if record['punch_out_time'] is None:
            status_emoji = "🟢"
            status_text = "Active"
        elif record['status'] == 'present':
            status_emoji = "✅"
            status_text = "Present"
        elif record['status'] == 'half-day':
            status_emoji = "⚠️"
            status_text = "Half-Day"
        else:
            status_emoji = "❌"
            status_text = "Early-Exit"
        
        # Find user info
        user = next(u for u in users if u['id'] == att['user_id'])
        
        attendance_table.append({
            "Status": f"{status_emoji} {status_text}",
            "Employee ID": user['employee_id'],
            "Name": att['name'],
            "Punch In": datetime.fromisoformat(record['punch_in_time']).strftime('%H:%M:%S'),
            "Punch Out": datetime.fromisoformat(record['punch_out_time']).strftime('%H:%M:%S') if record['punch_out_time'] else "-",
            "Hours": f"{record['duration']:.2f}h" if record['duration'] else "In Progress",
            "Department": user['department'] or "N/A"
        })
    
    st.dataframe(attendance_table, use_container_width=True, hide_index=True)
else:
    st.info("📭 No attendance recorded today yet.")

# Absent users
absent_users = [u for u in users if u['id'] not in [a['user_id'] for a in today_attendance]]

if absent_users:
    with st.expander(f"❌ Absent Today ({len(absent_users)})"):
        absent_table = []
        for user in absent_users:
            absent_table.append({
                "Employee ID": user['employee_id'],
                "Name": user['name'],
                "Department": user['department'] or "N/A"
            })
        
        st.dataframe(absent_table, use_container_width=True, hide_index=True)

st.markdown("---")

# Analytics
st.subheader("📈 Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Attendance Trends", "Department Stats", "User Rankings", "System Health"])

with tab1:
    st.markdown("#### Last 30 Days Attendance Trend")
    
    # Calculate daily attendance for last 30 days
    days_range = 30
    trend_data = []
    
    for i in range(days_range, -1, -1):
        check_date = today - timedelta(days=i)
        day_count = 0
        
        for user in users:
            records = get_attendance_by_user_date(user['id'], check_date)
            if records:
                day_count += 1
        
        trend_data.append({
            'Date': check_date,
            'Count': day_count,
            'Percentage': (day_count / total_users * 100) if total_users > 0 else 0
        })
    
    df_trend = pd.DataFrame(trend_data)
    
    # Line chart
    st.line_chart(df_trend.set_index('Date')['Count'])
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_attendance = df_trend['Count'].mean()
        st.metric("Avg Daily Attendance", f"{avg_attendance:.1f}")
    
    with col2:
        max_attendance = df_trend['Count'].max()
        st.metric("Best Day", f"{max_attendance} users")
    
    with col3:
        avg_percentage = df_trend['Percentage'].mean()
        st.metric("Avg Attendance Rate", f"{avg_percentage:.1f}%")

with tab2:
    st.markdown("#### Department-wise Statistics")
    
    dept_stats = {}
    
    for user in users:
        dept = user['department'] or "Unassigned"
        
        if dept not in dept_stats:
            dept_stats[dept] = {'users': 0, 'present_today': 0}
        
        dept_stats[dept]['users'] += 1
        
        # Check if present today
        if user['id'] in [a['user_id'] for a in today_attendance]:
            dept_stats[dept]['present_today'] += 1
    
    # Create DataFrame
    dept_df = pd.DataFrame([
        {
            'Department': dept,
            'Total Users': stats['users'],
            'Present Today': stats['present_today'],
            'Attendance Rate': f"{(stats['present_today'] / stats['users'] * 100):.1f}%"
        }
        for dept, stats in dept_stats.items()
    ])
    
    st.dataframe(dept_df, use_container_width=True, hide_index=True)
    
    # Bar chart
    st.bar_chart(dept_df.set_index('Department')['Total Users'])

with tab3:
    st.markdown("#### Top Performers (Last 30 Days)")
    
    user_rankings = []
    
    for user in users:
        summary = get_attendance_summary(user['id'], days=30)
        
        user_rankings.append({
            'Employee ID': user['employee_id'],
            'Name': user['name'],
            'Days Present': summary['total_days'],
            'Total Hours': summary['total_hours'],
            'Avg Hours/Day': summary['average_hours'],
            'Department': user['department'] or "Unassigned"
        })
    
    # Sort by total hours
    user_rankings.sort(key=lambda x: x['Total Hours'], reverse=True)
    
    # Display top 10
    st.dataframe(user_rankings[:10], use_container_width=True, hide_index=True)

with tab4:
    st.markdown("#### System Health Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Database Status**")
        st.success("✅ Connected")
        st.write(f"- Total Records: {sum([len(get_attendance_history(u['id'], today - timedelta(days=365), today)) for u in users])}")
        st.write(f"- Database Size: ~{total_users * 50}KB estimated")
        
        st.markdown("**Recognition Performance**")
        st.info(f"- Model: {config.FACE_DETECTION_MODEL.upper()}")
        st.info(f"- Tolerance: {config.RECOGNITION_TOLERANCE}")
        st.info(f"- Anti-Spoofing: {'Enabled' if config.ENABLE_ANTI_SPOOFING else 'Disabled'}")
    
    with col2:
        st.markdown("**System Configuration**")
        st.write(f"- Camera Index: {config.CAMERA_INDEX}")
        st.write(f"- Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        st.write(f"- Min Work Hours: {config.MIN_WORK_HOURS}h")
        st.write(f"- Half Day Hours: {config.HALF_DAY_HOURS}h")
        
        st.markdown("**Recent Activity**")
        recent_count = len(today_attendance)
        st.write(f"- Today: {recent_count} attendance records")
        st.write(f"- Active Users: {active_count}")

st.markdown("---")

# Quick statistics cards
st.subheader("📌 Quick Stats")

col1, col2, col3, col4 = st.columns(4)

# Calculate various metrics
week_attendance = []
for user in users:
    records = get_attendance_history(user['id'], week_start, week_end)
    week_attendance.extend(records)

month_start = today.replace(day=1)
month_attendance = []
for user in users:
    records = get_attendance_history(user['id'], month_start, today)
    month_attendance.extend(records)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
        <h3 style='margin: 0; color: white;'>{}</h3>
        <p style='margin: 0.5rem 0 0 0;'>This Week</p>
    </div>
    """.format(len(week_attendance)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
        <h3 style='margin: 0; color: white;'>{}</h3>
        <p style='margin: 0.5rem 0 0 0;'>This Month</p>
    </div>
    """.format(len(month_attendance)), unsafe_allow_html=True)

with col3:
    avg_work_hours = today_hours / today_count if today_count > 0 else 0
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
        <h3 style='margin: 0; color: white;'>{:.1f}h</h3>
        <p style='margin: 0.5rem 0 0 0;'>Avg Work Today</p>
    </div>
    """.format(avg_work_hours), unsafe_allow_html=True)

with col4:
    on_time_count = len([a for a in today_attendance if datetime.fromisoformat(a['record']['punch_in_time']).time() <= datetime.strptime(config.WORK_START_TIME, '%H:%M').time()])
    on_time_rate = (on_time_count / today_count * 100) if today_count > 0 else 0
    st.markdown("""
    <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
        <h3 style='margin: 0; color: white;'>{:.0f}%</h3>
        <p style='margin: 0.5rem 0 0 0;'>On-Time Today</p>
    </div>
    """.format(on_time_rate), unsafe_allow_html=True)

# Export dashboard data
with st.expander("📥 Export Dashboard Data"):
    # Prepare comprehensive data
    export_data = {
        'Today': today.strftime('%Y-%m-%d'),
        'Total Users': total_users,
        'Present Today': today_count,
        'Attendance Rate': f"{attendance_rate:.2f}%",
        'Total Hours Today': f"{today_hours:.2f}h",
        'Active Users': active_count
    }
    
    st.json(export_data)
    
    # Download button
    import json
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="Download Dashboard Data (JSON)",
        data=json_str,
        file_name=f"dashboard_{today.strftime('%Y%m%d')}.json",
        mime="application/json"
    )
