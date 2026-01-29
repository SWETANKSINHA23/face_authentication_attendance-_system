"""
View Attendance Records Page
View, filter, and export attendance history
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
    get_attendance_by_user_date,
    get_user_by_employee_id
)
from services.attendance_service import get_attendance_summary

st.title("📊 View Attendance Records")

st.markdown("""
View and analyze attendance history with filtering and export options.
""")

# Filters
st.sidebar.subheader("🔍 Filters")

# User filter
users = get_all_active_users()

if not users:
    st.warning("⚠️ No users registered yet.")
    st.stop()

user_options = ["All Users"] + [f"{u['employee_id']} - {u['name']}" for u in users]
selected_user = st.sidebar.selectbox("Select User", user_options)

# Date filter
date_range_type = st.sidebar.radio("Date Range", 
                                   ["Last 7 Days", "Last 30 Days", "Custom Range", "All Time"])

if date_range_type == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", value=date.today())
elif date_range_type == "Last 7 Days":
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
elif date_range_type == "Last 30 Days":
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
else:  # All Time
    start_date = None
    end_date = None

# Status filter
status_filter = st.sidebar.multiselect("Status", 
                                      ["present", "half-day", "early-exit"],
                                      default=["present", "half-day", "early-exit"])

# Fetch records
if selected_user == "All Users":
    # Get records for all users
    all_records = []
    
    for user in users:
        records = get_attendance_history(user['id'], start_date, end_date)
        
        for record in records:
            if record['status'] in status_filter:
                all_records.append({
                    "Date": record['date'],
                    "Employee ID": user['employee_id'],
                    "Name": user['name'],
                    "Punch In": record['punch_in_time'],
                    "Punch Out": record['punch_out_time'] or "Active",
                    "Duration (hrs)": f"{record['duration']:.2f}" if record['duration'] else "In Progress",
                    "Status": record['status'].title()
                })
    
    records_data = all_records
else:
    # Extract employee ID
    employee_id = selected_user.split(" - ")[0]
    user = get_user_by_employee_id(employee_id)
    
    records = get_attendance_history(user['id'], start_date, end_date)
    
    # Convert to display format
    records_data = []
    for record in records:
        if record['status'] in status_filter:
            records_data.append({
                "Date": record['date'],
                "Punch In": record['punch_in_time'],
                "Punch Out": record['punch_out_time'] or "Active",
                "Duration (hrs)": f"{record['duration']:.2f}" if record['duration'] else "In Progress",
                "Status": record['status'].title()
            })

# Display statistics
if records_data:
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(records_data)
    present_count = len([r for r in records_data if r['Status'] == 'Present'])
    half_day_count = len([r for r in records_data if r['Status'] == 'Half-Day'])
    
    # Calculate total hours
    total_hours = 0
    for r in records_data:
        if r['Duration (hrs)'] != "In Progress":
            try:
                total_hours += float(r['Duration (hrs)'])
            except:
                pass
    
    with col1:
        st.metric("Total Records", total_records)
    
    with col2:
        st.metric("Present", present_count)
    
    with col3:
        st.metric("Half Days", half_day_count)
    
    with col4:
        st.metric("Total Hours", f"{total_hours:.2f}")
    
    st.markdown("---")

# Display records
st.subheader("📋 Attendance Records")

if records_data:
    # Convert to DataFrame
    df = pd.DataFrame(records_data)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export (requires openpyxl)
        try:
            from io import BytesIO
            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("Install openpyxl for Excel export: pip install openpyxl")
    
    with col3:
        # JSON export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="🗂️ Download as JSON",
            data=json_data,
            file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Visualizations
    if len(records_data) > 1:
        st.markdown("---")
        st.subheader("📈 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Daily Hours", "Status Distribution", "Trends"])
        
        with tab1:
            # Daily hours chart
            if 'Duration (hrs)' in df.columns:
                chart_df = df[df['Duration (hrs)'] != "In Progress"].copy()
                if not chart_df.empty:
                    chart_df['Duration (hrs)'] = pd.to_numeric(chart_df['Duration (hrs)'])
                    chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                    
                    daily_hours = chart_df.groupby('Date')['Duration (hrs)'].sum().reset_index()
                    
                    st.line_chart(daily_hours.set_index('Date'))
                    st.caption("Total working hours per day")
        
        with tab2:
            # Status distribution
            status_counts = df['Status'].value_counts()
            st.bar_chart(status_counts)
            st.caption("Distribution of attendance status")
        
        with tab3:
            # Attendance trend
            df_trend = df.copy()
            df_trend['Date'] = pd.to_datetime(df_trend['Date'])
            df_trend = df_trend.sort_values('Date')
            
            daily_count = df_trend.groupby('Date').size().reset_index(name='Count')
            
            st.line_chart(daily_count.set_index('Date'))
            st.caption("Number of attendance records per day")

else:
    st.info("📭 No records found for the selected filters.")
    st.write("Try adjusting the filters or selecting a different date range.")

# Individual user summary
if selected_user != "All Users":
    st.markdown("---")
    st.subheader("📊 User Summary")
    
    employee_id = selected_user.split(" - ")[0]
    user = get_user_by_employee_id(employee_id)
    
    # Calculate days based on filter
    if start_date and end_date:
        days = (end_date - start_date).days
    elif date_range_type == "Last 7 Days":
        days = 7
    elif date_range_type == "Last 30 Days":
        days = 30
    else:
        days = 365  # Default for all time
    
    summary = get_attendance_summary(user['id'], days=days)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days Worked", summary['total_days'])
        st.metric("Present Days", summary['present_days'])
    
    with col2:
        st.metric("Half Days", summary['half_days'])
        st.metric("Total Hours", f"{summary['total_hours']:.2f}h")
    
    with col3:
        st.metric("Average Hours/Day", f"{summary['average_hours']:.2f}h")
        attendance_rate = (summary['total_days'] / days * 100) if days > 0 else 0
        st.metric("Attendance Rate", f"{attendance_rate:.1f}%")

# Quick actions
with st.expander("⚡ Quick Actions", expanded=False):
    st.markdown("""
    **Available Actions:**
    
    - **Filter by User**: Select specific user from sidebar
    - **Filter by Date**: Choose date range from sidebar
    - **Filter by Status**: Select attendance status from sidebar
    - **Export Data**: Download records as CSV, Excel, or JSON
    - **View Charts**: Analyze data with visualizations
    
    **Tips:**
    - Use "All Users" to see organization-wide attendance
    - Select custom date range for specific analysis
    - Export data for further analysis in Excel or other tools
    """)
