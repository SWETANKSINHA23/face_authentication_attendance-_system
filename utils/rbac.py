"""
Role-Based Access Control Module
Manages user roles and permissions
"""

from enum import Enum
from typing import List, Optional
import streamlit as st


class Role(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"


class Permission(Enum):
    """System permissions"""
    # User management
    CREATE_USER = "create_user"
    DELETE_USER = "delete_user"
    UPDATE_USER = "update_user"
    VIEW_ALL_USERS = "view_all_users"
    
    # Attendance
    MARK_ATTENDANCE = "mark_attendance"
    VIEW_OWN_ATTENDANCE = "view_own_attendance"
    VIEW_ALL_ATTENDANCE = "view_all_attendance"
    EDIT_ATTENDANCE = "edit_attendance"
    
    # Reports
    VIEW_REPORTS = "view_reports"
    EXPORT_DATA = "export_data"
    
    # System
    VIEW_DASHBOARD = "view_dashboard"
    MANAGE_SETTINGS = "manage_settings"
    VIEW_LOGS = "view_logs"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        # All permissions
        Permission.CREATE_USER,
        Permission.DELETE_USER,
        Permission.UPDATE_USER,
        Permission.VIEW_ALL_USERS,
        Permission.MARK_ATTENDANCE,
        Permission.VIEW_OWN_ATTENDANCE,
        Permission.VIEW_ALL_ATTENDANCE,
        Permission.EDIT_ATTENDANCE,
        Permission.VIEW_REPORTS,
        Permission.EXPORT_DATA,
        Permission.VIEW_DASHBOARD,
        Permission.MANAGE_SETTINGS,
        Permission.VIEW_LOGS,
    ],
    Role.MANAGER: [
        # Most permissions except critical ones
        Permission.VIEW_ALL_USERS,
        Permission.MARK_ATTENDANCE,
        Permission.VIEW_OWN_ATTENDANCE,
        Permission.VIEW_ALL_ATTENDANCE,
        Permission.VIEW_REPORTS,
        Permission.EXPORT_DATA,
        Permission.VIEW_DASHBOARD,
    ],
    Role.USER: [
        # Basic user permissions
        Permission.MARK_ATTENDANCE,
        Permission.VIEW_OWN_ATTENDANCE,
    ],
    Role.GUEST: [
        # Read-only access
        Permission.VIEW_REPORTS,
    ]
}


def get_role_permissions(role: Role) -> List[Permission]:
    """
    Get list of permissions for a role
    
    Args:
        role: User role
    
    Returns:
        List of permissions
    """
    return ROLE_PERMISSIONS.get(role, [])


def has_permission(role: Role, permission: Permission) -> bool:
    """
    Check if role has specific permission
    
    Args:
        role: User role
        permission: Permission to check
    
    Returns:
        True if role has permission
    """
    return permission in get_role_permissions(role)


def require_permission(permission: Permission):
    """
    Decorator to require permission for a function
    Use in Streamlit pages to restrict access
    
    Args:
        permission: Required permission
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get current user role from session
            current_role = st.session_state.get('user_role', Role.GUEST)
            
            if has_permission(current_role, permission):
                return func(*args, **kwargs)
            else:
                st.error(f"⛔ Access Denied: You don't have permission to {permission.value}")
                st.info(f"Your role: {current_role.value}")
                st.info(f"Required permission: {permission.value}")
                st.stop()
        
        return wrapper
    return decorator


def init_session_role(default_role: Role = Role.USER):
    """
    Initialize user role in Streamlit session
    
    Args:
        default_role: Default role if not set
    """
    if 'user_role' not in st.session_state:
        st.session_state.user_role = default_role


def set_user_role(role: Role):
    """
    Set current user role
    
    Args:
        role: Role to set
    """
    st.session_state.user_role = role


def get_current_role() -> Role:
    """
    Get current user role
    
    Returns:
        Current role
    """
    return st.session_state.get('user_role', Role.GUEST)


def show_role_selector():
    """
    Display role selector in sidebar (for demo/development)
    In production, this would be replaced with actual authentication
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Access Control")
    
    current_role = get_current_role()
    
    role_options = {
        "Admin (Full Access)": Role.ADMIN,
        "Manager (Reports & Viewing)": Role.MANAGER,
        "User (Basic Access)": Role.USER,
        "Guest (Read-Only)": Role.GUEST
    }
    
    # Find current selection
    current_selection = None
    for label, role in role_options.items():
        if role == current_role:
            current_selection = label
            break
    
    selected = st.sidebar.selectbox(
        "Select Role (Demo Mode)",
        options=list(role_options.keys()),
        index=list(role_options.keys()).index(current_selection) if current_selection else 0,
        help="In production, this would be determined by login credentials"
    )
    
    new_role = role_options[selected]
    
    if new_role != current_role:
        set_user_role(new_role)
        st.rerun()
    
    # Show current permissions
    with st.sidebar.expander("📋 Current Permissions"):
        permissions = get_role_permissions(current_role)
        for perm in permissions:
            st.write(f"✅ {perm.value.replace('_', ' ').title()}")


def show_access_badge():
    """Display current role badge"""
    role = get_current_role()
    
    role_colors = {
        Role.ADMIN: "🔴",
        Role.MANAGER: "🟡",
        Role.USER: "🟢",
        Role.GUEST: "⚪"
    }
    
    color = role_colors.get(role, "⚪")
    
    st.sidebar.markdown(f"{color} **Role:** {role.value.title()}")


# Simple authentication (for demo purposes)
def simple_login() -> Optional[Role]:
    """
    Simple login interface
    In production, replace with proper authentication
    
    Returns:
        Role if authenticated, None otherwise
    """
    st.title("🔐 Login")
    
    # Demo credentials
    credentials = {
        "admin": ("admin123", Role.ADMIN),
        "manager": ("manager123", Role.MANAGER),
        "user": ("user123", Role.USER),
    }
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Try: admin, manager, or user")
        password = st.text_input("Password", type="password", placeholder="password123")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username in credentials:
                stored_password, role = credentials[username]
                if password == stored_password:
                    st.success(f"✅ Logged in as {role.value.title()}")
                    return role
                else:
                    st.error("❌ Invalid password")
            else:
                st.error("❌ Invalid username")
    
    # Show demo credentials
    with st.expander("📝 Demo Credentials"):
        st.write("**Admin:**")
        st.code("Username: admin\nPassword: admin123")
        st.write("**Manager:**")
        st.code("Username: manager\nPassword: manager123")
        st.write("**User:**")
        st.code("Username: user\nPassword: user123")
    
    return None


if __name__ == "__main__":
    print("Role-Based Access Control Module")
    print("\nRoles:")
    for role in Role:
        print(f"  - {role.value.title()}: {len(ROLE_PERMISSIONS[role])} permissions")
