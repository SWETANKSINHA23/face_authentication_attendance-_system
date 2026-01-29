"""
Test script for Face Registration Module
Demonstrates the complete registration workflow
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.user_service import register_user
from models.database import init_database, get_all_active_users
import config


def main():
    """Main registration demo"""
    
    print("\n" + "="*70)
    print(" FACE AUTHENTICATION ATTENDANCE SYSTEM - REGISTRATION MODULE")
    print("="*70 + "\n")
    
    # Initialize database
    print("Initializing database...")
    init_database()
    print("✓ Database ready\n")
    
    # Get user input
    print("Enter user details:")
    print("-" * 40)
    
    employee_id = input("Employee ID: ").strip()
    if not employee_id:
        print("Error: Employee ID is required")
        return
    
    name = input("Full Name: ").strip()
    if not name:
        print("Error: Name is required")
        return
    
    email = input("Email (optional): ").strip() or None
    department = input("Department (optional): ").strip() or None
    
    # Confirm registration
    print("\n" + "-" * 40)
    print("Registration Details:")
    print(f"  Employee ID: {employee_id}")
    print(f"  Name: {name}")
    print(f"  Email: {email or 'N/A'}")
    print(f"  Department: {department or 'N/A'}")
    print(f"  Samples to capture: {config.REGISTRATION_SAMPLES}")
    print("-" * 40)
    
    confirm = input("\nProceed with registration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Registration cancelled")
        return
    
    # Perform registration
    result = register_user(
        employee_id=employee_id,
        name=name,
        email=email,
        department=department,
        show_preview=True
    )
    
    # Display result
    print("\n" + "="*70)
    if result.success:
        print("✓ REGISTRATION SUCCESSFUL")
        print(f"  User ID: {result.user_id}")
        print(f"  Images Captured: {result.captured_images}")
        print(f"  Message: {result.message}")
    else:
        print("✗ REGISTRATION FAILED")
        print(f"  Message: {result.message}")
        print(f"  Images Captured: {result.captured_images}")
        
        if result.issues:
            print("\n  Issues encountered:")
            for issue in result.issues:
                print(f"    - {issue}")
    
    print("="*70 + "\n")
    
    # Show all registered users
    print("Currently registered users:")
    print("-" * 40)
    users = get_all_active_users()
    
    if users:
        for user in users:
            print(f"  [{user['employee_id']}] {user['name']}")
            if user['department']:
                print(f"    Department: {user['department']}")
    else:
        print("  No users registered yet")
    
    print("-" * 40)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRegistration cancelled by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
