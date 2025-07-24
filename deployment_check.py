#!/usr/bin/env python3
"""
Deployment diagnostics for Streamlit Cloud issues.
Run this to check for common problems that cause deployment failures.
"""
import os
import sys
from pathlib import Path
import subprocess

def check_file_permissions():
    """Check if all files have proper permissions."""
    print("🔧 Checking file permissions...")
    
    issues = []
    
    # Check key files
    key_files = [
        'requirements.txt',
        'src/web/streamlit_app.py', 
        '.streamlit/config.toml',
        'README.md'
    ]
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            try:
                # Try to read file
                with open(path, 'r') as f:
                    content = f.read(100)  # Read first 100 chars
                print(f"✅ {file_path} - readable")
            except Exception as e:
                print(f"❌ {file_path} - permission issue: {e}")
                issues.append(f"Cannot read {file_path}")
        else:
            print(f"⚠️ {file_path} - missing")
            issues.append(f"Missing {file_path}")
    
    return issues

def check_git_status():
    """Check git repository status."""
    print("\n🔧 Checking git status...")
    
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("⚠️ Uncommitted changes detected:")
            print(result.stdout)
            return ["Uncommitted changes in repository"]
        else:
            print("✅ Repository is clean")
            
        # Check remote status
        result = subprocess.run(['git', 'status', '-uno'], 
                              capture_output=True, text=True, check=True)
        
        if "ahead" in result.stdout or "behind" in result.stdout:
            print("⚠️ Repository is not in sync with remote:")
            print(result.stdout)
            return ["Repository not synced with remote"]
        else:
            print("✅ Repository is synced with remote")
            
        return []
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        return ["Git repository issues"]
    except FileNotFoundError:
        print("❌ Git not found")
        return ["Git not available"]

def check_requirements():
    """Check requirements.txt for issues."""
    print("\n🔧 Checking requirements.txt...")
    
    issues = []
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        print(f"✅ Found {len(lines)} requirements")
        
        # Check for common problematic packages
        problematic = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Check for version conflicts
                if 'torch' in line and 'cpu' not in line:
                    problematic.append(f"PyTorch GPU version may cause issues: {line}")
                elif 'tensorflow' in line:
                    problematic.append(f"TensorFlow may cause conflicts: {line}")
                elif line.count('==') > 1:
                    problematic.append(f"Invalid version specification: {line}")
        
        if problematic:
            print("⚠️ Potentially problematic requirements:")
            for p in problematic:
                print(f"  - {p}")
            issues.extend(problematic)
        else:
            print("✅ No obvious requirement issues")
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        issues.append("Missing requirements.txt")
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        issues.append("Cannot read requirements.txt")
    
    return issues

def check_streamlit_config():
    """Check Streamlit configuration."""
    print("\n🔧 Checking Streamlit configuration...")
    
    issues = []
    
    config_path = Path('.streamlit/config.toml')
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            print("✅ Found Streamlit config")
            
            # Check for our torch fix
            if 'fileWatcherType = "none"' in content:
                print("✅ File watcher disabled in config")
            else:
                print("⚠️ File watcher not disabled in config")
                issues.append("File watcher not properly configured")
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            issues.append("Cannot read Streamlit config")
    else:
        print("⚠️ No Streamlit config found")
        issues.append("Missing Streamlit config")
    
    return issues

def check_main_app():
    """Check main application file."""
    print("\n🔧 Checking main application...")
    
    issues = []
    
    app_path = Path('src/web/streamlit_app.py')
    if app_path.exists():
        try:
            with open(app_path, 'r') as f:
                content = f.read()
            
            print("✅ Found main app file")
            
            # Check for our torch fix import
            if 'import fix_torch_watcher' in content:
                print("✅ Torch fix import found")
            else:
                print("⚠️ Torch fix import missing")
                issues.append("Torch fix not imported in main app")
            
            # Check for basic imports
            required_imports = ['streamlit', 'sys', 'os']
            for imp in required_imports:
                if f'import {imp}' in content:
                    print(f"✅ {imp} imported")
                else:
                    print(f"⚠️ {imp} not imported")
                    issues.append(f"{imp} not imported")
            
        except Exception as e:
            print(f"❌ Error reading main app: {e}")
            issues.append("Cannot read main application")
    else:
        print("❌ Main app file not found")
        issues.append("Missing main application file")
    
    return issues

def main():
    """Run all deployment checks."""
    print("🚀 Streamlit Cloud Deployment Diagnostics")
    print("=" * 60)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_file_permissions())
    all_issues.extend(check_git_status())
    all_issues.extend(check_requirements())
    all_issues.extend(check_streamlit_config())
    all_issues.extend(check_main_app())
    
    # Summary
    print("\n" + "=" * 60)
    if all_issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        print(f"\n🔧 Fix these {len(all_issues)} issues and try deploying again.")
    else:
        print("✅ ALL CHECKS PASSED!")
        print("🎉 Repository should deploy successfully to Streamlit Cloud.")
    
    print("\n💡 If deployment still fails after fixing issues:")
    print("  1. Try restarting the Streamlit Cloud app")
    print("  2. Check Streamlit Cloud status page")
    print("  3. Contact Streamlit support")

if __name__ == "__main__":
    main() 