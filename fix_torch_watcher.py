"""
Aggressive fix for torch.classes Streamlit watcher issue.
This module patches the problem before Streamlit can encounter it.
Import this FIRST in your main app.
"""
import sys
import logging
import os

# Suppress torch-related warnings immediately
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch"
os.environ["TORCH_LOGS"] = "ERROR"

# Optimize Streamlit performance
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

logger = logging.getLogger(__name__)

# Suppress torch logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.classes").setLevel(logging.ERROR)

def fix_torch_classes_issue():
    """Comprehensively fix the torch._classes.__path__ issue for Streamlit."""
    
    try:
        # Method 1: Patch torch if already imported
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            if hasattr(torch, '_classes'):
                # Create a safe mock path that won't cause Streamlit watcher issues
                class SafePathMock:
                    def __init__(self):
                        self._path = []
                    
                    def __iter__(self):
                        return iter([])
                    
                    def __getattr__(self, name):
                        if name == '_path':
                            return []
                        return lambda *args, **kwargs: None
                
                # Replace the problematic __path__ attribute
                torch._classes.__path__ = SafePathMock()  # type: ignore
                logger.info("✅ Patched existing torch._classes.__path__")
        
        # Method 2: Prevent the issue by intercepting torch import
        class TorchImportWrapper:
            def __init__(self, original_import):
                self.original_import = original_import
            
            def __call__(self, name, *args, **kwargs):
                module = self.original_import(name, *args, **kwargs)
                
                # Patch torch._classes immediately upon import
                if name == 'torch' or (hasattr(module, '__name__') and module.__name__ == 'torch'):
                    if hasattr(module, '_classes') and hasattr(module._classes, '__path__'):
                        class SafePathMock:
                            def __init__(self):
                                self._path = []
                            def __iter__(self):
                                return iter([])
                            def __getattr__(self, name):
                                if name == '_path':
                                    return []
                                return lambda *args, **kwargs: None
                        
                        module._classes.__path__ = SafePathMock()  # type: ignore
                        logger.info("✅ Patched torch._classes.__path__ on import")
                
                return module
        
        # Install the import wrapper
        if not hasattr(__builtins__, '_original_import_patched'):
            original_import = __builtins__['__import__']
            __builtins__['__import__'] = TorchImportWrapper(original_import)
            __builtins__['_original_import_patched'] = True
            logger.info("✅ Installed torch import wrapper")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not patch torch._classes: {e}")
        return False

def disable_streamlit_file_watcher():
    """Additional method to disable Streamlit file watching entirely."""
    
    try:
        # Set environment variables that Streamlit checks
        import os
        os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
        os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
        
        logger.info("✅ Set Streamlit environment variables to disable file watching")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not set Streamlit environment variables: {e}")
        return False

def apply_comprehensive_fix():
    """Apply all available fixes for the torch.classes issue."""
    
    logger.info("🔧 Applying comprehensive torch.classes fix...")
    
    fixes_applied = []
    
    # Fix 1: Patch torch._classes
    if fix_torch_classes_issue():
        fixes_applied.append("torch_patch")
    
    # Fix 2: Disable file watcher
    if disable_streamlit_file_watcher():
        fixes_applied.append("watcher_disable")
    
    if fixes_applied:
        logger.info(f"✅ Applied fixes: {', '.join(fixes_applied)}")
    else:
        logger.warning("⚠️ No fixes could be applied")
    
    return len(fixes_applied) > 0

# Apply fixes immediately when this module is imported
apply_comprehensive_fix() 