import os

def get_library_path():
    """Returns the path to the compiled shared library."""
    base_path = os.path.dirname(__file__)
    # Try different extensions
    for ext in ['.so', '.dll', '.dylib']:
        lib_path = os.path.join(base_path, 'libmamba_ops' + ext)
        if os.path.exists(lib_path):
            return lib_path
    
    # Fallback/Debug
    return os.path.join(base_path, 'libmamba_ops.so')

def register_custom_ops(session_options):
    """Registers the custom ops library with the session options."""
    lib_path = get_library_path()
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Custom ops library not found at {lib_path}")
    
    session_options.register_custom_ops_library(lib_path)
