
def dpi_fix():
    from ctypes import windll
    user32 = windll.user32
    user32.SetProcessDPIAware()
