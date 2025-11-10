import pkgutil
import importlib
print('autorom' in [m.name for m in pkgutil.iter_modules()])
try:
    import autorom
    print('autorom imported:', autorom)
    if hasattr(autorom, 'install_roms'):
        print('autorom.install_roms available')
    else:
        print('autorom has no install_roms attribute; listing dir:')
        print(dir(autorom))
except Exception as e:
    print('import autorom failed:', e)
