# -*- coding: utf-8 -*-
"""
SkogVegPlanering – Forest Road Planning QGIS Plugin
Version 1.0.0
"""

def classFactory(iface):
    """Load the main plugin class."""
    from .plugin import SkogVegPlaneringPlugin
    return SkogVegPlaneringPlugin(iface)
