# SkogVegPlanering

Unified QGIS plugin for forest road planning with batch PDF import, validation, and cost distribution analysis.

## Features

- **Batch PDF Import:** Convert scanned road maps to GIS data automatically
- **Road Validation:** Check slope, curve radius, and road standards
- **Cost Distribution:** Calculate property-based cost shares
- **Cable Way Planning:** Auto-generate cable way stations at intervals
- **Interactive Editor:** Edit roads, stations, and dump sites

## Installation

1. Clone repository:
   ```
   git clone https://github.com/nikko617/SkogVegPlanering.git
   ```

2. Place in QGIS plugins folder:
   ```
   ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/SkogVegPlanering
   ```

3. Enable in QGIS: Plugins → Manage and Install Plugins → SkogVegPlanering

## Usage

1. Open QGIS project
2. Launch SkogVegPlanering from toolbar
3. Use "Batch Import Wizard" to import PDFs
4. Validate and edit roads
5. Export results to GeoPackage

## Development

See STEG 2, STEG 3, etc. for implementation roadmap.

## License

MIT
