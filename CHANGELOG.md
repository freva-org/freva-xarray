# Changelog

All notable changes to this project will be documented in this file.
## [v2605.0.0]
## Fixed
- `FileNotFoundError` for missing local files instead of misleading "cannot detect format"
- `storage_options` leaking into posix backends (cfgrib, scipy, netcdf4, rasterio), causing TypeError
- cfgrib .idx files now written to prism cache dir instead of next to the original file
## Removed
- line-clearing ANSI escape machinery unreliable across terminals and notebooks
- logging.basicConfig call from library code
## Changed
- detection log messages from INFO to DEBUG
## Added
- `XARRAY_PRISM_LOG_LEVEL` env var to control log verbosity

## [v2603.0.0]
## Fixed
- an issue regading passing the storage_options to aiohttp
### Added
- Cache cleanup strategy to evict automatically

## [v2602.1.0]
### Added
- Initial release of the project.


# Template:
## [Unreleased]

### Added
- New feature X.
- New feature Y.

### Changed
- Improved performance in component A.
- Updated dependency B to version 2.0.0.

### Fixed
- Fixed issue causing application crash on startup.
- Fixed bug preventing users from logging in.