# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `fetch-corpus` downloads the FastAPI 0.165.0 docs, verifies the archive checksum and keeps the English documentation pages (#7).
- `ingest` cuts pages into chunks by heading section or into fixed-size windows with overlap, without ever splitting a code block (#7).
